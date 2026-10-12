"""Embedding backend abstraction.

Tries `sentence-transformers` (per the spec's `all-MiniLM-L6-v2` suggestion)
first. If the model/library can't be loaded (no internet access to download
weights, or the package isn't installed) it falls back to a TF-IDF + SVD
dense-vector approximation, logs a low-confidence warning, and keeps the
pipeline running — the same fallback philosophy the spec asks for on the
data-ingestion side.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from app.config import settings
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

EmbedFn = Callable[[List[str]], np.ndarray]


class TfidfEmbedder:
    """Lightweight, dependency-light fallback embedder. Deterministic and
    needs no model download, so it also doubles as the offline/test embedder."""

    def __init__(self, n_components: int = 128):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.n_components = n_components
        self._svd = None
        self._TruncatedSVD = TruncatedSVD
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        sparse = self.vectorizer.fit_transform(corpus)
        n_comp = max(1, min(self.n_components, sparse.shape[1] - 1, sparse.shape[0] - 1))
        self._svd = self._TruncatedSVD(n_components=n_comp, random_state=42)
        self._svd.fit(sparse)
        self._fitted = True

    def __call__(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        sparse = self.vectorizer.transform(texts)
        dense = self._svd.transform(sparse)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense / norms).astype("float32")


def _try_cohere_embedder(model_name: str) -> EmbedFn | None:
    if not settings.cohere_api_key:
        logger.warning("Cohere API key is missing for embeddings; falling back.")
        return None
    try:
        import cohere
        co = cohere.Client(settings.cohere_api_key)

        def _embed(texts: List[str]) -> np.ndarray:
            response = co.embed(
                texts=texts,
                model=model_name,
                input_type="search_document",
                embedding_types=["float"]
            )
            vecs = np.array(response.embeddings.float, dtype="float32")
            # Normalize vectors
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (vecs / norms).astype("float32")

        _embed(["healthcheck"])
        return _embed
    except Exception as exc:
        logger.warning("Cohere embedder failed (%s); falling back.", exc)
        return None

def _try_sentence_transformer(model_name_or_path: str) -> EmbedFn | None:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name_or_path)

        def _embed(texts: List[str]) -> np.ndarray:
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")

        # Smoke-test it immediately; if weights can't be fetched/loaded this raises
        # (ImportError: package missing; OSError/HFValidationError: no local cache
        # and no internet to download; ValueError: bad path/model id).
        _embed(["healthcheck"])
        return _embed
    except (ImportError, OSError, ValueError) as exc:
        logger.warning("sentence-transformers model '%s' unavailable (%s); falling back.", model_name_or_path, exc)
        return None


def get_embedder() -> tuple[EmbedFn, str]:
    """Returns (embed_fn, provider_used).

    Resolution order when EMBEDDING_PROVIDER=sentence-transformers:
      1. EMBEDDING_MODEL_LOCAL_PATH, if set — loads straight from disk, so
         real semantic embeddings work with zero runtime internet access
         (e.g. a model pre-baked into a Docker image or an HF Spaces
         persistent volume). This is the preferred path for "no internet,
         but still want retrieval quality better than TF-IDF".
      2. EMBEDDING_MODEL by name — downloads from the Hugging Face Hub on
         first use if there's no local cache.
      3. TF-IDF + SVD fallback — always available, no download required.
    """
    if settings.embedding_provider == "cohere":
        fn = _try_cohere_embedder(settings.embedding_model or "embed-english-v3.0")
        if fn is not None:
            return fn, "cohere"

    if settings.embedding_provider == "sentence-transformers":
        if settings.embedding_model_local_path:
            fn = _try_sentence_transformer(settings.embedding_model_local_path)
            if fn is not None:
                return fn, "sentence-transformers-local"
        fn = _try_sentence_transformer(settings.embedding_model)
        if fn is not None:
            return fn, "sentence-transformers"

    # Explicit "tfidf" setting, or fallback from above.
    embedder = TfidfEmbedder()
    return embedder, "tfidf"
