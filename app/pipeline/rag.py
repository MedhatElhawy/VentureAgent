from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from rank_bm25 import BM25Okapi
from app.config import settings
from app.pipeline.embeddings import EmbedFn, get_embedder
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    id: int
    text: str


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[Chunk]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    separators = ["\n\n", "\n", ". ", " "]

    def _split(t: str) -> List[str]:
        if len(t) <= chunk_size:
            return [t] if t.strip() else []
        for sep in separators:
            if sep in t:
                parts, buf, out = t.split(sep), "", []
                for p in parts:
                    candidate = (buf + sep + p) if buf else p
                    if len(candidate) <= chunk_size:
                        buf = candidate
                    else:
                        if buf:
                            out.append(buf)
                        buf = p
                if buf:
                    out.append(buf)
                final: List[str] = []
                for piece in out:
                    if len(piece) <= chunk_size:
                        final.append(piece)
                    else:
                        final.extend(piece[i:i + chunk_size] for i in range(0, len(piece), chunk_size))
                return final
        return [t[i:i + chunk_size] for i in range(0, len(t), chunk_size)]

    raw_chunks = _split(text.strip())

    overlapped: List[str] = []
    for i, c in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            tail = raw_chunks[i - 1][-overlap:]
            c = tail + c
        overlapped.append(c)

    return [Chunk(id=i, text=c) for i, c in enumerate(overlapped) if c.strip()]


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], embed_fn: EmbedFn | None = None, embedding_provider: str | None = None):
        self.chunks = chunks
        self.embed_fn, self.embedding_provider = (embed_fn, embedding_provider) if embed_fn else get_embedder()

        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

        self._dense_index = None
        self._dense_vectors = None
        if chunks:
            vectors = self.embed_fn([c.text for c in chunks])
            self._dense_vectors = np.asarray(vectors, dtype="float32")
            self._build_faiss(self._dense_vectors)

    def _build_faiss(self, vectors: np.ndarray) -> None:
        import faiss

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self._dense_index = index

    def retrieve(self, query: str, pool_size: int = 10) -> List[tuple[Chunk, float]]:
        if not self.chunks:
            return []

        n = len(self.chunks)
        pool_size = min(pool_size, n)

        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranked = np.argsort(-bm25_scores)[:pool_size]

        q_vec = self.embed_fn([query])
        _, dense_idx = self._dense_index.search(np.asarray(q_vec, dtype="float32"), pool_size)
        dense_ranked = dense_idx[0]

        k_rrf = 60
        fused: dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            if idx < 0:
                continue
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        return [(self.chunks[idx], score) for idx, score in ranked]


def rerank(query: str, candidates: List[tuple[Chunk, float]], top_k: int | None = None) -> List[Chunk]:
    """Cross-encoder re-ranking down to top-K."""
    top_k = top_k or settings.rerank_top_k
    if not candidates:
        return []

    if settings.rerank_provider == "cohere" and settings.cohere_api_key:
        try:
            import cohere

            co = cohere.Client(api_key=settings.cohere_api_key)
            docs = [c.text for c, _ in candidates]
            response = co.rerank(
                query=query,
                documents=docs,
                top_n=min(top_k, len(candidates)),
                model=getattr(settings, "rerank_model", "rerank-english-v3.0")
            )
            return [candidates[r.index][0] for r in response.results]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cohere rerank failed (%s); falling back to hybrid-score order.", exc)

    elif settings.rerank_provider == "flashrank":
        try:
            from flashrank import Ranker, RerankRequest

            ranker = Ranker()
            passages = [{"id": i, "text": c.text} for i, (c, _) in enumerate(candidates)]
            req = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(req)
            id_to_chunk = {i: c for i, (c, _) in enumerate(candidates)}
            return [id_to_chunk[r["id"]] for r in results[:top_k]]
        except Exception as exc:  # noqa: BLE001
            logger.warning("FlashRank rerank failed (%s); falling back to hybrid-score order.", exc)

    return [c for c, _ in candidates[:top_k]]