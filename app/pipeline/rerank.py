from __future__ import annotations
from typing import List
from app.config import settings
from app.schemas import Chunk
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

def rerank_chunks(query: str, chunks: List[Chunk], top_k: int | None = None) -> List[Chunk]:
    if top_k is None:
        top_k = settings.rerank_top_k

    if not chunks:
        return []

    if settings.rerank_provider == "cohere" and settings.cohere_api_key:
        try:
            import cohere
            co = cohere.Client(api_key=settings.cohere_api_key)
            doc_texts = [c.text for c in chunks]
            
            response = co.rerank(
                model=settings.rerank_model,
                query=query,
                documents=doc_texts,
                top_n=min(top_k, len(chunks))
            )
            
            reranked_chunks = [chunks[result.index] for result in response.results]
            return reranked_chunks
        except Exception as exc:
            logger.warning("Cohere reranking failed (%s); returning original chunk ranking.", exc)

    # Default / Fallback: return top_k without re-ordering
    return chunks[:top_k]