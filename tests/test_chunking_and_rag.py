import os

os.environ.setdefault("EMBEDDING_PROVIDER", "tfidf")  # avoid network model downloads in CI/tests
os.environ.setdefault("LLM_PROVIDER", "offline")

from app.pipeline.rag import HybridRetriever, chunk_text, rerank  # noqa: E402


def test_chunk_text_respects_chunk_size():
    text = "Sentence one. " * 200  # long text
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 260 for c in chunks)  # allow slack for overlap prefix


def test_chunk_text_short_text_single_chunk():
    text = "Just one short paragraph."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_hybrid_retriever_finds_relevant_chunk():
    docs = [
        "The quarterly financial report shows MRR growth of 15 percent.",
        "Our office is located in downtown San Francisco near the park.",
        "The cafeteria menu changes every Monday and Thursday.",
        "Burn rate increased due to higher payroll and cloud infrastructure costs.",
    ]
    chunks = chunk_text("\n\n".join(docs), chunk_size=1000, overlap=0)
    retriever = HybridRetriever(chunks)
    results = retriever.retrieve("What is the company's burn rate?", pool_size=4)
    assert results, "retriever returned no candidates"
    top_chunk_text = results[0][0].text
    assert "burn rate" in top_chunk_text.lower() or "Burn rate" in top_chunk_text

    reranked = rerank("What is the company's burn rate?", results, top_k=2)
    assert 1 <= len(reranked) <= 2


def test_retriever_handles_empty_corpus_gracefully():
    retriever = HybridRetriever([])
    assert retriever.retrieve("anything") == []
