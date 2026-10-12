"""
Central configuration for VentureAgent.

Everything is loaded from environment variables (via python-dotenv), never
hardcoded, per the project's secret-management requirement. Sensible
defaults make the pipeline runnable end-to-end with zero keys configured
(LLM_PROVIDER=offline, EMBEDDING_PROVIDER falls back to tfidf, email/sheet
actions fall back to mock/local-file mode).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "offline").lower())
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)

    # Embeddings
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").lower())
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    embedding_model_local_path: str | None = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_LOCAL_PATH") or None)

    # Reranking
    rerank_provider: str = field(default_factory=lambda: os.getenv("RERANK_PROVIDER", "none").lower())
    cohere_api_key: str | None = field(default_factory=lambda: os.getenv("COHERE_API_KEY") or None)

    # Google Sheets
    google_service_account_json: str | None = field(default_factory=lambda: os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or None)

    # Market data
    newsapi_key: str | None = field(default_factory=lambda: os.getenv("NEWSAPI_KEY") or None)

    # Email
    smtp_host: str | None = field(default_factory=lambda: os.getenv("SMTP_HOST") or None)
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_username: str | None = field(default_factory=lambda: os.getenv("SMTP_USERNAME") or None)
    smtp_password: str | None = field(default_factory=lambda: os.getenv("SMTP_PASSWORD") or None)
    smtp_from_addr: str = field(default_factory=lambda: os.getenv("SMTP_FROM_ADDR", "ventureagent@example.com"))
    investment_team_emails: str = field(default_factory=lambda: os.getenv("INVESTMENT_TEAM_EMAILS", "team@example.com"))
    sendgrid_api_key: str | None = field(default_factory=lambda: os.getenv("SENDGRID_API_KEY") or None)

    # Pipeline tuning
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_TOP_K", "3")))
    mrr_discrepancy_tolerance_pct: float = field(default_factory=lambda: float(os.getenv("MRR_DISCREPANCY_TOLERANCE_PCT", "15")))
    gradio_concurrency_limit: int = field(default_factory=lambda: int(os.getenv("GRADIO_CONCURRENCY_LIMIT", "5")))


settings = Settings()
