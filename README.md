# VentureAgent — Autonomous Startup Valuation & Due Diligence Platform

A multi-agent AI pipeline that ingests a startup's pitch deck (PDF) and financial sheet (CSV/Google Sheet), cross-references the two for discrepancies, and produces a grounded investment memo — complete with automated PDF report generation, email dispatch, and real-time dashboard updates.

Built for the Week 6 Final Project ("Developing a Multi-Agent AI Agent System"), extending the mid-term "Basic AI Knowledge Assistant" from simple QA into multi-step orchestration and automation.

---

## Key Features & Recent Improvements

- **Modularized Re-ranking (`app/pipeline/rerank.py`)**: Dedicated re-ranking supporting Cohere (`cohere>=5.0` via `rerank-english-v3.0`), FlashRank, and score-fused fallbacks.
- **Strict Citation Standard**: Grounding sources and citations are formatted as clean, non-disruptive bullet lists appended strictly at the very end of summaries and memos.
- **Smart Financial Metrics**:
  - **Runway**: Distinguishes "profitable / infinite runway" from "insufficient data" instead of collapsing both into `None`.
  - **Burn Multiple**: Uses real Net New ARR figures when available and flags approximate ratios (`*`) rather than understating burn.
  - **Annualized Burn Rate Fix**: Ensures monthly burn rate is annualized (12 × Monthly Burn) before dividing against Net New ARR.
- **Integrated Gradio & FastAPI**: Mounted into a single process/port (`/ui` + `/api/v1/...`), eliminating dual-process overhead.
- **Zero-Internet Execution**: Optional `EMBEDDING_MODEL_LOCAL_PATH` allows pre-downloaded `sentence-transformers` models to run entirely offline.
- **Flexible Data Ingestion**: Supports both standard long-format CSVs (`Metric,Value`) and wide-format SEC-style CSV filings.

---

## Why It Runs With Zero API Keys

Every external integration has a built-in **offline/mock fallback**, allowing `pytest` and the demo UI to run out of the box without external dependencies:

| Integration | Live Path | Offline / Mock Fallback (Default) |
| :--- | :--- | :--- |
| **LLM Synthesis** | Cohere, OpenAI API | Deterministic template synthesis, 100% grounded in extracted facts |
| **Embeddings** | Cohere or `sentence-transformers` (`all-MiniLM-L6-v2`) | TF-IDF + SVD dense vectors (no download required) |
| **Re-ranking** | Cohere Rerank or FlashRank | Hybrid BM25 + FAISS fused-score order |
| **Google Sheets** | `gspread` + Service Account | Local `dashboard.csv` append |
| **Email** | SendGrid or SMTP | Logged mock send |
| **Market Data** | `yfinance` or NewsAPI | Bundled `cached_market_multiples.json` |

---

## Quickstart

### 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Optional: add real API keys or leave blank for mock mode
```

### 2. Run Test Suite

```bash
# Forces offline/TF-IDF providers so no keys or network access are required
pytest tests/ -v
```

### 3. Launch App (FastAPI + Gradio UI)

```bash
# Run REST API + Interactive Docs + Dashboard from one process:
uvicorn app.main:app --reload
```

Access the application in your browser:

- **Gradio Dashboard:** http://localhost:8000/ui
- **Interactive API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/v1/health

> **Note:** Sample data is provided in `sample_data/sample_pitch_deck.pdf` and `sample_data/sample_financials.csv`. These files intentionally contain conflicting MRR figures ($50K claimed vs. $30K actual) to demonstrate discrepancy detection.

---

## Architecture & Course Objectives

| # | Course Objective | Implementation |
| :--- | :--- | :--- |
| 1 | Data Retrieval | `app/pipeline/ingestion.py` — PyMuPDF + pandas/gspread + yfinance |
| 2 | Processing & Summarization | `app/pipeline/rag.py` & `rerank.py` → `app/pipeline/analysis.py` |
| 3 | Automated Actions | `app/pipeline/actions.py` — PDF generation (reportlab), Email, Sheets |
| 4 | Workflow Orchestration | `app/pipeline/orchestrator.py` — Step-isolated pipeline with retry (tenacity) |
| 5 | Presentation | `gradio_app.py` (4-tab dashboard) + Documentation |
| 6 | Testing & Evaluation | `app/eval/evaluator.py` (LLM-as-judge) + `tests/` suite |

### Repository Layout

```text
ventureagent/
├── app/
│   ├── config.py                     # Environment-driven settings
│   ├── schemas.py                    # Shared Pydantic schemas
│   ├── main.py                       # FastAPI entry point
│   ├── eval/                         # Groundedness evaluation
│   ├── pipeline/                     # RAG, Ingestion, Valuation & Orchestration
│   └── utils/                        # Logging & cached data helpers
├── sample_data/                      # Synthetic PDF and CSV test data
├── tests/                            # Offline test suite
├── .env.example                      # Example environment configurations
├── gradio_app.py                     # Standalone / Mounted UI
└── requirements.txt                  # Python dependencies
```

---

## Configuration

Core variables supported in `.env`:

```env
# ---- LLM Synthesis ----
LLM_PROVIDER=cohere            # cohere | openai | anthropic | offline
COHERE_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# ---- Embeddings ----
EMBEDDING_PROVIDER=cohere      # cohere | sentence-transformers | tfidf
EMBEDDING_MODEL=embed-english-v3.0
EMBEDDING_MODEL_LOCAL_PATH=    # Path to local sentence-transformers model directory

# ---- Re-ranking ----
RERANK_PROVIDER=cohere         # cohere | flashrank | none

# ---- External Services (Optional) ----
NEWSAPI_KEY=
GOOGLE_SERVICE_ACCOUNT_JSON=
SENDGRID_API_KEY=
SMTP_HOST=
```

---