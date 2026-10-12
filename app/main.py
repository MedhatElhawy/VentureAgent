"""FastAPI backend — Section 3 of the spec.

    POST /api/v1/analyze          multipart: pitch_deck file, sheet_id, company_name
    GET  /api/v1/report/{job_id}  downloads the generated Investment_Memo.pdf
    GET  /api/v1/hea2lth           API + vector-DB readiness check

Heavy work (PDF chunking, vector search, PDF/email/sheet actions) runs via
asyncio.to_thread so the event loop stays free for concurrent requests,
satisfying the "non-blocking API performance" requirement without needing
a separate task queue for this project's scope.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from app.config import settings
from app.pipeline.orchestrator import run_pipeline
from app.schemas import AnalyzeResponse, HealthResponse
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="VentureAgent API",
    description="Autonomous Startup Valuation & Due Diligence Platform",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")

# job_id -> generated report path (in-memory store; fine for a single-process
# demo deployment such as a HF Space, per the project's scope).
_REPORTS: Dict[str, str] = {}
_BASE_TMP = Path(tempfile.gettempdir()) / "ventureagent_sessions"
_BASE_TMP.mkdir(parents=True, exist_ok=True)


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    pitch_deck: UploadFile = File(...),
    sheet_id: str = Form(...),
    company_name: str = Form(...),
) -> AnalyzeResponse:
    # Session file isolation: unique UUID4 dir per request, cleaned up after.
    session_id = str(uuid.uuid4())
    session_dir = _BASE_TMP / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    pitch_deck_path = session_dir / (pitch_deck.filename or "pitch_deck.pdf")
    with open(pitch_deck_path, "wb") as fh:
        shutil.copyfileobj(pitch_deck.file, fh)

    try:
        result = await asyncio.to_thread(
            run_pipeline, pitch_deck_path, sheet_id, company_name, session_dir,
        )
    except Exception as exc:
        logger.error("Pipeline run failed for %s: %s", company_name, exc)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    if result.report_path:
        _REPORTS[result.job_id] = result.report_path

    any_failed = any(a.status == "failed" for a in result.action_statuses)
    status = "completed_with_warnings" if any_failed else "completed"

    return AnalyzeResponse(
        job_id=result.job_id,
        status=status,
        company_name=result.company_name,
        extracted_metrics=result.analysis.financial_metrics,
        red_flags=result.analysis.red_flags,
        verdict=result.analysis.decision,
        action_statuses=result.action_statuses,
        report_available=result.report_path is not None,
    )


@app.get("/api/v1/report/{job_id}")
async def get_report(job_id: str) -> FileResponse:
    path = _REPORTS.get(job_id)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Report not found for this job_id.")
    return FileResponse(path, media_type="application/pdf", filename="Investment_Memo.pdf")


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    vector_db_ready, bm25_ready, detail = True, True, ""
    try:
        import faiss
    except Exception as exc:
        vector_db_ready = False
        detail += f"faiss unavailable: {exc}. "
    try:
        import rank_bm25
    except Exception as exc:
        bm25_ready = False
        detail += f"rank_bm25 unavailable: {exc}. "

    status = "ok" if (vector_db_ready and bm25_ready) else "degraded"
    return HealthResponse(
        status=status,
        vector_db_ready=vector_db_ready,
        bm25_ready=bm25_ready,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        detail=detail,
    )


# --- Mount the Gradio dashboard onto this same FastAPI app/port ------------

import gradio as gr

from gradio_app import demo as _gradio_demo

app = gr.mount_gradio_app(app, _gradio_demo, path="/ui")
