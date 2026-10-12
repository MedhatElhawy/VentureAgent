"""Step 4 / overall glue: sequential Python orchestration flow with
built-in retry and fallback mechanisms, tying Steps 1-3 together and
producing the structured JSON result + run log the API/UI both consume.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_fixed

from app.eval.evaluator import score_groundedness
from app.pipeline import actions, valuation
from app.pipeline.analysis import (
    decide_verdict,
    detect_red_flags,
    extract_pdf_claims,
    sheet_facts_from_metrics,
    synthesize_executive_summary,
)
from app.pipeline.ingestion import run_ingestion
from app.pipeline.rag import HybridRetriever, chunk_text, rerank
from app.schemas import ActionStatus, AnalysisResult, PipelineRunResult, StepLog
from app.utils.logging_utils import RunLog, StepTimer, get_logger

logger = get_logger(__name__)


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=False)
def _ingest_with_retry(pitch_deck_path, sheet_source, sector):
    return run_ingestion(pitch_deck_path, sheet_source, sector)


def run_pipeline(
    pitch_deck_path: str | Path,
    sheet_source: str | Path,
    company_name: str,
    session_dir: str | Path,
    sector: str = "software",
) -> PipelineRunResult:
    """Runs the full sequential pipeline for one analysis request.

    `session_dir` should be a per-request isolated temp directory (the
    FastAPI/Gradio layers create one via tempfile.TemporaryDirectory() and
    a uuid4 session id, per the multi-user concurrency spec).
    """
    job_id = str(uuid.uuid4())
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    run_log = RunLog(job_id)

    # --- Step 1: Ingestion & Validation -------------------------------------------------
    with StepTimer("ingestion", logger) as t:
        try:
            ingestion = _ingest_with_retry(pitch_deck_path, sheet_source, sector)
            if ingestion.warnings:
                t.fallback("; ".join(ingestion.warnings))
            run_log.add_tool_call("load_pitch_deck_pdf", "ok" if ingestion.pitch_deck_ok else "fallback")
            run_log.add_tool_call("load_financial_sheet", "ok")
            run_log.add_tool_call("fetch_market_data", "ok" if not ingestion.warnings else "fallback")
        except Exception as exc:  # noqa: BLE001 — this is the top-level,
            # per-step safety net the spec's orchestration requirement asks
            # for (one failing step must not crash the whole run); it is
            # intentionally broad, unlike the narrowed excepts inside
            # ingestion.py/actions.py for specific external-call failure
            # modes. logger.exception() below still records the full
            # traceback, so a real bug here is never silently hidden.
            t.status, t.detail = "error", str(exc)
            logger.exception("Ingestion failed entirely: %s", exc)
            ingestion = None
    run_log.add_step(t)

    if ingestion is None:
        raise RuntimeError("Ingestion failed for both PDF and sheet sources; cannot proceed.")

    # --- Step 2a: RAG retrieval over the pitch deck --------------------------------------
    with StepTimer("rag_retrieval", logger) as t:
        context_chunks: list[str] = []
        if ingestion.pitch_deck_ok and ingestion.pitch_deck_text.strip():
            chunks = chunk_text(ingestion.pitch_deck_text)
            retriever = HybridRetriever(chunks)
            query = f"{company_name} revenue burn rate MRR ARR TAM SAM SOM market size financials"
            candidates = retriever.retrieve(query)
            top_chunks = rerank(query, candidates)
            context_chunks = [c.text for c in top_chunks]
            t.detail = f"{len(chunks)} chunks indexed, embedding_provider={retriever.embedding_provider}"
        else:
            t.fallback("Pitch deck text unavailable; skipping RAG retrieval, proceeding on sheet data only.")
    run_log.add_step(t)

    # --- Step 2b: fact extraction, red-flag detection, valuation ------------------------
    with StepTimer("processing_and_valuation", logger) as t:
        pdf_facts = extract_pdf_claims(ingestion.pitch_deck_text) if ingestion.pitch_deck_ok else []
        sheet_facts = sheet_facts_from_metrics(ingestion.sheet_metrics)
        all_facts = pdf_facts + sheet_facts
        red_flags = detect_red_flags(pdf_facts, ingestion.sheet_metrics)
        metrics = valuation.aggregate_metrics(ingestion.sheet_metrics)
        runway_display = valuation.format_runway(metrics.runway_months, metrics.runway_status)
        summary_fields = {
            "mrr": metrics.mrr, "arr": metrics.arr, "burn_rate": metrics.burn_rate,
            "cash_balance": metrics.cash_balance, "runway": runway_display,
            "burn_multiple": metrics.burn_multiple,
            "burn_multiple_approximate": metrics.burn_multiple_approximate if metrics.burn_multiple is not None else None,
            "tam": metrics.tam, "sam": metrics.sam, "som": metrics.som,
            "tam_sam_som_consistent": metrics.tam_sam_som_consistent,
        }
        metrics_summary = ", ".join(
            f"{k}={v:,.0f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}={v}"
            for k, v in summary_fields.items() if v is not None
        )
    run_log.add_step(t)

    # --- Step 2c: LLM synthesis -----------------------------------------------------------
    with StepTimer("llm_synthesis", logger) as t:
        summary, provider_used = synthesize_executive_summary(company_name, all_facts, red_flags, metrics_summary)
        if provider_used == "offline":
            t.fallback("LLM_PROVIDER not configured or live call failed; used offline template synthesis.")
        decision = decide_verdict(red_flags)
        groundedness = score_groundedness(summary, context_chunks or [f.raw_text for f in pdf_facts])
    run_log.add_step(t)

    citations = [f"[{f.source}] {f.metric}: {f.raw_text if hasattr(f, 'raw_text') else f.description}" for f in all_facts]    
    if citations:
        formatted_citations = "\n\n### Grounding Sources & Citations\n" + "\n".join(f"- {c}" for c in citations)
        summary_with_citations = summary + formatted_citations
    else:
        summary_with_citations = summary

    analysis = AnalysisResult(
        company_name=company_name,
        executive_summary=summary_with_citations,
        financial_metrics=metrics,
        extracted_facts=all_facts,
        red_flags=red_flags,
        decision=decision,
        groundedness_score=groundedness,
        citations=citations,
    )

    # --- Step 3: Automated Actions (each isolated so one failure doesn't block others) --
    action_statuses: list[ActionStatus] = []
    report_path: Path | None = None

    with StepTimer("action_generate_pdf", logger) as t:
        try:
            report_path = actions.generate_pdf_report(analysis, session_dir / "Investment_Memo.pdf")
            action_statuses.append(ActionStatus(action="generate_pdf", status="success", detail=str(report_path)))
        except Exception as exc:  # noqa: BLE001 — top-level action-step
            # isolation net (Section 3's "each action independently
            # try/excepted"); traceback logged, not hidden.
            t.status, t.detail = "error", str(exc)
            logger.exception("PDF generation failed: %s", exc)
            action_statuses.append(ActionStatus(action="generate_pdf", status="failed", detail=str(exc)))
    run_log.add_step(t)

    with StepTimer("action_send_email", logger) as t:
        try:
            status = actions.send_email_alert(analysis, report_path) if report_path else ActionStatus(action="send_email", status="skipped", detail="No report to attach.")
            if status.status == "mock":
                t.fallback(status.detail)
            action_statuses.append(status)
        except Exception as exc:  # noqa: BLE001 — same top-level isolation net as above.
            t.status, t.detail = "error", str(exc)
            logger.exception("Email dispatch step raised unexpectedly: %s", exc)
            action_statuses.append(ActionStatus(action="send_email", status="failed", detail=str(exc)))
    run_log.add_step(t)

    with StepTimer("action_update_sheet", logger) as t:
        try:
            status = actions.update_google_sheet(analysis, session_dir)
            if status.status == "mock":
                t.fallback(status.detail)
            action_statuses.append(status)
        except Exception as exc:  # noqa: BLE001 — same top-level isolation net as above.
            t.status, t.detail = "error", str(exc)
            logger.exception("Sheet update step raised unexpectedly: %s", exc)
            action_statuses.append(ActionStatus(action="update_sheet", status="failed", detail=str(exc)))
    run_log.add_step(t)

    run_log.write(session_dir)

    return PipelineRunResult(
        job_id=job_id,
        company_name=company_name,
        analysis=analysis,
        action_statuses=action_statuses,
        step_logs=[StepLog(**s) for s in run_log.steps],
        report_path=str(report_path) if report_path else None,
    )
