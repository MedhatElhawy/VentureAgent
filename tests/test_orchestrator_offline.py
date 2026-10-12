"""End-to-end smoke test of the full sequential pipeline, forced onto the
offline/tfidf code paths so it needs zero API keys or internet access —
exercises Steps 1-3 and the eval harness together."""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "tfidf")
os.environ.setdefault("LLM_PROVIDER", "offline")

from app.pipeline.orchestrator import run_pipeline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_full_pipeline_runs_end_to_end_and_flags_mrr_discrepancy():
    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp)
        result = run_pipeline(
            pitch_deck_path=REPO_ROOT / "sample_data" / "sample_pitch_deck.pdf",
            sheet_source=REPO_ROOT / "sample_data" / "sample_financials.csv",
            company_name="NovaGrid AI",
            session_dir=session_dir,
        )

        # Structured result shape
        assert result.company_name == "NovaGrid AI"
        assert result.analysis.executive_summary
        assert result.analysis.decision in {"Invest", "Conditional", "Pass"}

        # Discrepancy detection worked end-to-end
        mrr_flags = [f for f in result.analysis.red_flags if f.metric == "MRR"]
        assert mrr_flags, "expected the MRR discrepancy to surface through the full pipeline"
        assert mrr_flags[0].severity == "HIGH"
        assert result.analysis.decision == "Pass"  # HIGH red flag -> Pass per decide_verdict

        # Grounding: every citation is source-tagged
        assert all(c.startswith("[") for c in result.analysis.citations)
        assert result.analysis.groundedness_score is not None
        assert 1.0 <= result.analysis.groundedness_score <= 5.0

        # Actions: PDF was generated, email/sheet ran in mock mode (no creds in test env)
        assert result.report_path and Path(result.report_path).exists()
        action_names = {a.action for a in result.action_statuses}
        assert {"generate_pdf", "send_email", "update_sheet"} <= action_names
        pdf_action = next(a for a in result.action_statuses if a.action == "generate_pdf")
        assert pdf_action.status == "success"

        # Step logs recorded for every stage, with latency captured
        step_names = {s.step for s in result.step_logs}
        assert {"ingestion", "rag_retrieval", "processing_and_valuation", "llm_synthesis"} <= step_names
        assert all(s.latency_ms >= 0 for s in result.step_logs)

        # Run log JSON was persisted for the reliability evaluation
        run_log_files = list(session_dir.glob("run_*.json"))
        assert run_log_files, "expected a run_<job_id>.json log to be written"


def test_pipeline_falls_back_gracefully_on_missing_pdf():
    """Step 1's fallback rule: unreadable/missing PDF -> proceed on sheet
    data alone with a low-confidence warning, not a hard crash."""
    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp)
        missing_pdf = session_dir / "does_not_exist.pdf"
        result = run_pipeline(
            pitch_deck_path=missing_pdf,
            sheet_source=REPO_ROOT / "sample_data" / "sample_financials.csv",
            company_name="FallbackCo",
            session_dir=session_dir,
        )
        assert result.company_name == "FallbackCo"
        # No PDF facts possible, so no PDF-vs-sheet discrepancies either.
        assert result.analysis.red_flags == []
        assert result.report_path and Path(result.report_path).exists()
