import os

os.environ.setdefault("EMBEDDING_PROVIDER", "tfidf")
os.environ.setdefault("LLM_PROVIDER", "offline")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["llm_provider"] == "offline"


def test_analyze_and_download_report_end_to_end():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    pdf_path = repo_root / "sample_data" / "sample_pitch_deck.pdf"

    with open(pdf_path, "rb") as fh:
        resp = client.post(
            "/api/v1/analyze",
            files={"pitch_deck": ("sample_pitch_deck.pdf", fh, "application/pdf")},
            data={"sheet_id": str(repo_root / "sample_data" / "sample_financials.csv"), "company_name": "NovaGrid AI"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["company_name"] == "NovaGrid AI"
    assert body["verdict"] in {"Invest", "Conditional", "Pass"}
    assert body["report_available"] is True
    assert any(f["metric"] == "MRR" for f in body["red_flags"])

    report_resp = client.get(f"/api/v1/report/{body['job_id']}")
    assert report_resp.status_code == 200
    assert report_resp.headers["content-type"] == "application/pdf"


def test_report_404_for_unknown_job_id():
    resp = client.get("/api/v1/report/not-a-real-job-id")
    assert resp.status_code == 404
