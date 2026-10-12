"""Pydantic data contracts shared by the FastAPI service, the Gradio UI,
and the orchestrator, so all three layers agree on one shape."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["LOW", "MEDIUM", "HIGH"]
Decision = Literal["Invest", "Conditional", "Pass"]
SourceTag = Literal["Pitch Deck PDF", "Financial Sheet", "Market API"]


class ExtractedFact(BaseModel):
    metric: str
    value: Optional[float] = None
    raw_text: str
    source: SourceTag


class RedFlag(BaseModel):
    metric: str
    severity: Severity
    description: str
    pdf_value: Optional[float] = None
    sheet_value: Optional[float] = None


class FinancialMetrics(BaseModel):
    mrr: Optional[float] = None
    arr: Optional[float] = None
    burn_rate: Optional[float] = None
    cash_balance: Optional[float] = None
    runway_months: Optional[float] = None
    runway_status: Literal["finite", "profitable", "unknown"] = "unknown"
    burn_multiple: Optional[float] = None
    burn_multiple_approximate: bool = False
    tam: Optional[float] = None
    sam: Optional[float] = None
    som: Optional[float] = None
    tam_sam_som_consistent: Optional[bool] = None


class ActionStatus(BaseModel):
    action: str
    status: Literal["success", "skipped", "mock", "failed"]
    detail: str


class StepLog(BaseModel):
    step: str
    status: Literal["ok", "fallback", "error"]
    latency_ms: float
    detail: str = ""


class AnalysisResult(BaseModel):
    company_name: str
    executive_summary: str
    financial_metrics: FinancialMetrics
    extracted_facts: List[ExtractedFact]
    red_flags: List[RedFlag]
    decision: Decision
    groundedness_score: Optional[float] = Field(None, description="LLM-as-judge score, 1-5")
    citations: List[str] = Field(default_factory=list)


class PipelineRunResult(BaseModel):
    job_id: str
    company_name: str
    analysis: AnalysisResult
    action_statuses: List[ActionStatus]
    step_logs: List[StepLog]
    report_path: Optional[str] = None


class AnalyzeResponse(BaseModel):
    job_id: str
    status: Literal["completed", "completed_with_warnings", "failed"]
    company_name: str
    extracted_metrics: FinancialMetrics
    red_flags: List[RedFlag]
    verdict: Decision
    action_statuses: List[ActionStatus]
    report_available: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    vector_db_ready: bool
    bm25_ready: bool
    llm_provider: str
    embedding_provider: str
    detail: str = ""
