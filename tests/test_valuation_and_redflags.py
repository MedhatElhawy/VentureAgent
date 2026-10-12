import os

os.environ.setdefault("EMBEDDING_PROVIDER", "tfidf")
os.environ.setdefault("LLM_PROVIDER", "offline")

from pathlib import Path 

from app.pipeline.analysis import decide_verdict, detect_red_flags, extract_pdf_claims
from app.pipeline.ingestion import load_financial_sheet

REPO_ROOT = Path(__file__).resolve().parent.parent
from app.pipeline.valuation import ( 
    aggregate_metrics,
    check_tam_sam_som_consistency,
    compute_burn_multiple,
    compute_runway,
    compute_runway_months,
    format_runway,
)
from app.utils.sample_data import FINANCIAL_SHEET_ROWS, PITCH_DECK_TEXT


def test_compute_runway_months():
    assert compute_runway_months(cash_balance=540000, burn_rate=45000) == 12.0
    assert compute_runway_months(cash_balance=100000, burn_rate=0) is None
    assert compute_runway_months(cash_balance=None, burn_rate=1000) is None


def test_compute_burn_multiple():
    assert compute_burn_multiple(net_burn=45000, net_new_arr=90000) == 0.5
    assert compute_burn_multiple(net_burn=45000, net_new_arr=0) is None


def test_compute_runway_profitable_company_is_not_none():
    """A company with zero/negative burn (breakeven or cash-flow positive)
    should be reported as 'profitable' with infinite runway, not collapsed
    into the same None as 'we have no data'."""
    months, status = compute_runway(cash_balance=500000, burn_rate=0)
    assert months is None
    assert status == "profitable"
    assert format_runway(months, status) == "Infinite — profitable / self-sustaining"

    months, status = compute_runway(cash_balance=500000, burn_rate=-10000)  # net income positive
    assert status == "profitable"


def test_compute_runway_missing_data_is_unknown_not_profitable():
    months, status = compute_runway(cash_balance=None, burn_rate=5000)
    assert months is None
    assert status == "unknown"
    assert format_runway(months, status) == "N/A"


def test_compute_runway_finite_case_unchanged():
    months, status = compute_runway(cash_balance=540000, burn_rate=45000)
    assert months == 12.0
    assert status == "finite"
    assert format_runway(months, status) == "12 months"


def test_tam_sam_som_consistency():
    assert check_tam_sam_som_consistency(tam=100, sam=50, som=10) is True
    assert check_tam_sam_som_consistency(tam=10, sam=50, som=100) is False
    assert check_tam_sam_som_consistency(None, 50, 10) is None


def test_extract_pdf_claims_finds_mrr_and_conflicts_with_sheet():
    facts = extract_pdf_claims(PITCH_DECK_TEXT)
    mrr_facts = [f for f in facts if f.metric == "MRR"]
    assert mrr_facts, "MRR was not extracted from the sample pitch deck"
    assert mrr_facts[0].value == 50000
    assert mrr_facts[0].source == "Pitch Deck PDF"

    sheet_metrics = dict(FINANCIAL_SHEET_ROWS)
    assert sheet_metrics["MRR"] == 30000

    red_flags = detect_red_flags(facts, sheet_metrics)
    mrr_flags = [f for f in red_flags if f.metric == "MRR"]
    assert mrr_flags, "expected a HIGH-severity MRR discrepancy red flag"
    assert mrr_flags[0].severity == "HIGH"
    assert mrr_flags[0].pdf_value == 50000
    assert mrr_flags[0].sheet_value == 30000


def test_detect_red_flags_no_flag_when_values_match():
    from app.schemas import ExtractedFact

    facts = [ExtractedFact(metric="MRR", value=30500, raw_text="MRR: $30,500", source="Pitch Deck PDF")]
    sheet_metrics = {"MRR": 30000}
    flags = detect_red_flags(facts, sheet_metrics)
    assert flags == []  # within default 15% tolerance


def test_decide_verdict_logic():
    from app.schemas import RedFlag

    assert decide_verdict([]) == "Invest"
    assert decide_verdict([RedFlag(metric="MRR", severity="MEDIUM", description="x")]) == "Conditional"
    assert decide_verdict([RedFlag(metric="MRR", severity="HIGH", description="x")]) == "Pass"


def test_wide_format_csv_is_detected_and_mapped_to_latest_period():
    """Uses the bundled WeWork-style multi-year SEC-filing export
    (sample_wide_format_financials.csv) — one row per fiscal year, named
    columns, no Metric/Value schema at all."""
    df, metrics, note = load_financial_sheet(REPO_ROOT / "sample_data" / "sample_wide_format_financials.csv")
    assert len(df) == 3 

    assert metrics["ARR"] == 1821751          
    assert metrics["Cash Balance"] == 1744209  
    assert metrics["Burn Rate"] == 160618.25 

    assert metrics["Prior ARR"] == 886004

    assert "FY2018" in note
    assert "MRR" in note 


def test_wide_format_burn_multiple_matches_weworks_known_poor_capital_efficiency():
    """End-to-end sanity check with real, public numbers: WeWork's FY2018
    Burn Multiple should land around ~2 (burn roughly 2x net-new revenue) —
    not an artificially tiny number from an un-annualized burn rate, and
    not burn_multiple_approximate (a real prior period was available)."""
    _, metrics, _ = load_financial_sheet(REPO_ROOT / "sample_data" / "sample_wide_format_financials.csv")
    financials = aggregate_metrics(metrics)
    assert financials.burn_multiple_approximate is False
    assert 1.9 <= financials.burn_multiple <= 2.2
    assert financials.mrr is None
    assert financials.tam is None


def test_aggregate_metrics_from_sample_sheet():
    sheet_metrics = dict(FINANCIAL_SHEET_ROWS)
    metrics = aggregate_metrics(sheet_metrics)
    assert metrics.mrr == 30000
    assert metrics.arr == 600000
    assert metrics.runway_months == 12.0
    assert metrics.runway_status == "finite"
    assert metrics.tam_sam_som_consistent is True
    assert metrics.burn_multiple_approximate is True


def test_aggregate_metrics_uses_real_net_new_arr_when_prior_period_given():
    sheet_metrics = dict(FINANCIAL_SHEET_ROWS)
    sheet_metrics["Prior ARR"] = 480000  
    metrics = aggregate_metrics(sheet_metrics)
    assert metrics.burn_multiple_approximate is False
    assert metrics.burn_multiple == compute_burn_multiple(45000 * 12, 120000)


def test_aggregate_metrics_explicit_net_new_arr_column_wins():
    sheet_metrics = dict(FINANCIAL_SHEET_ROWS)
    sheet_metrics["Net New ARR"] = 200000
    metrics = aggregate_metrics(sheet_metrics)
    assert metrics.burn_multiple_approximate is False
    assert metrics.burn_multiple == compute_burn_multiple(45000 * 12, 200000)


def test_aggregate_metrics_burn_rate_is_annualized_for_burn_multiple():
    """Regression test for the unit-mismatch bug: Burn Rate is monthly,
    Net New ARR is annual — dividing them directly (without annualizing
    burn rate) silently understates Burn Multiple by ~12x."""
    sheet_metrics = {"Burn Rate": 50000, "Net New ARR": 100000, "ARR": 100000}
    metrics = aggregate_metrics(sheet_metrics)
    assert metrics.burn_multiple == 6.0
