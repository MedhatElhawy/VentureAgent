"""Financial metric computation: Burn Multiple, Runway, TAM/SAM/SOM
consistency — the quantitative half of Step 2 (Processing & Valuation)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from app.schemas import FinancialMetrics


def compute_burn_multiple(net_burn: Optional[float], net_new_arr: Optional[float]) -> Optional[float]:
    """Burn Multiple = Net Burn / Net New ARR. Lower is better (capital efficient)."""
    if net_burn is None or net_new_arr in (None, 0):
        return None
    return round(net_burn / net_new_arr, 2)


def compute_runway(cash_balance: Optional[float], burn_rate: Optional[float]) -> Tuple[Optional[float], str]:
    """Returns (runway_months, status).

    burn_rate > 0  -> "finite", a real months-of-runway number.
    burn_rate <= 0 -> "profitable": the company is breakeven or cash-flow
                      positive, so runway is effectively infinite — this is
                      reported explicitly rather than collapsing to None,
                      which used to be indistinguishable from "unknown".
    missing data   -> "unknown".
    """
    if cash_balance is None or burn_rate is None:
        return None, "unknown"
    if burn_rate <= 0:
        return None, "profitable"
    return round(cash_balance / burn_rate, 1), "finite"


def compute_runway_months(cash_balance: Optional[float], burn_rate: Optional[float]) -> Optional[float]:
    """Back-compat wrapper around compute_runway() returning just the number."""
    months, _ = compute_runway(cash_balance, burn_rate)
    return months


def format_runway(months: Optional[float], status: str) -> str:
    """Human-readable runway string for the PDF memo / UI."""
    if status == "profitable":
        return "Infinite — profitable / self-sustaining"
    if status == "finite" and months is not None:
        return f"{months:g} months"
    return "N/A"


def check_tam_sam_som_consistency(tam: Optional[float], sam: Optional[float], som: Optional[float]) -> Optional[bool]:
    """A sane market-sizing funnel requires SOM <= SAM <= TAM."""
    if tam is None or sam is None or som is None:
        return None
    return som <= sam <= tam


def aggregate_metrics(sheet_metrics: Dict[str, float]) -> FinancialMetrics:
    mrr = sheet_metrics.get("MRR")
    arr = sheet_metrics.get("ARR") or (mrr * 12 if mrr is not None else None)
    burn_rate = sheet_metrics.get("Burn Rate")
    cash_balance = sheet_metrics.get("Cash Balance")
    tam, sam, som = sheet_metrics.get("TAM"), sheet_metrics.get("SAM"), sheet_metrics.get("SOM")

    # Net New ARR: prefer an explicit sheet column. Next best, derive it
    # from a prior-period ARR column ("Prior ARR" / "Previous ARR") if the
    # sheet provides one. Only as a last resort do we fall back to total
    # ARR as a rough proxy — and when we do, burn_multiple_approximate is
    # set so the PDF/UI make clear the ratio is understated, rather than
    # silently presenting it as precise.
    prior_arr = sheet_metrics.get("Prior ARR", sheet_metrics.get("Previous ARR"))
    burn_multiple_approximate = False
    if "Net New ARR" in sheet_metrics:
        net_new_arr = sheet_metrics["Net New ARR"]
    elif prior_arr is not None and arr is not None:
        net_new_arr = arr - prior_arr
    else:
        net_new_arr = arr
        burn_multiple_approximate = arr is not None  # only "approximate" if we actually used a proxy

    runway_months, runway_status = compute_runway(cash_balance, burn_rate)

    # Burn Multiple = Net Burn / Net New ARR, both measured over the SAME
    # period. "Burn Rate" throughout this system is a MONTHLY figure (that's
    # what makes `cash_balance / burn_rate` above come out in months of
    # runway) while Net New ARR is ANNUAL — dividing the two directly
    # without annualizing the burn rate first silently understates Burn
    # Multiple by ~12x. (Caught by testing against a real multi-year
    # filing: a company with an obviously poor burn profile was coming out
    # with an implausibly good-looking multiple until this was fixed.)
    annualized_burn_rate = burn_rate * 12 if burn_rate is not None else None

    return FinancialMetrics(
        mrr=mrr,
        arr=arr,
        burn_rate=burn_rate,
        cash_balance=cash_balance,
        runway_months=runway_months,
        runway_status=runway_status,
        burn_multiple=compute_burn_multiple(annualized_burn_rate, net_new_arr),
        burn_multiple_approximate=burn_multiple_approximate,
        tam=tam,
        sam=sam,
        som=som,
        tam_sam_som_consistent=check_tam_sam_som_consistency(tam, sam, som),
    )
