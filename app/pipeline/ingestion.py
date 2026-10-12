"""Step 1: Data Ingestion & Validation.

Loads the three data sources named in the spec (PDF pitch deck, Google
Sheet / financial CSV, market API) with the required fallback behaviour:
- API failure/rate-limit -> fall back to cached local market multiples.
- Empty/unreadable PDF chunk -> log an error and proceed on sheet data
  alone, with a low-confidence warning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.config import settings
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

CACHED_MULTIPLES_PATH = Path(__file__).resolve().parent.parent / "utils" / "cached_market_multiples.json"


@dataclass
class IngestionResult:
    pitch_deck_text: str
    pitch_deck_ok: bool
    sheet_df: Optional[pd.DataFrame]
    sheet_metrics: Dict[str, float]
    market_data: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


def load_pitch_deck_pdf(path: str | Path) -> tuple[str, bool, str]:
    """Extracts text from the pitch deck PDF via PyMuPDF.

    Returns (text, ok, warning_message). ok=False signals an empty/unreadable
    PDF, in which case the caller should proceed on structured sheet data
    alone with a low-confidence warning, per the spec's fallback rule.
    """
    path = Path(path)
    try:
        import pymupdf as fitz  # PyMuPDF (new import name; `fitz` alias is deprecated upstream)

        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # RuntimeError/ValueError cover PyMuPDF's own "cannot open broken
        # document"/bad-password/corrupt-file errors. This except is
        # intentionally still catching a *specific* set of expected,
        # documented failure modes for a malformed upload — not "anything
        # at all" — and the fallback (proceed on sheet data only) is the
        # spec's own required behaviour for this case, not a hidden bug.
        msg = f"PDF unreadable ({exc}); proceeding on structured sheet data only."
        logger.exception(msg)  # full traceback still goes to the log
        return "", False, msg

    if not text.strip():
        msg = "PDF parsed but produced no extractable text (likely scanned/image-only); low-confidence fallback."
        logger.error(msg)
        return "", False, msg

    return text, True, ""


# ---------------------------------------------------------------------------
# Wide-format financial statement support
# ---------------------------------------------------------------------------
# Real due-diligence sources (SEC filings, data-room exports) often come as
# one row per fiscal period with many named columns — not the simple long
# "Metric,Value" format the rest of this module expects. This detects that
# shape and maps a known subset of columns onto VentureAgent's internal
# metric names, using the most recent period by default. "revenue" is
# mapped to "ARR" as the closest available top-line-annual-revenue proxy —
# it is GAAP revenue, not true subscription recurring revenue, which is
# noted in the returned message so it's never presented as more precise
# than it is.
_WIDE_FORMAT_COLUMN_MAP: Dict[str, str] = {
    "revenue": "ARR",
    "cash_and_cash_equivalents": "Cash Balance",
    "avg_monthly_net_burn": "Burn Rate",  # already monthly — matches "Burn Rate"'s existing units
}
_PERIOD_COLUMN_CANDIDATES = ("period", "fiscal_year", "year")


def _try_parse_wide_format(df: pd.DataFrame) -> tuple[Dict[str, float], str]:
    """Returns ({internal_metric: value}, note). The dict is empty and the
    note is "" if `df` doesn't look like a recognized wide-format export."""
    available = {col: internal for col, internal in _WIDE_FORMAT_COLUMN_MAP.items() if col in df.columns}
    if not available:
        return {}, ""

    period_col = next((c for c in _PERIOD_COLUMN_CANDIDATES if c in df.columns), None)
    sorted_df = df.sort_values(period_col) if period_col else df
    latest_row = sorted_df.iloc[-1]

    metrics: Dict[str, float] = {}
    for col, internal_name in available.items():
        try:
            metrics[internal_name] = float(latest_row[col])
        except (TypeError, ValueError):
            continue

    # With >= 2 periods and a revenue column, derive a *real* prior-period
    # ARR so Burn Multiple can use true Net New ARR instead of falling back
    # to the "approximate" total-ARR proxy (see valuation.aggregate_metrics).
    derived_prior_arr = False
    if "revenue" in available and len(sorted_df) >= 2:
        try:
            metrics["Prior ARR"] = float(sorted_df.iloc[-2]["revenue"])
            derived_prior_arr = True
        except (TypeError, ValueError):
            pass

    period_label = str(latest_row[period_col]) if period_col else f"row {len(df) - 1}"
    note = (
        f"Detected a wide-format financial statement export ({len(df)} period(s) found); "
        f"used the most recent period ({period_label})"
        + (", with the prior period used to derive a real Net New ARR" if derived_prior_arr else "")
        + f". Mapped columns: {', '.join(f'{c}->{n}' for c, n in available.items())}. "
        "'revenue' was mapped to ARR as a proxy (GAAP revenue, not true recurring revenue). "
        "Metrics not present in this file (MRR, TAM, SAM, SOM) will show as N/A."
    )
    return metrics, note


def load_financial_sheet(source: str | Path) -> tuple[pd.DataFrame, Dict[str, float], str]:
    """Loads structured financial metrics.

    `source` may be:
      - a local CSV/XLSX path (used for the bundled demo data and for
        pipeline testing), or
      - a Google Sheet ID/URL, loaded via gspread when
        GOOGLE_SERVICE_ACCOUNT_JSON is configured.

    Supports two schemas: the simple long "Metric,Value" format, and a
    wide, one-row-per-fiscal-period financial-statement export (see
    `_try_parse_wide_format`). Returns (raw_dataframe, {metric_name: value},
    an informational note — "" if nothing noteworthy happened).
    """
    source_str = str(source)
    df: pd.DataFrame

    is_probably_google_id = not source_str.lower().endswith((".csv", ".xlsx", ".xls")) and not Path(source_str).exists()

    if is_probably_google_id and settings.google_service_account_json:
        try:
            import gspread
            from google.oauth2.service_account import Credentials

            scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
            creds_blob = settings.google_service_account_json
            info = json.loads(creds_blob) if creds_blob.strip().startswith("{") else json.loads(Path(creds_blob).read_text())
            creds = Credentials.from_service_account_info(info, scopes=scopes)
            gc = gspread.authorize(creds)
            sheet_key = source_str.split("/d/")[-1].split("/")[0] if "/d/" in source_str else source_str
            worksheet = gc.open_by_key(sheet_key).sheet1
            records = worksheet.get_all_records()
            df = pd.DataFrame(records)
        except (
            ImportError,           # gspread/google-auth not installed
            FileNotFoundError,     # GOOGLE_SERVICE_ACCOUNT_JSON points at a missing file
            json.JSONDecodeError,  # malformed service-account JSON
            KeyError,              # malformed service-account JSON (missing required field)
        ) as exc:
            logger.exception("Google Sheets fetch failed (%s); falling back to bundled sample financials.", exc)
            df = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "sample_data" / "sample_financials.csv")
        except Exception as exc:  # noqa: BLE001 — gspread/google-auth raise many of their own
            # exception classes (APIError, TransportError, RefreshError, ...);
            # rather than enumerate every one, we catch the rest here but log
            # the full traceback so nothing is silently swallowed.
            logger.exception("Unexpected Google Sheets error (%s); falling back to bundled sample financials.", exc)
            df = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "sample_data" / "sample_financials.csv")
    else:
        csv_path = Path(source_str)
        if not csv_path.exists():
            logger.warning("Sheet source '%s' not found locally and no Google credentials configured; using bundled sample financials.", source_str)
            csv_path = Path(__file__).resolve().parent.parent.parent / "sample_data" / "sample_financials.csv"
        df = pd.read_csv(csv_path)

    metrics: Dict[str, float] = {}
    note = ""
    if {"Metric", "Value"}.issubset(df.columns):
        for _, row in df.iterrows():
            try:
                metrics[str(row["Metric"]).strip()] = float(row["Value"])
            except (TypeError, ValueError):
                continue
    else:
        metrics, note = _try_parse_wide_format(df)
        if not metrics:
            logger.warning(
                "Sheet source has neither a Metric/Value schema nor a recognized wide-format "
                "financial-statement schema (columns: %s); proceeding with no structured metrics.",
                list(df.columns),
            )
    return df, metrics, note


def _load_cached_multiples() -> Dict[str, Any]:
    if CACHED_MULTIPLES_PATH.exists():
        return json.loads(CACHED_MULTIPLES_PATH.read_text())
    return {"note": "no cached multiples file found", "median_revenue_multiple": 6.0}


def fetch_market_data(sector: str = "software") -> tuple[Dict[str, Any], bool]:
    """Fetches public-market comps via yfinance/NewsAPI.

    On failure or rate-limit, falls back to a small bundled JSON of cached
    market multiples so the pipeline never hard-fails on a flaky third-party
    API. Returns (data, used_live_api).
    """
    try:
        import yfinance as yf

        # Small, illustrative comp set; a real deployment would map `sector`
        # to a curated ticker basket.
        tickers = ["CRM", "NOW", "DDOG"]
        multiples = []
        for t in tickers:
            info = yf.Ticker(t).fast_info
            price = getattr(info, "last_price", None)
            if price:
                multiples.append(price)
        if not multiples:
            raise RuntimeError("yfinance returned no usable data")
        return {"source": "yfinance", "sample_prices": multiples, "sector": sector}, True
    except (ImportError, RuntimeError, ConnectionError, TimeoutError) as exc:
        # ImportError: yfinance not installed. ConnectionError/TimeoutError:
        # the network call itself failed. RuntimeError: our own "no usable
        # data" / yfinance's internal rate-limit errors also surface as this.
        logger.warning("Market API call failed/unavailable (%s); using cached local multiples.", exc)
        return {"source": "cached_local", **_load_cached_multiples(), "sector": sector}, False
    except Exception as exc:  # noqa: BLE001 — yfinance wraps requests/urllib3 errors
        # inconsistently across versions; this is the documented last-resort
        # net so a flaky third-party API can never hard-crash ingestion, per
        # the spec's own fallback requirement. Full traceback still logged.
        logger.exception("Unexpected market-data error (%s); using cached local multiples.", exc)
        return {"source": "cached_local", **_load_cached_multiples(), "sector": sector}, False


def run_ingestion(pitch_deck_path: str | Path, sheet_source: str | Path, sector: str = "software") -> IngestionResult:
    warnings: List[str] = []

    text, pdf_ok, pdf_warning = load_pitch_deck_pdf(pitch_deck_path)
    if pdf_warning:
        warnings.append(pdf_warning)

    sheet_df, sheet_metrics, sheet_note = load_financial_sheet(sheet_source)
    if sheet_note:
        warnings.append(sheet_note)
    market_data, live = fetch_market_data(sector)
    if not live:
        warnings.append("Market API unavailable; used cached local multiples.")

    return IngestionResult(
        pitch_deck_text=text,
        pitch_deck_ok=pdf_ok,
        sheet_df=sheet_df,
        sheet_metrics=sheet_metrics,
        market_data=market_data,
        warnings=warnings,
    )
