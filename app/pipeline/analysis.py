"""Step 2: Data Processing, Red-Flag Detection & Anti-Hallucination.

Extracted facts are regex-pulled verbatim from the PDF/sheet (never
LLM-guessed), which is what makes the "Strict Grounding Constraint"
enforceable: the LLM only ever summarizes numbers that are already
tagged with a source, it never invents them.
"""
from __future__ import annotations

import re
from typing import Dict, List

from app.config import settings
from app.pipeline.rag import Chunk
from app.schemas import ExtractedFact, RedFlag
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# metric name -> regex patterns that may appear in pitch-deck prose
_METRIC_PATTERNS: Dict[str, List[str]] = {
    "MRR": [r"MRR[^\d$]{0,15}\$?([\d,]+(?:\.\d+)?)\s*[kK]?", r"Monthly Recurring Revenue[^\d$]{0,15}\$?([\d,]+(?:\.\d+)?)\s*[kK]?"],
    "ARR": [r"ARR[^\d$]{0,15}\$?([\d,]+(?:\.\d+)?)\s*[kK]?", r"Annual Recurring Revenue[^\d$]{0,15}\$?([\d,]+(?:\.\d+)?)\s*[kK]?"],
    "Burn Rate": [r"[Bb]urn [Rr]ate[^\d$]{0,15}\$?([\d,]+(?:\.\d+)?)\s*[kK]?"],
    "Cash Balance": [r"[Cc]ash [Bb]alance[^\d$]{0,15}\$?([\d,]+(?:\.\d+)?)\s*[kK]?"],
    "TAM": [r"TAM[^\d$]{0,15}\$?([\d,]+(?:\.\d+)?)\s*[kKmMbB]?"],
    "SAM": [r"SAM[^\d$]{0,15}\$?([\d,]+(?:\.\d+)?)\s*[kKmMbB]?"],
    "SOM": [r"SOM[^\d$]{0,15}\$?([\d,]+(?:\.\d+)?)\s*[kKmMbB]?"],
}


def _to_number(raw: str, unit_hint: str = "") -> float:
    raw = raw.replace(",", "")
    val = float(raw)
    suffix = unit_hint[-1].lower() if unit_hint and unit_hint[-1].isalpha() else ""
    if suffix == "k":
        val *= 1_000
    elif suffix == "m":
        val *= 1_000_000
    elif suffix == "b":
        val *= 1_000_000_000
    return val


def extract_pdf_claims(pitch_deck_text: str) -> List[ExtractedFact]:
    """Deterministic, regex-based extraction from the pitch deck — grounded
    by construction, tagged [Source: Pitch Deck PDF]."""
    facts: List[ExtractedFact] = []
    for metric, patterns in _METRIC_PATTERNS.items():
        for pattern in patterns:
            m = re.search(pattern, pitch_deck_text)
            if m:
                snippet_start = max(0, m.start() - 40)
                snippet = pitch_deck_text[snippet_start:m.end() + 10].replace("\n", " ").strip()
                try:
                    value = _to_number(m.group(1), m.group(0))
                except (ValueError, IndexError):
                    value = None
                facts.append(ExtractedFact(metric=metric, value=value, raw_text=snippet, source="Pitch Deck PDF"))
                break  # first matching pattern per metric is enough
    return facts


def sheet_facts_from_metrics(sheet_metrics: Dict[str, float]) -> List[ExtractedFact]:
    return [
        ExtractedFact(metric=k, value=v, raw_text=f"{k} = {v}", source="Financial Sheet")
        for k, v in sheet_metrics.items()
    ]


def detect_red_flags(pdf_facts: List[ExtractedFact], sheet_metrics: Dict[str, float]) -> List[RedFlag]:
    """Cross-references PDF claims against sheet figures. A discrepancy
    beyond MRR_DISCREPANCY_TOLERANCE_PCT is flagged HIGH severity, per the
    spec's worked example (PDF says $50K MRR, sheet says $30K -> HIGH)."""
    tolerance = settings.mrr_discrepancy_tolerance_pct / 100.0
    flags: List[RedFlag] = []

    for fact in pdf_facts:
        if fact.value is None or fact.metric not in sheet_metrics:
            continue
        sheet_val = sheet_metrics[fact.metric]
        if sheet_val == 0:
            continue
        pct_diff = abs(fact.value - sheet_val) / abs(sheet_val)
        if pct_diff > tolerance:
            severity = "HIGH" if pct_diff > 0.30 else "MEDIUM"
            flags.append(RedFlag(
                metric=fact.metric,
                severity=severity,
                description=(
                    f"Pitch deck claims {fact.metric} of {fact.value:,.0f}, but the financial "
                    f"sheet records {sheet_val:,.0f} ({pct_diff * 100:.0f}% discrepancy)."
                ),
                pdf_value=fact.value,
                sheet_value=sheet_val,
            ))
    return flags


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior VC due-diligence analyst. You are given
EXTRACTED FACTS (each explicitly source-tagged), COMPUTED FINANCIAL METRICS,
and DETECTED RED FLAGS. Produce a concise executive summary of the startup's
investment profile.

STRICT GROUNDING CONSTRAINT: do not state any financial number, market size,
or valuation metric that is not explicitly present in the provided facts/
metrics. If something is unknown, say so rather than estimating. Do not
inline citations in the body — sources are appended separately."""


def _build_user_prompt(company_name: str, facts: List[ExtractedFact], red_flags: List[RedFlag], metrics_summary: str) -> str:
    facts_block = "\n".join(f"- [{f.source}] {f.metric}: {f.value} ({f.raw_text})" for f in facts) or "(none extracted)"
    flags_block = "\n".join(f"- [{f.severity}] {f.description}" for f in red_flags) or "(none detected)"
    return (
        f"Company: {company_name}\n\nEXTRACTED FACTS:\n{facts_block}\n\n"
        f"COMPUTED METRICS:\n{metrics_summary}\n\nRED FLAGS:\n{flags_block}\n\n"
        "Write a 3-5 sentence executive summary grounded strictly in the above."
    )


def _offline_summary(company_name: str, facts: List[ExtractedFact], red_flags: List[RedFlag], metrics_summary: str) -> str:
    """Deterministic template synthesis — zero API calls, used when
    LLM_PROVIDER=offline (the default) or as a fallback if a live call fails.
    Every sentence is built only from data already present in `facts`/
    `red_flags`, so it satisfies the same grounding constraint as the LLM path."""
    n_facts = len(facts)
    if red_flags:
        worst = max(red_flags, key=lambda f: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[f.severity])
        flag_sentence = f" {len(red_flags)} discrepancy(ies) were found between the pitch deck and financial sheet, the most severe being: {worst.description}"
    else:
        flag_sentence = " No discrepancies were found between the pitch deck and the financial sheet for the metrics that were cross-referenced."
    return (
        f"{company_name} was evaluated using {n_facts} source-tagged fact(s) extracted from the pitch "
        f"deck and financial sheet.{flag_sentence} "
        f"Financial metrics computed from the sheet data: {metrics_summary}. "
        "This summary is generated offline (LLM_PROVIDER=offline) using only the grounded facts above; "
        "no figures were inferred or estimated."
    )

def _call_cohere(system_prompt: str, user_prompt: str) -> str | None:
    if not settings.cohere_api_key:
        logger.warning("Cohere API key missing for LLM synthesis.")
        return None
    try:
        import cohere
        co = cohere.Client(api_key=settings.cohere_api_key)
        
        response = co.chat(
            model="command-r-08-2024",
            message=user_prompt,
            preamble=system_prompt,
            temperature=0.3
        )
        return response.text
    except ImportError:
        logger.warning("cohere package not installed; falling back to offline synthesis.")
        return None
    except Exception as exc:
        logger.warning("Cohere LLM synthesis call failed (%s); falling back to offline synthesis.", exc)
        return None

def _call_openai(system_prompt: str, user_prompt: str) -> str | None:
    if not settings.openai_api_key:
        return None
    try:
        from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

        client = OpenAI(api_key=settings.openai_api_key)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except (APIConnectionError, APIStatusError, RateLimitError) as exc:
            # Expected, documented OpenAI SDK failure modes: network issue,
            # non-2xx response (bad key, quota, etc.), or rate limiting.
            logger.warning("OpenAI call failed (%s); falling back to offline synthesis.", exc)
            return None
    except ImportError as exc:
        logger.warning("openai package not installed (%s); falling back to offline synthesis.", exc)
        return None



def synthesize_executive_summary(
    company_name: str,
    facts: List[ExtractedFact],
    red_flags: List[RedFlag],
    metrics_summary: str,
) -> tuple[str, str]:
    """Returns (summary_text, provider_actually_used)."""
    user_prompt = _build_user_prompt(company_name, facts, red_flags, metrics_summary)

    if settings.llm_provider == "cohere":
        result = _call_cohere(_SYSTEM_PROMPT, user_prompt)
        if result:
            return result.strip(), "cohere"
    elif settings.llm_provider == "openai":
        result = _call_openai(_SYSTEM_PROMPT, user_prompt)
        if result:
            return result.strip(), "openai"

    return _offline_summary(company_name, facts, red_flags, metrics_summary), "offline"


def decide_verdict(red_flags: List[RedFlag]) -> str:
    high = sum(1 for f in red_flags if f.severity == "HIGH")
    medium = sum(1 for f in red_flags if f.severity == "MEDIUM")
    if high > 0:
        return "Pass"
    if medium > 0:
        return "Conditional"
    return "Invest"
