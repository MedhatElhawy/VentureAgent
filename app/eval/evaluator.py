"""Step 7: Testing & Evaluation Framework.

- LLM-as-a-judge groundedness scoring (1-5) of the generated summary
  against the raw retrieved context.
- System reliability logging (latency/tokens/tool status) is handled by
  RunLog in app.utils.logging_utils and wired in by the orchestrator.
"""
from __future__ import annotations

import re
from typing import List

from app.config import settings
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _offline_groundedness(summary: str, context_chunks: List[str]) -> float:
    """Heuristic fallback: scores groundedness by lexical overlap between the
    summary and the retrieved context. Deterministic and dependency-free, so
    it always produces a usable score even with LLM_PROVIDER=offline."""
    context_words = set(re.findall(r"[a-zA-Z0-9]+", " ".join(context_chunks).lower()))
    summary_words = re.findall(r"[a-zA-Z0-9]+", summary.lower())
    if not summary_words:
        return 1.0
    overlap = sum(1 for w in summary_words if w in context_words) / len(summary_words)
    # Map overlap ratio (0-1) onto a 1-5 scale.
    return round(1 + overlap * 4, 1)


def _llm_judge(summary: str, context_chunks: List[str]) -> float | None:
    prompt = (
        "You are grading a due-diligence summary for factual groundedness against "
        "the provided context. Score from 1 (ungrounded/hallucinated) to 5 (fully "
        "grounded, every claim traceable to the context). Reply with ONLY the number.\n\n"
        f"CONTEXT:\n{chr(10).join(context_chunks)}\n\nSUMMARY:\n{summary}"
    )
    try:
        if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=10, messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in resp.content if b.type == "text")
            return float(re.search(r"[\d.]+", text).group())
        if settings.llm_provider == "openai" and settings.openai_api_key:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key)
            resp = client.chat.completions.create(model="gpt-4o-mini", max_tokens=10, messages=[{"role": "user", "content": prompt}])
            text = resp.choices[0].message.content
            return float(re.search(r"[\d.]+", text).group())
    except ImportError as exc:
        logger.warning("LLM SDK not installed (%s); falling back to lexical-overlap heuristic.", exc)
    except (AttributeError, ValueError) as exc:
        # The judge model replied with something we couldn't parse a
        # number out of (regex found nothing / bad float) — expected on
        # occasion, not a reason to hide the actual bug class below.
        logger.warning("Could not parse a groundedness score from the judge's reply (%s); using lexical-overlap heuristic.", exc)
    except Exception as exc:  # noqa: BLE001 — last-resort net for the
        # provider SDKs' own exception hierarchies (connection/rate-limit/
        # API-status errors); full traceback still logged.
        logger.exception("LLM-as-judge call failed (%s); falling back to lexical-overlap heuristic.", exc)
    return None


def score_groundedness(summary: str, context_chunks: List[str]) -> float:
    score = _llm_judge(summary, context_chunks)
    if score is not None:
        return max(1.0, min(5.0, score))
    return _offline_groundedness(summary, context_chunks)
