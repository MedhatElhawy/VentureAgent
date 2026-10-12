"""Structured, step-level logging used by the orchestrator and the eval harness."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@dataclass
class StepTimer:
    """Times a pipeline step and records status/latency/token usage for the
    system-reliability evaluation (Section 7 of the spec)."""

    step: str
    logger: logging.Logger
    _start: float = field(default=0.0, init=False)
    status: str = field(default="ok", init=False)
    detail: str = field(default="", init=False)
    tokens_used: int = field(default=0, init=False)

    def __enter__(self) -> "StepTimer":
        self._start = time.perf_counter()
        self.logger.info("STEP START: %s", self.step)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        latency_ms = (time.perf_counter() - self._start) * 1000
        if exc_type is not None:
            self.status = "error"
            self.detail = f"{exc_type.__name__}: {exc}"
            self.logger.error("STEP FAILED: %s (%.1fms) — %s", self.step, latency_ms, self.detail)
        else:
            self.logger.info(
                "STEP DONE: %s (%.1fms) status=%s%s",
                self.step, latency_ms, self.status,
                f" — {self.detail}" if self.detail else "",
            )
        self.latency_ms = latency_ms
        # Never let logging/timing itself crash the pipeline.
        return False

    def fallback(self, detail: str) -> None:
        self.status = "fallback"
        self.detail = detail
        self.logger.warning("STEP FALLBACK: %s — %s", self.step, detail)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "status": self.status,
            "latency_ms": round(getattr(self, "latency_ms", 0.0), 2),
            "detail": self.detail,
        }


class RunLog:
    """Collects step logs + eval metrics for one pipeline run and persists
    them as JSON — satisfies the 'log step-by-step latency, token usage,
    and tool invocation status' requirement."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.steps: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []

    def add_step(self, timer: StepTimer) -> None:
        self.steps.append(timer.as_dict())

    def add_tool_call(self, tool: str, status: str, detail: str = "") -> None:
        self.tool_calls.append({"tool": tool, "status": status, "detail": detail})

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"run_{self.job_id}.json"
        path.write_text(json.dumps(
            {"job_id": self.job_id, "steps": self.steps, "tool_calls": self.tool_calls},
            indent=2,
        ))
        return path
