"""Gradio dashboard — Section 4 of the spec.

File input for the pitch deck, textboxes for the sheet ID/URL and company
name, a run button, and four output tabs: (1) executive summary + red-flag
table, (2) financial charts (Plotly), (3) action logs, (4) PDF preview +
download.

This module builds `demo` and can run standalone (`python gradio_app.py`)
for UI-only iteration or a classic Gradio-SDK HF Space, but is normally
mounted inside the FastAPI app (see app/main.py) so the API and the
dashboard are served from a single process on a single port, both calling
the same `run_pipeline` orchestrator — no HTTP round-trip between them.

Visual design: a due-diligence tool lives or dies on whether people trust
its numbers, so the palette leans on a disciplined charcoal/teal/gold base
— navy-to-teal for structure and trust, gold for "value" — with color used
*semantically* elsewhere: emerald/amber/rose only ever mean Invest/
Conditional/Pass or Low/Medium/High severity, never decoration.
"""
from __future__ import annotations

import base64
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from app.config import settings
from app.pipeline.orchestrator import run_pipeline
from app.pipeline.valuation import format_runway
from app.schemas import ActionStatus, PipelineRunResult, RedFlag
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

_BASE_TMP = Path(tempfile.gettempdir()) / "ventureagent_gradio_sessions"
_BASE_TMP.mkdir(parents=True, exist_ok=True)

_SAMPLE_SHEET_PATH = "sample_data/sample_financials.csv"

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------

_VERDICT_STYLE = {
    "Invest":      {"bg": "#ECFDF5", "text": "#065F46", "accent": "#10B981"},
    "Conditional": {"bg": "#FFFBEB", "text": "#92400E", "accent": "#F59E0B"},
    "Pass":        {"bg": "#FEF2F2", "text": "#991B1B", "accent": "#EF4444"},
}
_SEVERITY_ACCENT = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#64748B"}
_INK, _SUBINK, _BORDER, _PANEL = "#0F172A", "#52606D", "#E4E7EB", "#FFFFFF"


_NAVY_DARK, _NAVY, _TEAL, _TEAL_DARK, _GOLD, _GOLD_LIGHT = (
    "#0B1220", "#102A3C", "#0F766E", "#115E59", "#D97706", "#FBBF24",
)

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#F4F6F8",
    body_background_fill_dark=_NAVY_DARK,
    block_background_fill=_PANEL,
    block_border_width="1px",
    block_border_color=_BORDER,
    block_label_text_weight="600",
    button_primary_background_fill=_TEAL,
    button_primary_background_fill_hover=_TEAL_DARK,
    button_primary_text_color="#FFFFFF",
    input_border_color="#D3D8DE",
)

_CSS = """
.va-hint { font-size: 0.85rem; color: #64748B; }
"""

# ---------------------------------------------------------------------------
# HTML renderers — small, focused, and reused across tabs
# ---------------------------------------------------------------------------


def _verdict_badge_html(decision: str, company_name: str) -> str:
    style = _VERDICT_STYLE.get(decision, {"bg": "#F1F5F9", "text": "#334155", "accent": "#64748B"})
    return f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                flex-wrap:wrap;gap:10px;padding:14px 18px;border-radius:12px;
                background:{style['bg']};border:1px solid {style['accent']}33;">
      <div style="font-size:1rem;color:{_INK};font-weight:600;">{company_name}</div>
      <div style="display:inline-flex;align-items:center;gap:8px;padding:6px 16px;
                  border-radius:999px;background:{style['accent']}1A;color:{style['text']};
                  font-weight:700;font-size:0.95rem;">
        <span style="width:8px;height:8px;border-radius:50%;background:{style['accent']};"></span>
        {decision}
      </div>
    </div>
    """


def _flags_table_html(red_flags: list[RedFlag]) -> str:
    if not red_flags:
        return f"""
        <div style="padding:16px 18px;border:1px solid {_BORDER};border-radius:10px;
                    color:{_SUBINK};background:{_PANEL};">
          No discrepancies detected between the pitch deck and the financial sheet.
        </div>"""

    rows = []
    for f in red_flags:
        accent = _SEVERITY_ACCENT.get(f.severity, "#64748B")
        values = ""
        if f.pdf_value is not None and f.sheet_value is not None:
            values = (
                f'<div style="font-size:0.82rem;color:{_SUBINK};margin-top:3px;">'
                f"Pitch deck: {f.pdf_value:,.0f} &nbsp;·&nbsp; Sheet: {f.sheet_value:,.0f}</div>"
            )
        rows.append(f"""
        <tr>
          <td style="padding:10px 14px;border-left:4px solid {accent};white-space:nowrap;">
            <span style="color:{accent};font-weight:700;">{f.severity.title()}</span>
          </td>
          <td style="padding:10px 14px;font-weight:600;color:{_INK};white-space:nowrap;">{f.metric}</td>
          <td style="padding:10px 14px;color:{_SUBINK};">{f.description}{values}</td>
        </tr>""")

    return f"""
    <table style="width:100%;border-collapse:collapse;background:{_PANEL};
                  border:1px solid {_BORDER};border-radius:10px;overflow:hidden;">
      <thead>
        <tr style="background:#F8FAFC;text-align:left;">
          <th style="padding:10px 14px;color:{_SUBINK};font-size:0.8rem;">SEVERITY</th>
          <th style="padding:10px 14px;color:{_SUBINK};font-size:0.8rem;">METRIC</th>
          <th style="padding:10px 14px;color:{_SUBINK};font-size:0.8rem;">DETAIL</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _mock_warning_banner_html(action_statuses: list[ActionStatus]) -> str:
    mocks = [a for a in action_statuses if a.status == "mock"]
    if not mocks:
        return ""
    items = "".join(f'<li style="margin:2px 0;">{a.detail}</li>' for a in mocks)
    return f"""
    <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;
                padding:12px 16px;margin-bottom:14px;color:#92400E;">
      <strong>⚠️ Running in mock mode — no real external side effects occurred.</strong>
      <ul style="margin:6px 0 0 18px;padding:0;font-size:0.9rem;">{items}</ul>
    </div>"""


def _pdf_preview_html(pdf_path: str | None) -> str:
    if not pdf_path or not Path(pdf_path).exists():
        return f'<div style="padding:16px;color:{_SUBINK};">No report generated for this run.</div>'
    b64 = base64.b64encode(Path(pdf_path).read_bytes()).decode()
    return (
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'style="width:100%;height:640px;border:1px solid {_BORDER};border-radius:10px;"></iframe>'
    )


def _metrics_summary_html(result: PipelineRunResult) -> str:
    m = result.analysis.financial_metrics
    runway_text = format_runway(m.runway_months, m.runway_status)
    burn_multiple_text = "N/A" if m.burn_multiple is None else f"{m.burn_multiple:,.2f}"
    if m.burn_multiple is not None and m.burn_multiple_approximate:
        burn_multiple_text += " *"

    def _cell(label: str, value: str) -> str:
        return (
            f'<div style="padding:12px 14px;border:1px solid {_BORDER};border-radius:10px;background:{_PANEL};">'
            f'<div style="font-size:0.75rem;color:{_SUBINK};">{label}</div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:{_INK};margin-top:2px;">{value}</div></div>'
        )

    cells = [
        _cell("MRR", "N/A" if m.mrr is None else f"${m.mrr:,.0f}"),
        _cell("ARR", "N/A" if m.arr is None else f"${m.arr:,.0f}"),
        _cell("Burn Rate", "N/A" if m.burn_rate is None else f"${m.burn_rate:,.0f}/mo"),
        _cell("Runway", runway_text),
        _cell("Burn Multiple", burn_multiple_text),
    ]
    note = (
        f'<div style="font-size:0.78rem;color:{_SUBINK};margin-top:6px;">'
        "* approximate — no Net New ARR / prior-period ARR on the sheet, computed against total ARR instead.</div>"
        if (m.burn_multiple is not None and m.burn_multiple_approximate) else ""
    )
    return f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;">{"".join(cells)}</div>{note}'


def _header_html() -> str:
    """Banner: dark navy-to-teal gradient, a small hand-drawn bar-chart
    mark (no emoji — a deliberate mark reads more like a real product),
    and a row of capability chips that double as an at-a-glance
    architecture summary."""
    chips = ["Hybrid BM25 + FAISS RAG", "Grounded Red-Flag Detection", "Automated Memo · Email · Sheet", "LLM-as-Judge Eval"]
    chips_html = "".join(
        f'<span style="padding:5px 12px;border-radius:999px;background:rgba(255,255,255,0.08);'
        f'color:#E4E7EB;font-size:0.76rem;font-weight:600;border:1px solid rgba(255,255,255,0.14);'
        f'white-space:nowrap;">{c}</span>'
        for c in chips
    )
    return f"""
    <div style="background:linear-gradient(135deg, {_NAVY_DARK} 0%, {_NAVY} 45%, {_TEAL_DARK} 100%);
                border-radius:16px;padding:26px 30px;box-shadow:0 10px 28px rgba(11,18,32,0.22);">
      <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
        <div style="flex-shrink:0;width:54px;height:54px;border-radius:14px;
                    background:rgba(255,255,255,0.08);display:flex;align-items:center;justify-content:center;">
          <svg width="28" height="28" viewBox="0 0 30 30" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="2" y="17" width="5" height="11" rx="1.4" fill="{_GOLD}"/>
            <rect x="12.5" y="10" width="5" height="18" rx="1.4" fill="{_GOLD_LIGHT}"/>
            <rect x="23" y="3" width="5" height="25" rx="1.4" fill="#FDE68A"/>
          </svg>
        </div>
        <div style="min-width:240px;">
          <div style="color:#FFFFFF;font-size:1.8rem;font-weight:800;letter-spacing:-0.02em;line-height:1.15;">VentureAgent</div>
          <div style="color:#B9C3CE;font-size:0.95rem;margin-top:3px;">Autonomous Startup Valuation &amp; Due Diligence Platform</div>
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:16px;">{chips_html}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _metrics_bar_chart(result: PipelineRunResult) -> go.Figure:
    m = result.analysis.financial_metrics
    labels, values = [], []
    for label, val in [("MRR", m.mrr), ("ARR", m.arr), ("Burn Rate", m.burn_rate), ("Cash Balance", m.cash_balance)]:
        if val is not None:
            labels.append(label)
            values.append(val)
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=_TEAL)])
    fig.update_layout(title="Key Financial Metrics ($)", yaxis_title="USD", template="plotly_white",
                       margin=dict(t=48, l=10, r=10, b=10))
    return fig


def _pdf_vs_sheet_chart(result: PipelineRunResult) -> go.Figure:
    flags = result.analysis.red_flags
    if not flags:
        fig = go.Figure()
        fig.update_layout(title="No discrepancies detected between Pitch Deck and Financial Sheet", template="plotly_white")
        return fig
    fig = go.Figure(data=[
        go.Bar(name="Pitch Deck (claimed)", x=[f.metric for f in flags], y=[f.pdf_value for f in flags], marker_color=_GOLD),
        go.Bar(name="Financial Sheet (verified)", x=[f.metric for f in flags], y=[f.sheet_value for f in flags], marker_color=_TEAL),
    ])
    fig.update_layout(title="Pitch Deck Claims vs. Financial Sheet (Discrepancies)", barmode="group",
                       template="plotly_white", margin=dict(t=48, l=10, r=10, b=10))
    return fig


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


def _resolve_sheet_source(sheet_mode: str, sheet_id: str, csv_file) -> tuple[str, str]:
    """Returns (sheet_source, source_label_for_logging).

    Precedence follows whichever mode the user selected — not just
    "whichever field happens to be non-empty" — so switching the radio
    back and forth can't silently pick up a stale value left in the other
    field. Raises gr.Error for the one truly ambiguous case: CSV mode
    selected but nothing uploaded.
    """
    if sheet_mode == "Upload CSV File":
        if csv_file is None:
            raise gr.Error("You selected 'Upload CSV File' but didn't upload one — please upload a .csv, or switch to 'Google Sheet URL/ID'.")
        return csv_file.name, "uploaded CSV"

    # sheet_mode == "Google Sheet URL/ID"
    if sheet_id and sheet_id.strip():
        return sheet_id.strip(), "Google Sheet URL/ID"
    return _SAMPLE_SHEET_PATH, "bundled sample data (no sheet ID/URL given)"


def run_ventureagent(pitch_deck_file, sheet_mode, sheet_id, csv_file, company_name, sector):
    if pitch_deck_file is None or not company_name.strip():
        raise gr.Error("Please upload a pitch deck PDF and enter a company name.")

    sheet_source, source_label = _resolve_sheet_source(sheet_mode, sheet_id, csv_file)
    logger.info("Financial data source resolved to: %s (%s)", source_label, sheet_source)

    session_dir = _BASE_TMP / str(uuid.uuid4())
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_pipeline(
            pitch_deck_path=pitch_deck_file.name,
            sheet_source=sheet_source,
            company_name=company_name.strip(),
            session_dir=session_dir,
            sector=sector or "software",
        )
    except Exception as exc:
        logger.exception("Gradio run failed: %s", exc)
        raise gr.Error(f"Pipeline failed: {exc}") from exc

    badge_html = _verdict_badge_html(result.analysis.decision, result.company_name)
    summary_md = result.analysis.executive_summary
    flags_html = _flags_table_html(result.analysis.red_flags)
    metrics_html = _metrics_summary_html(result)

    metrics_fig = _metrics_bar_chart(result)
    discrepancy_fig = _pdf_vs_sheet_chart(result)

    warning_banner = _mock_warning_banner_html(result.action_statuses)
    action_rows = [[a.action, a.status, a.detail] for a in result.action_statuses]
    step_rows = [[s.step, s.status, round(s.latency_ms, 1), s.detail] for s in result.step_logs]
    action_df = pd.DataFrame(action_rows, columns=["Action", "Status", "Detail"])
    step_df = pd.DataFrame(step_rows, columns=["Step", "Status", "Latency (ms)", "Detail"])

    report_file = result.report_path if result.report_path and Path(result.report_path).exists() else None
    pdf_preview = _pdf_preview_html(result.report_path)

    return (badge_html, summary_md, metrics_html, flags_html, metrics_fig, discrepancy_fig,
            warning_banner, action_df, step_df, report_file, pdf_preview)


def _toggle_sheet_source_inputs(mode: str):
    is_csv = mode == "Upload CSV File"
    return gr.update(visible=not is_csv), gr.update(visible=is_csv)


with gr.Blocks(title="VentureAgent — Autonomous Startup Valuation & Due Diligence", theme=_THEME, css=_CSS) as demo:
    gr.HTML(_header_html())

    with gr.Row():
        with gr.Column(scale=1):
            pitch_deck_input = gr.File(label="Pitch Deck PDF", file_types=[".pdf"])

            sheet_mode_input = gr.Radio(
                ["Google Sheet URL/ID", "Upload CSV File"],
                value="Google Sheet URL/ID",
                label="Financial Data Source",
            )
            sheet_input = gr.Textbox(
                label="Google Sheet URL/ID (leave blank to use bundled sample data)",
                placeholder="1AbC...xyz or full URL",
                visible=True,
            )
            csv_file_input = gr.File(
                label="Financial CSV — either a simple Metric,Value sheet, or a wide, "
                      "one-row-per-fiscal-year statement export (revenue/cash/burn columns auto-detected)",
                file_types=[".csv"], visible=False,
            )

            company_input = gr.Textbox(label="Target Company Name", placeholder="e.g. NovaGrid AI")
            sector_input = gr.Textbox(label="Sector (for market comps)", value="software")
            run_btn = gr.Button("Run VentureAgent Pipeline", variant="primary")
            gr.Markdown(
                f"LLM provider: `{settings.llm_provider}` · Embedding provider: `{settings.embedding_provider}`\n\n"
                "No pitch deck handy? Try the bundled `sample_data/sample_pitch_deck.pdf` "
                "— it deliberately conflicts with the sample financial sheet to demo red-flag detection. "
                "For a wide, multi-year SEC-style export, try `sample_data/sample_wide_format_financials.csv` "
                "(the most recent fiscal year is used automatically).",
                elem_classes=["va-hint"],
            )

            sheet_mode_input.change(
                fn=_toggle_sheet_source_inputs,
                inputs=[sheet_mode_input],
                outputs=[sheet_input, csv_file_input],
            )

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Executive Summary & Red Flags"):
                    verdict_out = gr.HTML()
                    summary_out = gr.Markdown()
                    metrics_summary_out = gr.HTML()
                    gr.Markdown("**Red Flags**")
                    flags_out = gr.HTML()
                with gr.Tab("Financial Charts"):
                    metrics_chart_out = gr.Plot(label="Key Metrics")
                    discrepancy_chart_out = gr.Plot(label="Pitch Deck vs. Sheet")
                with gr.Tab("Action Logs"):
                    warning_out = gr.HTML()
                    action_out = gr.Dataframe(label="Automated Action Status (Email / Sheet / PDF)")
                    step_out = gr.Dataframe(label="Pipeline Step Latency & Status")
                with gr.Tab("PDF Report"):
                    report_out = gr.File(label="Download Investment_Memo.pdf")
                    pdf_preview_out = gr.HTML()

    run_btn.click(
        fn=run_ventureagent,
        inputs=[pitch_deck_input, sheet_mode_input, sheet_input, csv_file_input, company_input, sector_input],
        outputs=[verdict_out, summary_out, metrics_summary_out, flags_out, metrics_chart_out, discrepancy_chart_out,
                 warning_out, action_out, step_out, report_out, pdf_preview_out],
    )

# Queueing for HF Spaces multi-user concurrency, per the spec.
demo.queue(default_concurrency_limit=settings.gradio_concurrency_limit)

if __name__ == "__main__":
    demo.launch()
