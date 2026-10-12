"""Step 3: Automated Actions — PDF memo generation, email dispatch, and
Google Sheet / dashboard update.

Every action degrades gracefully: with no SMTP/SendGrid credentials, email
"sends" are logged and returned as status=mock; with no Google service
account, the dashboard row is appended to a local CSV instead. This keeps
the pipeline runnable end-to-end for grading/demo purposes while the real
integration code path is fully implemented for a live deployment.
"""
from __future__ import annotations

import csv
import json
import smtplib
from datetime import datetime, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.config import settings
from app.schemas import ActionStatus, AnalysisResult
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    # sendgrid's client raises this (and subclasses like BadRequestsError,
    # UnauthorizedError) for any non-2xx API response. Imported at module
    # level, guarded, so the name is always bound even if sendgrid isn't
    # installed — an `except (..., PythonHTTPError)` clause must be able to
    # resolve the name regardless of which branch actually failed.
    from python_http_client.exceptions import HTTPError as PythonHTTPError
except ImportError:
    class PythonHTTPError(Exception):  # type: ignore[no-redef]
        """Placeholder so `except PythonHTTPError` is always valid even
        when the sendgrid/python_http_client package isn't installed."""


def generate_pdf_report(analysis: AnalysisResult, output_path: str | Path) -> Path:
    """Builds a clean, multi-page Investment_Memo.pdf with reportlab.
    Citations are appended strictly at the end, per the spec's UI policy."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Investment Memo — {analysis.company_name}", styles["Title"]))
    story.append(Paragraph(datetime.now(timezone.utc).strftime("Generated %Y-%m-%d %H:%M UTC"), styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    story.append(Paragraph(analysis.executive_summary, styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Financial Metrics", styles["Heading1"]))
    m = analysis.financial_metrics
    from app.pipeline.valuation import format_runway

    burn_multiple_display = "N/A" if m.burn_multiple is None else f"{m.burn_multiple:,.2f}"
    if m.burn_multiple is not None and m.burn_multiple_approximate:
        burn_multiple_display += "  (approximate — no Net New ARR / prior-period ARR on sheet)"

    metric_rows = [["Metric", "Value"]]
    for label, val in [
        ("MRR", m.mrr), ("ARR", m.arr), ("Burn Rate", m.burn_rate), ("Cash Balance", m.cash_balance),
        ("Runway", format_runway(m.runway_months, m.runway_status)), ("Burn Multiple", burn_multiple_display),
        ("TAM", m.tam), ("SAM", m.sam), ("SOM", m.som),
        ("TAM/SAM/SOM Consistent", m.tam_sam_som_consistent),
    ]:
        metric_rows.append([label, "N/A" if val is None else f"{val:,.2f}" if isinstance(val, (int, float)) else str(val)])
    table = Table(metric_rows, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Red Flags & Risk Assessment", styles["Heading1"]))
    if analysis.red_flags:
        flag_rows = [["Severity", "Metric", "Description"]]
        for f in analysis.red_flags:
            flag_rows.append([f.severity, f.metric, f.description])
        flag_table = Table(flag_rows, hAlign="LEFT", colWidths=[60, 70, 330])
        flag_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7f1d1d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(flag_table)
    else:
        story.append(Paragraph("No red flags detected.", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Investment Decision", styles["Heading1"]))
    story.append(Paragraph(f"<b>{analysis.decision}</b>", styles["Normal"]))
    if analysis.groundedness_score is not None:
        story.append(Paragraph(f"Groundedness score (LLM-as-judge, 1-5): {analysis.groundedness_score:.1f}", styles["Normal"]))

    story.append(PageBreak())
    story.append(Paragraph("Sources & Citations", styles["Heading1"]))
    for fact in analysis.extracted_facts:
        story.append(Paragraph(f"[{fact.source}] {fact.metric}: {fact.raw_text}", styles["Normal"]))
    for c in analysis.citations:
        story.append(Paragraph(f"- {c}", styles["Normal"]))

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    doc.build(story)
    return output_path


def send_email_alert(analysis: AnalysisResult, pdf_path: str | Path) -> ActionStatus:
    """Sends (or mocks) the investment-team summary email with the memo attached."""
    recipients = [addr.strip() for addr in settings.investment_team_emails.split(",") if addr.strip()]
    subject = f"VentureAgent Analysis — {analysis.company_name} — Verdict: {analysis.decision}"
    body = f"{analysis.executive_summary}\n\nRed flags detected: {len(analysis.red_flags)}\nSee attached memo for full detail."

    if settings.sendgrid_api_key:
        try:
            import base64

            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Attachment, Mail

            pdf_bytes = Path(pdf_path).read_bytes()
            message = Mail(from_email=settings.smtp_from_addr, to_emails=recipients, subject=subject, plain_text_content=body)
            message.attachment = Attachment(
                file_content=base64.b64encode(pdf_bytes).decode(),
                file_name=Path(pdf_path).name,
                file_type="application/pdf",
                disposition="attachment",
            )
            SendGridAPIClient(settings.sendgrid_api_key).send(message)
            return ActionStatus(action="send_email", status="success", detail=f"Sent via SendGrid to {recipients}")
        except (ImportError, FileNotFoundError, PythonHTTPError) as exc:
            # ImportError: sendgrid not installed. FileNotFoundError: pdf_path
            # missing. PythonHTTPError: SendGrid's own client raises this for
            # any non-2xx API response (bad key, invalid sender, etc).
            logger.exception("SendGrid dispatch failed: %s", exc)
            return ActionStatus(action="send_email", status="failed", detail=str(exc))
        except Exception as exc:  # noqa: BLE001 — last-resort net; SendGrid's
            # client can raise transport-layer errors too. Full traceback logged.
            logger.exception("Unexpected SendGrid error: %s", exc)
            return ActionStatus(action="send_email", status="failed", detail=str(exc))

    if settings.smtp_host and settings.smtp_username and settings.smtp_password:
        try:
            msg = MIMEMultipart()
            msg["From"] = settings.smtp_from_addr
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            with open(pdf_path, "rb") as fh:
                part = MIMEApplication(fh.read(), Name=Path(pdf_path).name)
            part["Content-Disposition"] = f'attachment; filename="{Path(pdf_path).name}"'
            msg.attach(part)

            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                server.starttls()
                server.login(settings.smtp_username, settings.smtp_password)
                server.sendmail(settings.smtp_from_addr, recipients, msg.as_string())
            return ActionStatus(action="send_email", status="success", detail=f"Sent via SMTP to {recipients}")
        except (smtplib.SMTPException, OSError, FileNotFoundError) as exc:
            # SMTPException family: auth failures, refused recipients,
            # protocol errors. OSError: connection/socket failures.
            logger.exception("SMTP dispatch failed: %s", exc)
            return ActionStatus(action="send_email", status="failed", detail=str(exc))

    # No SMTP/SendGrid credentials configured — this is a MOCK send, not a
    # real one. Logged at WARNING (not INFO) and given an explicit ⚠️ prefix
    # in the detail so it can't be mistaken for a successful send when
    # surfaced in the UI's action-log table or the PDF memo.
    warning = (
        f"⚠️ MOCK MODE — no email was actually sent. The report was generated locally "
        f"at '{Path(pdf_path).name}', but SMTP_HOST/SENDGRID_API_KEY are not configured, "
        f"so no message reached {recipients}. Set them in .env to send for real."
    )
    logger.warning("MOCK EMAIL (not sent) to %s | subject=%r | attachment=%s", recipients, subject, Path(pdf_path).name)
    return ActionStatus(action="send_email", status="mock", detail=warning)


def update_google_sheet(analysis: AnalysisResult, dashboard_dir: str | Path) -> ActionStatus:
    """Appends [Timestamp, Company Name, Valuation, Red Flag Count, Verdict]
    to the live Google Sheet when credentials are available, else to a
    local dashboard.csv so the demo is still fully functional."""
    row = [
        datetime.now(timezone.utc).isoformat(),
        analysis.company_name,
        analysis.financial_metrics.arr or "N/A",
        len(analysis.red_flags),
        analysis.decision,
    ]

    live_attempt_failed_reason: str | None = None

    if settings.google_service_account_json:
        try:
            import json as _json

            import gspread
            from google.oauth2.service_account import Credentials

            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            blob = settings.google_service_account_json
            info = _json.loads(blob) if blob.strip().startswith("{") else _json.loads(Path(blob).read_text())
            creds = Credentials.from_service_account_info(info, scopes=scopes)
            gc = gspread.authorize(creds)
            # Expects a "Dashboard" sheet/tab to exist; real deployments
            # would pass the sheet key through explicitly.
            sheet = gc.open("VentureAgent Dashboard").sheet1
            sheet.append_row([str(v) for v in row])
            return ActionStatus(action="update_sheet", status="success", detail="Row appended to Google Sheet dashboard.")
        except (ImportError, FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
            logger.exception("Google Sheet update failed (config/credentials problem): %s", exc)
            live_attempt_failed_reason = str(exc)
        except Exception as exc:  # noqa: BLE001 — gspread/google-auth raise many
            # of their own exception classes (SpreadsheetNotFound, APIError,
            # RefreshError, ...); caught here as a last resort so one flaky
            # Sheets call can't crash the whole action step. Traceback logged.
            logger.exception("Unexpected Google Sheet update error: %s", exc)
            live_attempt_failed_reason = str(exc)
        # fall through to local CSV so the run still produces a record

    dashboard_dir = Path(dashboard_dir)
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dashboard_dir / "dashboard.csv"
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["Timestamp", "Company Name", "Valuation", "Red Flag Count", "Verdict"])
        writer.writerow(row)

    if live_attempt_failed_reason:
        detail = f"⚠️ Google Sheet update FAILED ({live_attempt_failed_reason}) — row appended to local dashboard instead: {csv_path}"
    else:
        detail = f"⚠️ MOCK MODE — GOOGLE_SERVICE_ACCOUNT_JSON not configured; row appended to local dashboard instead: {csv_path}"
    return ActionStatus(action="update_sheet", status="mock", detail=detail)
