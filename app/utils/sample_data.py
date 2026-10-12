"""Synthetic demo data for a fictional startup ("NovaGrid AI"). The pitch
deck deliberately overstates MRR ($50K) relative to the financial sheet
($30K) so the discrepancy/red-flag detection has something real to catch
out of the box — mirrors the worked example in the project spec.
"""
from __future__ import annotations

PITCH_DECK_TEXT = """NovaGrid AI — Series A Pitch Deck

About Us
NovaGrid AI builds predictive-maintenance software for industrial IoT
fleets, using sensor data and machine learning to flag equipment failures
before they happen. We are raising a $4M Series A to expand our sales team
and accelerate product development.

Traction
Our Monthly Recurring Revenue (MRR) has grown to $50,000 as of this
quarter, up from $22,000 a year ago. Annual Recurring Revenue (ARR) now
stands at $600,000. We serve 34 mid-market industrial customers across
North America and Europe.

Market Opportunity
We estimate our Total Addressable Market (TAM) at $12,000,000,000. Our
Serviceable Addressable Market (SAM) is $1,800,000,000, focused on
mid-market manufacturers in North America and Europe. Our near-term
Serviceable Obtainable Market (SOM) is $90,000,000, representing realistic
penetration over the next three years given our current go-to-market
capacity.

Team & Financials
The founding team has previously scaled two industrial SaaS companies to
acquisition. Our current Burn Rate is $45,000 per month, and we maintain a
Cash Balance of $540,000, giving us a healthy runway to hit our next
milestones. We believe our capital efficiency and enterprise-grade
reliability make NovaGrid AI a compelling Series A investment.

Competitive Landscape
Compared to legacy SCADA vendors, NovaGrid AI offers a modern, cloud-native
architecture with 10x faster deployment times and predictive (not just
reactive) alerting.
"""

# NOTE the intentional MRR discrepancy vs. the pitch deck above:
# pitch deck claims $50,000 MRR, the sheet (source of truth) says $30,000.
FINANCIAL_SHEET_ROWS = [
    ("MRR", 30000),
    ("ARR", 600000),
    ("Burn Rate", 45000),
    ("Cash Balance", 540000),
    ("TAM", 12000000000),
    ("SAM", 1800000000),
    ("SOM", 90000000),
]


def write_sample_files(sample_data_dir) -> None:
    """(Re)generates sample_data/sample_financials.csv from the rows above.
    The pitch deck PDF is generated separately via reportlab — see
    scripts/generate_sample_pdf.py."""
    import csv
    from pathlib import Path

    sample_data_dir = Path(sample_data_dir)
    sample_data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sample_data_dir / "sample_financials.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Metric", "Value"])
        writer.writerows(FINANCIAL_SHEET_ROWS)


if __name__ == "__main__":
    from pathlib import Path

    write_sample_files(Path(__file__).resolve().parent.parent.parent / "sample_data")
    print("sample_financials.csv written")
