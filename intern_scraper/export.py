"""Module 4 - Output Generator.

Turns a list of :class:`ParsedListing` into a Pandas DataFrame (for CSV/JSON
dumps) and into a self-contained HTML page with a strikethrough-checkbox UX.

The HTML template is rendered with Jinja2 and persists "applied" state in
``localStorage`` so re-opening the file keeps your progress.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from typing import Iterable, List

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .schemas import ParsedListing


log = logging.getLogger(__name__)


TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


def to_dataframe(listings: Iterable[ParsedListing]) -> pd.DataFrame:
    rows = [li.model_dump() for li in listings]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize the date column so sorting works even when LLM returns mixed
    # representations.
    df["post_date_parsed"] = pd.to_datetime(df.get("post_date"), errors="coerce")

    # Deduplicate on (company, role_title, application_link).
    df["__dedupe_key"] = (
        df["company"].astype(str).str.lower().str.strip() + "|"
        + df["role_title"].astype(str).str.lower().str.strip() + "|"
        + df["application_link"].astype(str).str.lower().str.strip()
    )
    df = df.drop_duplicates(subset="__dedupe_key", keep="first")
    df = df.drop(columns="__dedupe_key")

    # Sort by post date desc (NaT goes last), then by confidence desc.
    df = df.sort_values(
        by=["post_date_parsed", "confidence"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    return df


def _row_key(row: dict) -> str:
    raw = f"{row.get('company','')}|{row.get('role_title','')}|{row.get('application_link','')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def render_html(df: pd.DataFrame, query: str, output_path: str) -> str:
    """Render the DataFrame to a standalone HTML page and return the path."""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("results.html")

    rows: List[dict] = []
    if not df.empty:
        for _, r in df.iterrows():
            d = r.to_dict()
            d["key"] = _row_key(d)
            # Keep the human-friendly post_date string, not the pd.Timestamp.
            d["post_date"] = (
                d.get("post_date") if d.get("post_date")
                else (r["post_date_parsed"].date().isoformat()
                      if pd.notna(r.get("post_date_parsed")) else "")
            )
            d["confidence"] = float(d.get("confidence") or 0.0)
            rows.append(d)

    html = template.render(
        query=query,
        rows=rows,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Wrote HTML report (%d rows) to %s", len(rows), output_path)
    return output_path


def write_csv(df: pd.DataFrame, output_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    # Drop the helper column before writing.
    if "post_date_parsed" in df.columns:
        df = df.drop(columns="post_date_parsed")
    df.to_csv(output_path, index=False)
    return output_path
