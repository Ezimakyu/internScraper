"""Module 3 - LLM Filter & Parser.

Takes :class:`RawListing` objects and asks ``gpt-4o-mini`` to produce a list of
:class:`ParsedListing` rows using structured outputs. Past / non-USA /
non-undergrad rows are dropped here.

Implementation notes:

* JobSpy rows are already structured, so we mostly use the LLM to apply the
  *filters* (Is Undergraduate / Is USA / Is Past Date). We still pass it the
  raw text in case the description contradicts the title.
* Web text chunks can be very large (a full README is 100k chars). We slice
  them into ~6000-char windows so each LLM call stays cheap and fast.
* All calls go through the structured-output endpoint with a Pydantic schema,
  which makes invalid JSON essentially impossible.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import List, Optional

from openai import OpenAI

from .llm import get_client, model_name
from .schemas import ParsedListing, ParsedListings, QueryConfig, RawListing, today_iso


log = logging.getLogger(__name__)


CHUNK_CHARS = 6000
CHUNK_OVERLAP = 400
MAX_CHUNKS_PER_URL = 8  # cap to keep cost predictable


_SYSTEM_PROMPT_TMPL = """You extract structured internship postings from messy text.

Today's date is {today}. The user is searching for: "{query}".
Defaults: USA-based, {level} eligible, season={season}, year={year}.

For each *distinct* internship/co-op posting in the input, output one row with:
- company: company name
- role_title: e.g. "Software Engineer Intern"
- location: "City, ST" or "Remote (US)" or null
- post_date: ISO YYYY-MM-DD if you can determine it, else null
- application_link: the most specific apply URL you can find in the text
- is_undergraduate: true unless the posting explicitly requires graduate enrollment
- is_usa_based: true if the role is in the United States or US-remote
- is_past_date: true if the role's season/year has clearly passed, or if the
  text explicitly says "closed" / "expired" / "filled" / "no longer accepting"
- confidence: 0.0-1.0 self-assessment
- notes: short rationale (<=120 chars), optional

Hard rules:
- Do NOT invent links. If you can't find an apply URL, use the source URL.
- Do NOT output marketing pages, team pages, or non-internship roles.
- Skip duplicates within the same input.
- If the input has no real postings, return an empty list.
"""


def _build_system_prompt(config: QueryConfig) -> str:
    return _SYSTEM_PROMPT_TMPL.format(
        today=today_iso(),
        query=config.raw_query,
        level=config.target_level,
        season=config.season or "any",
        year=config.year or "any",
    )


def _chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text or ""
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
        if len(chunks) >= MAX_CHUNKS_PER_URL:
            break
    return chunks


def _normalize_url(listing: ParsedListing, raw: RawListing) -> None:
    """Patch obvious LLM URL hallucinations using the raw hint."""

    if not listing.application_link or listing.application_link.strip() in {"", "null", "None"}:
        if raw.hint_url:
            listing.application_link = raw.hint_url
        elif raw.source_url:
            listing.application_link = raw.source_url


def _parse_one(raw: RawListing, config: QueryConfig, client: OpenAI, model: str) -> List[ParsedListing]:
    """Run a single raw listing through the LLM and return parsed rows."""

    system = _build_system_prompt(config)
    out: List[ParsedListing] = []

    chunks = _chunk_text(raw.text)
    for i, chunk in enumerate(chunks):
        user_msg = (
            f"Source URL: {raw.source_url or 'unknown'}\n"
            f"Source: {raw.source}\n"
            f"Hints: company={raw.hint_company!r} title={raw.hint_title!r} "
            f"location={raw.hint_location!r} date={raw.hint_date!r} url={raw.hint_url!r}\n\n"
            f"--- BEGIN INPUT (chunk {i+1}/{len(chunks)}) ---\n{chunk}\n--- END INPUT ---"
        )
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                response_format=ParsedListings,
                temperature=0.0,
            )
            parsed = completion.choices[0].message.parsed
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM parse failed for %s chunk %d: %s",
                        raw.source_url, i, exc)
            continue

        if not parsed or not parsed.listings:
            continue
        for listing in parsed.listings:
            _normalize_url(listing, raw)
            out.append(listing)

        # Short-circuit: JobSpy rows are 1 record, no need for more chunks.
        if raw.source == "jobspy":
            break

    return out


def parse_listings(
    raws: List[RawListing],
    config: QueryConfig,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> List[ParsedListing]:
    """Parse every raw listing into ParsedListings. Best-effort, never raises."""

    if not raws:
        return []
    client = client or get_client()
    mdl = model or model_name()

    all_out: List[ParsedListing] = []
    for raw in raws:
        all_out.extend(_parse_one(raw, config, client, mdl))
    log.info("LLM produced %d candidate parsed listings.", len(all_out))
    return all_out


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%b %d, %Y", "%B %d, %Y", "%d %b %Y",
]


def _try_parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def filter_listings(
    listings: List[ParsedListing],
    config: QueryConfig,
    drop_past: bool = True,
) -> List[ParsedListing]:
    """Apply the project's default filters (USA / undergrad / not expired)."""

    today = date.today()
    out: List[ParsedListing] = []
    for li in listings:
        if drop_past and li.is_past_date:
            continue
        if config.target_level == "undergraduate" and not li.is_undergraduate:
            continue
        if "USA" in config.location.upper() and not li.is_usa_based:
            continue
        # Sanity: if the LLM gave us a post_date that's > 18 months old, drop it.
        d = _try_parse_date(li.post_date)
        if d is not None and (today - d).days > 540:
            continue
        out.append(li)
    log.info("Kept %d / %d listings after filters.", len(out), len(listings))
    return out
