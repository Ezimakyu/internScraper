"""Module 1 - Input & Configuration Parser.

Takes a free-form natural language query (e.g. "aerospace software internships
summer 2026") and turns it into a :class:`QueryConfig` we can hand to the
Scraper. A short LLM call is used to do the parsing because heuristic regex
work tends to break on creative phrasings.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from .llm import get_client, model_name
from .schemas import QueryConfig


log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You convert a user's free-form internship search request \
into a structured JSON object. Be conservative: only set fields when the user \
clearly implies them. The user is searching for tech / software internships \
unless they say otherwise.

Rules:
- "role_keywords" should be short job-title phrases the user would actually \
type into LinkedIn search (e.g. "software engineer intern", "ml intern"). \
Always include at least one.
- "industry_keywords" are domain words (aerospace, defense, fintech, robotics, \
biotech, gaming, computer vision, etc.). Empty list if none implied.
- "season" is one of Summer / Fall / Spring / Winter, or null.
- "year" is a 4-digit int or null.
- "location" defaults to "USA" unless the user names a region/state/city.
- "target_level" is "undergraduate" unless the user says otherwise.
- Do not invent companies or schools.
"""


def _fallback_parse(raw_query: str) -> QueryConfig:
    """A regex-only fallback used when the LLM is unavailable.

    It deliberately keeps role keywords broad so we still hit JobSpy with
    something sensible.
    """

    q = raw_query.lower()
    season = None
    for s in ("summer", "fall", "spring", "winter"):
        if s in q:
            season = s.capitalize()
            break

    year_match = re.search(r"(20\d{2})", raw_query)
    year = int(year_match.group(1)) if year_match else None

    industry = []
    for kw in (
        "aerospace", "defense", "robotics", "fintech", "biotech",
        "gaming", "computer vision", "ml", "machine learning", "ai",
        "embedded", "hardware", "quant",
    ):
        if kw in q:
            industry.append(kw)

    role_keywords = ["software engineer intern"]
    if "ml" in q or "machine learning" in q or "ai " in q:
        role_keywords.append("machine learning intern")
    if "data" in q:
        role_keywords.append("data science intern")
    if "hardware" in q or "embedded" in q:
        role_keywords.append("hardware engineer intern")

    return QueryConfig(
        raw_query=raw_query,
        role_keywords=role_keywords,
        industry_keywords=industry,
        season=season,
        year=year,
    )


def parse_query(
    raw_query: str,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> QueryConfig:
    """Convert a natural-language query into a :class:`QueryConfig`.

    Uses the OpenAI structured-output API (``responses.parse``-style via
    ``chat.completions.parse``). Falls back to a regex parser if the LLM call
    fails for any reason, so the CLI is still usable offline.
    """

    raw_query = raw_query.strip()
    if not raw_query:
        raise ValueError("query is empty")

    try:
        client = client or get_client()
        completion = client.beta.chat.completions.parse(
            model=model or model_name(),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": raw_query},
            ],
            response_format=QueryConfig,
            temperature=0.0,
        )
        cfg = completion.choices[0].message.parsed
        if cfg is None:
            raise RuntimeError("LLM returned no parsed config")
        # Always re-attach the original query.
        cfg.raw_query = raw_query
        log.debug("Parsed config from LLM: %s", json.dumps(cfg.model_dump(), default=str))
        return cfg
    except Exception as exc:  # noqa: BLE001 - we want a wide net here
        log.warning("LLM config parse failed (%s); using regex fallback.", exc)
        return _fallback_parse(raw_query)
