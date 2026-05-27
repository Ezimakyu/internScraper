"""Pydantic schemas shared across modules.

These models are the single source of truth for what data flows between the
Config -> Scraper -> Parser -> Export pipeline. They are intentionally simple
so the LLM can fill them via structured outputs.
"""

from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


JobSite = Literal["linkedin", "indeed", "glassdoor", "zip_recruiter", "google"]


class QueryConfig(BaseModel):
    """Structured form of a natural-language user query.

    The user types something like ``"aerospace software internships summer 2026"``
    and the Config module asks the LLM to break it into the fields below. The
    Scraper module then uses these fields to build a JobSpy / Google query.
    """

    model_config = ConfigDict(extra="ignore")

    raw_query: str = Field(..., description="The original user input verbatim.")
    role_keywords: List[str] = Field(
        default_factory=lambda: ["software engineer intern"],
        description="Job titles or role phrases to search for.",
    )
    industry_keywords: List[str] = Field(
        default_factory=list,
        description="Industry / domain words (e.g. aerospace, fintech, robotics).",
    )
    season: Optional[str] = Field(
        default=None,
        description="Internship season such as 'Summer', 'Fall', 'Spring', or 'Winter'.",
    )
    year: Optional[int] = Field(
        default=None,
        description="Target year (e.g. 2026). Used to filter expired postings.",
    )
    location: str = Field(
        default="USA",
        description="Free-text location filter. Defaults to USA per project spec.",
    )
    target_level: Literal["undergraduate", "graduate", "any"] = Field(
        default="undergraduate",
        description="Education level filter.",
    )
    results_per_site: int = Field(
        default=50, ge=1, le=200,
        description="How many results JobSpy should pull per board.",
    )
    hours_old: int = Field(
        default=24 * 30, ge=1,
        description="JobSpy 'hours_old' filter; default ~30 days of postings.",
    )
    extra_sites: List[str] = Field(
        default_factory=list,
        description="Extra GitHub / company URLs the user wants Path B to scrape.",
    )


class RawListing(BaseModel):
    """A pre-LLM-parse listing.

    Both Path A (JobSpy rows) and Path B (Playwright text chunks) get normalized
    into this shape so the Parser doesn't care about the source.
    """

    model_config = ConfigDict(extra="ignore")

    source: Literal["jobspy", "web"] = "web"
    source_url: Optional[str] = None
    site: Optional[str] = None
    text: str = Field(..., description="Raw or semi-structured text to parse.")
    # Optional structured hints from JobSpy (avoid re-asking the LLM).
    hint_company: Optional[str] = None
    hint_title: Optional[str] = None
    hint_location: Optional[str] = None
    hint_date: Optional[str] = None
    hint_url: Optional[str] = None


class ParsedListing(BaseModel):
    """The structured listing schema we force the LLM to produce."""

    model_config = ConfigDict(extra="ignore")

    company: str = Field(..., description="Hiring company.")
    role_title: str = Field(..., description="Job / internship title.")
    location: Optional[str] = Field(
        default=None, description="City, State, Country or 'Remote'."
    )
    post_date: Optional[str] = Field(
        default=None,
        description="ISO-8601 date string (YYYY-MM-DD) of when the role was posted "
                    "or last updated. Null if unknown.",
    )
    application_link: str = Field(
        ..., description="Direct apply URL or canonical listing URL."
    )
    is_undergraduate: bool = Field(
        ...,
        description="True if undergraduates are eligible (no Masters/PhD requirement).",
    )
    is_usa_based: bool = Field(
        ..., description="True if the role is in the USA or USA-remote."
    )
    is_past_date: bool = Field(
        ...,
        description="True if the application window has clearly closed or the "
                    "season is in the past relative to today.",
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="LLM self-reported confidence the extraction is correct.",
    )
    notes: Optional[str] = Field(
        default=None, description="Short reason/notes (kept for debugging)."
    )


class ParsedListings(BaseModel):
    """Wrapper used as the response_format for structured-output calls."""

    model_config = ConfigDict(extra="ignore")

    listings: List[ParsedListing] = Field(default_factory=list)


def today_iso() -> str:
    return date.today().isoformat()
