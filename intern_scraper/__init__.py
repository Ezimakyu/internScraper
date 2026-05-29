"""LLM-Powered Internship Scraper.

A small toolkit that combines fast structured scraping (JobSpy + Google search +
Playwright) with an LLM-driven parser (OpenAI gpt-4o-mini by default) to surface
USA-based undergraduate-friendly software / tech internships.
"""

from .schemas import ParsedListing, QueryConfig, RawListing

__all__ = ["ParsedListing", "QueryConfig", "RawListing"]

__version__ = "0.1.0"
