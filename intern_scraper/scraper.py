"""Module 2 - The Gatherer.

Two parallel paths into the same :class:`RawListing` shape:

* **Path A (Big Boards):** JobSpy hits LinkedIn / Indeed / Glassdoor / ZipRecruiter
  with structured search terms. JobSpy already returns DataFrames, so we just
  normalize each row.
* **Path B (Deep Web & GitHub):** ``googlesearch-python`` finds niche company
  career pages and aggregator READMEs; Playwright then extracts ``innerText``.
  A small list of well-known aggregator GitHub READMEs is seeded so the tool is
  useful even if Google search is rate-limited.

Both paths return a list of :class:`RawListing` for the parser to chew on.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Iterable, List, Optional

from .schemas import QueryConfig, RawListing


log = logging.getLogger(__name__)


# Aggregator README sources we want to always check. These are the GitHub raw
# URLs because they're plain Markdown (cheap to fetch, easy for the LLM to
# parse) and they're maintained year-round by the community.
DEFAULT_GITHUB_SOURCES = [
    # SimplifyJobs / Pitt CSC summer internships - the canonical list.
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README-Off-Season.md",
    # Speedyapply / vanshb03 (popular alt list)
    "https://raw.githubusercontent.com/vanshb03/Summer2026-Internships/main/README.md",
]


# ---------------------------------------------------------------------------
# Path A: JobSpy
# ---------------------------------------------------------------------------

def _jobspy_term(config: QueryConfig) -> str:
    """Build the search string we feed JobSpy."""

    parts: List[str] = []
    parts.extend(config.role_keywords[:2] or ["software engineer intern"])
    if config.industry_keywords:
        parts.append(" ".join(config.industry_keywords[:3]))
    if config.season:
        parts.append(config.season)
    if config.year:
        parts.append(str(config.year))
    return " ".join(parts).strip()


def gather_jobspy(config: QueryConfig, sites: Optional[List[str]] = None) -> List[RawListing]:
    """Run JobSpy across the big job boards and normalize rows to RawListing.

    JobSpy is imported lazily so the rest of the CLI works in environments
    where the package isn't installed (e.g. unit tests).
    """

    try:
        from jobspy import scrape_jobs  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.warning("JobSpy unavailable (%s); skipping big-board path.", exc)
        return []

    site_name = sites or ["linkedin", "indeed", "glassdoor", "zip_recruiter"]
    term = _jobspy_term(config)
    log.info("JobSpy search: term=%r location=%r sites=%s",
             term, config.location, site_name)

    try:
        df = scrape_jobs(
            site_name=site_name,
            search_term=term,
            location=config.location,
            results_wanted=config.results_per_site,
            hours_old=config.hours_old,
            country_indeed="USA" if "USA" in config.location.upper() else None,
            linkedin_fetch_description=False,
            verbose=0,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("JobSpy call failed: %s", exc)
        return []

    if df is None or len(df) == 0:
        return []

    out: List[RawListing] = []
    for _, row in df.iterrows():
        get = lambda k: (row[k] if k in row and row[k] is not None else None)  # noqa: E731
        company = get("company")
        title = get("title")
        loc = get("location")
        post_date = get("date_posted")
        url = get("job_url") or get("job_url_direct")
        desc = get("description") or ""
        # Build a small "text" that the LLM can parse if it needs more context.
        text = (
            f"Company: {company}\n"
            f"Title: {title}\n"
            f"Location: {loc}\n"
            f"Posted: {post_date}\n"
            f"URL: {url}\n\n{desc}"
        )
        out.append(RawListing(
            source="jobspy",
            source_url=str(url) if url else None,
            site=str(get("site") or ""),
            text=text,
            hint_company=str(company) if company else None,
            hint_title=str(title) if title else None,
            hint_location=str(loc) if loc else None,
            hint_date=str(post_date) if post_date else None,
            hint_url=str(url) if url else None,
        ))
    log.info("JobSpy returned %d rows.", len(out))
    return out


# ---------------------------------------------------------------------------
# Path B: Google + Playwright
# ---------------------------------------------------------------------------

def _google_queries(config: QueryConfig) -> List[str]:
    """Compose a handful of focused Google queries."""

    role = config.role_keywords[0] if config.role_keywords else "software engineer intern"
    season = config.season or ""
    year = str(config.year) if config.year else ""
    industry = " ".join(config.industry_keywords[:2])
    qs = [
        f'"{role}" {season} {year} {industry} careers'.strip(),
        f'{role} {season} {year} site:greenhouse.io',
        f'{role} {season} {year} site:lever.co',
        f'{role} {season} {year} site:ashbyhq.com',
        f'{role} {industry} internship {year} USA',
    ]
    return [re.sub(r"\s+", " ", q).strip() for q in qs if q.strip()]


def discover_urls(config: QueryConfig, max_per_query: int = 5) -> List[str]:
    """Use googlesearch-python to find candidate career pages.

    Returns a de-duplicated list. Google sometimes throttles us; that's fine,
    we still have the curated GitHub sources to fall back on.
    """

    urls: List[str] = []
    try:
        from googlesearch import search  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.warning("googlesearch unavailable (%s); skipping web discovery.", exc)
        return urls

    for query in _google_queries(config):
        try:
            log.info("Google: %s", query)
            for url in search(query, num_results=max_per_query, lang="en"):
                if url and url not in urls:
                    urls.append(url)
        except Exception as exc:  # noqa: BLE001
            log.warning("Google query failed (%s): %s", query, exc)
            continue
    return urls


async def _fetch_one(url: str, timeout_ms: int = 20000) -> Optional[RawListing]:
    """Use Playwright to grab the innerText of a single page."""

    try:
        from playwright.async_api import async_playwright  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.warning("Playwright unavailable (%s); skipping %s", exc, url)
        return None

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                    )
                )
                page = await context.new_page()
                await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                # GitHub READMEs and JS-heavy career pages settle quickly enough
                # with networkidle, but we cap it so we don't hang forever.
                try:
                    await page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:  # noqa: BLE001
                    pass
                text = await page.evaluate("() => document.body && document.body.innerText")
            finally:
                await browser.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("Playwright fetch failed for %s: %s", url, exc)
        return None

    if not text or len(text.strip()) < 80:
        return None
    return RawListing(source="web", source_url=url, text=text.strip())


async def _fetch_many(urls: Iterable[str], concurrency: int = 4) -> List[RawListing]:
    sem = asyncio.Semaphore(concurrency)

    async def _bound(u: str) -> Optional[RawListing]:
        async with sem:
            return await _fetch_one(u)

    results = await asyncio.gather(*[_bound(u) for u in urls], return_exceptions=False)
    return [r for r in results if r is not None]


def gather_web(
    config: QueryConfig,
    extra_urls: Optional[List[str]] = None,
    include_github_sources: bool = True,
) -> List[RawListing]:
    """Run the deep-web/GitHub path and return RawListings."""

    urls: List[str] = []
    if include_github_sources:
        urls.extend(DEFAULT_GITHUB_SOURCES)
    if extra_urls:
        urls.extend(extra_urls)
    urls.extend(discover_urls(config))
    if config.extra_sites:
        urls.extend(config.extra_sites)

    # de-dupe preserving order
    seen, deduped = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)

    if not deduped:
        return []

    log.info("Path B fetching %d URLs...", len(deduped))
    try:
        return asyncio.run(_fetch_many(deduped))
    except RuntimeError:
        # Already inside a running loop (e.g. Jupyter); fall back to a new loop.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_fetch_many(deduped))
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Public combined entry point
# ---------------------------------------------------------------------------

def gather_all(
    config: QueryConfig,
    sites: Optional[List[str]] = None,
    skip_deep_web: bool = False,
    extra_urls: Optional[List[str]] = None,
) -> List[RawListing]:
    """Run Path A then Path B and return the combined listing pool."""

    a = gather_jobspy(config, sites=sites)
    b = [] if skip_deep_web else gather_web(config, extra_urls=extra_urls)
    log.info("Gathered %d JobSpy rows + %d web chunks.", len(a), len(b))
    return a + b
