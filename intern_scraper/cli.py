"""Command line entry point.

Usage::

    python -m intern_scraper "aerospace software internships summer 2026"
    python -m intern_scraper --output ./out/results.html --skip-deep-web "ml intern"

The flow mirrors the four modules:

1. ``config.parse_query``        - natural language -> QueryConfig
2. ``scraper.gather_all``        - JobSpy + Google + Playwright -> RawListings
3. ``parser.parse_listings``     - LLM -> ParsedListings
4. ``parser.filter_listings``    - USA / undergrad / not-expired
5. ``export.render_html``        - DataFrame -> HTML (+ optional CSV)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional

from .config import parse_query
from .export import render_html, to_dataframe, write_csv
from .parser import filter_listings, parse_listings
from .scraper import gather_all


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="intern_scraper",
        description="LLM-powered internship scraper.",
    )
    p.add_argument("query", help="Natural-language search query, in quotes.")
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Path for the generated HTML report. "
             "Defaults to ./output/results-<timestamp>.html.",
    )
    p.add_argument(
        "--csv", default=None,
        help="Optional path to also dump a CSV of the results.",
    )
    p.add_argument(
        "--sites", nargs="*", default=None,
        help="JobSpy site list (default: linkedin indeed glassdoor zip_recruiter).",
    )
    p.add_argument(
        "--skip-deep-web", action="store_true",
        help="Skip Google + Playwright path; use only JobSpy.",
    )
    p.add_argument(
        "--skip-jobspy", action="store_true",
        help="Skip the big-board path; use only the deep web / GitHub path.",
    )
    p.add_argument(
        "--extra-url", action="append", default=[],
        help="Extra URL for the deep web path. Repeatable.",
    )
    p.add_argument(
        "--allow-past", action="store_true",
        help="Keep postings flagged as past-dated (for archival / debugging).",
    )
    p.add_argument(
        "--min-date", "--oldest-date", dest="min_date", default=None,
        help="Drop postings older than this ISO date (YYYY-MM-DD). "
             "E.g. --min-date 2026-05-15.",
    )
    p.add_argument(
        "--allow-non-technical", action="store_true",
        help="Keep marketing / sales / BD / recruiting / etc. roles that the "
             "LLM flags as non-technical (off by default).",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Verbose logging.",
    )
    return p


def _default_output_path() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join("output", f"results-{ts}.html")


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log = logging.getLogger("intern_scraper")

    # 1. Parse natural language query.
    config = parse_query(args.query)
    if args.min_date:
        config.min_post_date = args.min_date
    if args.allow_non_technical:
        config.technical_only = False
    log.info("QueryConfig: %s", config.model_dump())

    # 2. Gather.
    raws = []
    if not args.skip_jobspy:
        from .scraper import gather_jobspy
        raws.extend(gather_jobspy(config, sites=args.sites))
    if not args.skip_deep_web:
        from .scraper import gather_web
        raws.extend(gather_web(config, extra_urls=args.extra_url))
    if not raws:
        log.error("No raw listings gathered. Try a broader query or --debug.")
        return 2

    # 3. LLM parse.
    parsed = parse_listings(raws, config)

    # 4. Filter.
    final = filter_listings(parsed, config, drop_past=not args.allow_past)

    # 5. Export.
    df = to_dataframe(final)
    out_html = args.output or _default_output_path()
    render_html(df, query=args.query, output_path=out_html)
    if args.csv:
        write_csv(df, args.csv)

    print(f"\nWrote {len(df)} listings to {out_html}")
    if args.csv:
        print(f"CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
