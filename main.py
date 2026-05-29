"""Convenience top-level entrypoint.

Equivalent to ``python -m intern_scraper ...``.
"""

from intern_scraper.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
