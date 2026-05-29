# LLM-Powered Internship Scraper

An automated, LLM-assisted scraping tool that finds, parses, and filters
software / tech internships. It combines fast structured scraping
([JobSpy](https://github.com/Bunsly/JobSpy) for the big boards,
[`googlesearch-python`](https://pypi.org/project/googlesearch-python/) +
[Playwright](https://playwright.dev/python/) for the deep web and GitHub
aggregator READMEs) with an LLM parser (OpenAI `gpt-4o-mini` by default) that
turns messy text into clean rows and applies the project's defaults:
**USA-based**, **undergraduate eligible**, **not expired**.

> Heavy reliance on an LLM for the initial *fetch* would be slow and would
> trigger CAPTCHAs, so the LLM is used only for **parsing** and **filtering**.
> JobSpy + Playwright do the actual page work.

---

## Features

- **Natural-language query.** Type `"aerospace software internships summer 2026"`
  and the tool extracts season, year, role keywords, and industry tags via a
  one-shot LLM call (with a regex fallback when the LLM is unavailable).
- **Two-path gatherer.**
  - *Path A:* JobSpy hits LinkedIn / Indeed / Glassdoor / ZipRecruiter.
  - *Path B:* Google search + Playwright scrape niche career pages, plus a
    curated list of community-maintained GitHub internship READMEs
    (SimplifyJobs, vanshb03, etc.).
- **LLM parsing with structured outputs.** Every text chunk is sent to
  `gpt-4o-mini` with a strict Pydantic schema, so invalid JSON is impossible.
- **Smart filtering.** Drops postings flagged as past-dated, non-USA, or
  graduate-only.
- **Interactive HTML report.** Sortable / filterable table with a "Applied"
  strikethrough checkbox whose state persists in `localStorage`. Also exports
  CSV on request.

---

## Quick setup (conda)

```bash
# 1. Create the env
conda env create -f environment.yml
conda activate internscraper

# 2. Install Playwright browser binaries (one-time)
python -m playwright install chromium

# 3. Provide your OpenAI key
export OPENAI_API_KEY="sk-..."
# or drop it in a .env file in the repo root

# 4. Run it
python -m intern_scraper "aerospace software internships summer 2026"
```

### Pip alternative

If you prefer pip + virtualenv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
export OPENAI_API_KEY="sk-..."
python -m intern_scraper "software engineering internship summer 2026"
```

---

## Usage

```bash
python -m intern_scraper [OPTIONS] "<natural language query>"
```

Common flags:

| Flag | Default | What it does |
|------|---------|--------------|
| `-o, --output PATH` | `output/results-<ts>.html` | Where to write the HTML report. |
| `--csv PATH` | none | Also dump a CSV of the same rows. |
| `--sites …` | `linkedin indeed glassdoor zip_recruiter` | JobSpy boards to hit. |
| `--skip-deep-web` | off | Skip Google + Playwright (fastest run). |
| `--skip-jobspy` | off | Skip the big-board path. |
| `--extra-url URL` | — | Repeatable. Add a custom page to Path B. |
| `--allow-past` | off | Keep postings the LLM flags as expired. |
| `--min-date YYYY-MM-DD` | none | Drop postings older than this date. Alias: `--oldest-date`. |
| `--allow-non-technical` | off | Keep non-engineering roles (marketing, sales, BD, recruiting, etc.). |
| `--debug` | off | Verbose logging. |

Open the resulting HTML in any browser; check the "Applied" box on a row and
the row will be struck through and remembered next time you reload.

More examples in [`examples/sample_queries.md`](examples/sample_queries.md).

---

## How it works

The pipeline mirrors the four-module breakdown from the spec:

```
        natural language query
                  │
                  ▼
┌──────────────────────────────┐   intern_scraper/config.py
│ 1. Config parser (LLM-light) │   gpt-4o-mini → QueryConfig
└──────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────┐   intern_scraper/scraper.py
│ 2. The Gatherer              │
│    Path A: JobSpy            │   linkedin / indeed / glassdoor / zip
│    Path B: Google + Playwright │ greenhouse / lever / ashbyhq + GitHub
└──────────────────────────────┘
                  │   list[RawListing]
                  ▼
┌──────────────────────────────┐   intern_scraper/parser.py
│ 3. LLM filter & parser       │   gpt-4o-mini + Pydantic schema →
│    chunks text into ~6k-char │   ParsedListing rows; flags
│    windows; structured JSON  │   is_undergraduate / is_usa_based /
│                              │   is_past_date / confidence
└──────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────┐   intern_scraper/parser.filter_listings
│    Apply USA / undergrad /   │
│    not-expired filters       │
└──────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────┐   intern_scraper/export.py
│ 4. Output Generator          │
│    pandas dedupe + sort by   │
│    post_date desc → Jinja2   │
│    HTML w/ strikethrough JS  │
└──────────────────────────────┘
                  │
                  ▼
            output/*.html (+ optional CSV)
```

### What "technical role" means

By default the LLM classifies each posting and the pipeline drops any role
flagged `is_technical_role=False`. Kept: SWE, ML/AI, computer vision,
data science / analytics / engineering, robotics software, embedded /
firmware, hardware, security, infra / devops / SRE, quant, applied science.
Dropped: marketing, sales, BD, finance, HR, recruiting, technical writing
only, customer success, non-technical PM / operations.

If you want to see everything anyway, pass `--allow-non-technical`.

### Why LLM for parsing only?

- **Speed.** A single JobSpy call returns dozens of structured rows in seconds.
  Asking an LLM to do the same work via web crawling would cost orders of
  magnitude more time and tokens.
- **CAPTCHAs / rate limits.** LinkedIn and Glassdoor aggressively block naive
  scrapers. JobSpy already implements the workarounds; we don't duplicate
  them.
- **Variety.** Where the LLM *does* shine is on the long tail — niche
  greenhouse boards, ashbyhq pages, and big Markdown READMEs full of
  inconsistent formatting. That's exactly where Path B sends it.

### Configuration knobs

The most useful env vars:

| Env var | Purpose |
|---------|---------|
| `OPENAI_API_KEY` | Required. Your OpenAI key. |
| `INTERNSCRAPER_MODEL` | Override the default `gpt-4o-mini`. |

You can also edit `intern_scraper/scraper.py` to add or remove the curated
GitHub aggregator URLs in `DEFAULT_GITHUB_SOURCES`.

---

## Development

```bash
# Install with pip (in your conda env or venv)
pip install -r requirements.txt
pip install pytest

# Run the lightweight unit tests (no network / no LLM needed)
pytest -q
```

Project layout:

```
intern_scraper/
  __init__.py
  __main__.py        # python -m intern_scraper
  cli.py             # argparse + pipeline
  config.py          # Module 1 - NL query -> QueryConfig
  scraper.py         # Module 2 - JobSpy + Playwright gatherers
  parser.py          # Module 3 - LLM structured parsing + filters
  export.py          # Module 4 - DataFrame + HTML
  llm.py             # OpenAI client wrapper
  schemas.py         # Pydantic schemas (QueryConfig, RawListing, ParsedListing)
  templates/
    results.html     # Jinja2 template + JS for the strikethrough UI
tests/               # pytest unit tests (offline)
examples/            # sample queries
environment.yml      # conda env
requirements.txt     # pip alternative
```

---

## Notes & limitations

- LinkedIn / Glassdoor anti-bot measures change frequently. If JobSpy starts
  returning zero rows, upgrade `python-jobspy` first.
- Google search via `googlesearch-python` is unauthenticated and is sometimes
  rate-limited; that's why the curated GitHub aggregator READMEs are baked in.
- The LLM is asked for self-reported `confidence`. Use the column to sort
  manually if you want to be conservative.
- This tool is for personal job-search use. Respect the terms of service of
  each site you scrape.
