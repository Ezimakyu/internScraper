# Sample queries

A few prompts to try once you've installed dependencies and exported
`OPENAI_API_KEY`:

```bash
# Big-board only (fast, no Playwright needed)
python -m intern_scraper --skip-deep-web "software engineering internship summer 2026"

# Aerospace / defense focus, pulls from GitHub aggregator READMEs too
python -m intern_scraper "aerospace software internships summer 2026"

# ML/AI internships, custom output path + CSV dump
python -m intern_scraper \
  -o ./output/ml-internships.html \
  --csv ./output/ml-internships.csv \
  "machine learning intern summer 2026"

# Add a curated company careers page to the deep-web crawl
python -m intern_scraper \
  --extra-url https://boards.greenhouse.io/anthropic \
  "applied ai research intern 2026"
```
