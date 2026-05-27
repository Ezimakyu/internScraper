from intern_scraper.export import to_dataframe
from intern_scraper.parser import filter_listings
from intern_scraper.schemas import ParsedListing, QueryConfig


def _make(**kw):
    base = dict(
        company="Acme",
        role_title="Software Engineer Intern",
        location="San Francisco, CA",
        post_date="2026-04-01",
        application_link="https://example.com/apply",
        is_undergraduate=True,
        is_usa_based=True,
        is_past_date=False,
        confidence=0.9,
    )
    base.update(kw)
    return ParsedListing(**base)


def test_filter_drops_past_and_non_us():
    cfg = QueryConfig(raw_query="software internship")
    listings = [
        _make(),
        _make(company="Past Co", is_past_date=True),
        _make(company="UK Co", is_usa_based=False, location="London, UK"),
        _make(company="Grad Co", is_undergraduate=False),
    ]
    out = filter_listings(listings, cfg)
    companies = {x.company for x in out}
    assert "Acme" in companies
    assert "Past Co" not in companies
    assert "UK Co" not in companies
    assert "Grad Co" not in companies


def test_to_dataframe_dedupes_and_sorts():
    listings = [
        _make(company="A", post_date="2026-03-01"),
        _make(company="A", post_date="2026-03-01"),  # duplicate
        _make(company="B", post_date="2026-05-15"),
    ]
    df = to_dataframe(listings)
    assert len(df) == 2  # de-duplicated
    # Newest first
    assert df.iloc[0]["company"] == "B"
