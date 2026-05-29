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
        is_technical_role=True,
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
        _make(company="Marketing Co", is_technical_role=False,
              role_title="Marketing Intern"),
    ]
    out = filter_listings(listings, cfg)
    companies = {x.company for x in out}
    assert "Acme" in companies
    assert "Past Co" not in companies
    assert "UK Co" not in companies
    assert "Grad Co" not in companies
    assert "Marketing Co" not in companies


def test_filter_respects_min_post_date():
    cfg = QueryConfig(raw_query="swe internship", min_post_date="2026-05-15")
    listings = [
        _make(company="NewCo", post_date="2026-05-20"),
        _make(company="OldCo", post_date="2026-05-01"),
        _make(company="NoDateCo", post_date=None),
    ]
    out = filter_listings(listings, cfg)
    names = {x.company for x in out}
    assert "NewCo" in names
    assert "OldCo" not in names
    # Listings without a parseable date are kept (we err on the side of
    # surfacing more results, per the spec).
    assert "NoDateCo" in names


def test_filter_keeps_non_technical_when_allowed():
    cfg = QueryConfig(raw_query="any internship", technical_only=False)
    listings = [
        _make(company="SWE Co"),
        _make(company="Marketing Co", is_technical_role=False,
              role_title="Marketing Intern"),
    ]
    out = filter_listings(listings, cfg)
    names = {x.company for x in out}
    assert {"SWE Co", "Marketing Co"}.issubset(names)


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
