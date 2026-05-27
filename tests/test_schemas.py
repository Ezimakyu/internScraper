from intern_scraper.schemas import ParsedListing, QueryConfig, RawListing


def test_query_config_defaults():
    cfg = QueryConfig(raw_query="software internship")
    assert cfg.location == "USA"
    assert cfg.target_level == "undergraduate"
    assert cfg.results_per_site >= 1
    assert cfg.role_keywords == ["software engineer intern"]


def test_raw_listing_minimum():
    r = RawListing(text="hello")
    assert r.source == "web"
    assert r.text == "hello"


def test_parsed_listing_required_fields():
    p = ParsedListing(
        company="Acme",
        role_title="SWE Intern",
        application_link="https://example.com/apply",
        is_undergraduate=True,
        is_usa_based=True,
        is_past_date=False,
    )
    assert p.company == "Acme"
    assert 0.0 <= p.confidence <= 1.0
