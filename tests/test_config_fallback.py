from intern_scraper.config import _fallback_parse


def test_fallback_extracts_season_year_and_industry():
    cfg = _fallback_parse("aerospace software internships summer 2026")
    assert cfg.season == "Summer"
    assert cfg.year == 2026
    assert "aerospace" in cfg.industry_keywords
    assert any("software" in rk for rk in cfg.role_keywords)


def test_fallback_handles_minimal_query():
    cfg = _fallback_parse("ml intern")
    assert cfg.season is None
    assert cfg.year is None
    # Should still produce at least one role keyword
    assert cfg.role_keywords
