"""Domain profiles and Tavily include_domains."""
import pytest

from finsight.domains import DOMAINS, get_profile
from finsight.tools import get_search_tool


def test_profiles_match_the_curated_source_lists():
    regulatory = get_profile("regulatory")
    assert regulatory.include_domains == (
        "rbi.org.in",
        "npci.org.in",
        "sebi.gov.in",
        "pib.gov.in",
    )
    market = get_profile("market")
    assert market.include_domains == (
        "livemint.com",
        "economictimes.indiatimes.com",
        "moneycontrol.com",
        "medianama.com",
        "inc42.com",
        "entrackr.com",
    )
    assert get_profile("general").include_domains is None
    assert DOMAINS == ("regulatory", "market", "general")


def test_unknown_domain_is_rejected():
    with pytest.raises(ValueError):
        get_profile("equities")


def test_search_tool_filters_regulatory_domains(monkeypatch):
    captured = {}

    class _FakeTavily:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("finsight.tools.TavilySearch", _FakeTavily)
    get_search_tool.cache_clear()
    get_search_tool("regulatory")

    assert "rbi.org.in" in captured["include_domains"]
    assert captured["max_results"] >= 1


def test_general_search_is_not_domain_filtered(monkeypatch):
    captured = {}

    class _FakeTavily:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("finsight.tools.TavilySearch", _FakeTavily)
    get_search_tool.cache_clear()
    get_search_tool("general")

    assert "include_domains" not in captured
