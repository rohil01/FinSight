"""Set dummy API keys before the finsight package is imported."""
import os

os.environ.setdefault("GOOGLE_API_KEY", "test-google")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily")


import pytest


@pytest.fixture(autouse=True)
def _clear_finsight_caches():
    from finsight.config import get_settings
    from finsight.tools import get_llm, get_search_tool

    get_settings.cache_clear()
    get_llm.cache_clear()
    get_search_tool.cache_clear()
    yield
    get_settings.cache_clear()
    get_llm.cache_clear()
    get_search_tool.cache_clear()
