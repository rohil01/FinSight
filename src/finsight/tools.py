"""External tools: LLM factory, Tavily search, and web scraping."""
import logging
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from .config import get_settings
from .costs import usage_event
from .domains import get_profile

logger = logging.getLogger(__name__)


@lru_cache
def get_llm(model_name: str = "") -> ChatGoogleGenerativeAI:
    """Return a cached chat model. An empty name uses Settings.model_name."""
    settings = get_settings()
    name = model_name or settings.model_name
    return ChatGoogleGenerativeAI(model=name)


def get_fast_llm() -> ChatGoogleGenerativeAI:
    """Cheaper model for domain classification and eval judging."""
    return get_llm(get_settings().fast_model_name)


@lru_cache
def get_search_tool(domain: str = "general") -> TavilySearch:
    """Tavily client for one domain. Regulatory and market runs are domain-filtered.

    The cache key includes the domain so a regulatory client is never reused
    for an open-web search.
    """
    settings = get_settings()
    profile = get_profile(domain)
    kwargs = {"max_results": settings.max_search_results}
    if profile.include_domains:
        kwargs["include_domains"] = list(profile.include_domains)
    logger.info(
        "Search client for domain %s (include_domains=%s)",
        domain,
        kwargs.get("include_domains"),
    )
    return TavilySearch(**kwargs)


def invoke_structured(llm, schema, messages, node: str, model: str):
    """Call a structured-output model and keep the raw message for token usage.

    Returns (parsed_model, usage_event). Raises if parsing fails so callers
    can apply their own fallback.
    """
    result = llm.with_structured_output(schema, include_raw=True).invoke(messages)
    if isinstance(result, dict) and "parsed" in result:
        parsed = result.get("parsed")
        error = result.get("parsing_error")
        if parsed is None:
            raise error or RuntimeError(f"{node} structured output was empty")
        return parsed, usage_event(node, result.get("raw"), model)
    return result, usage_event(node, result, model)


def scrape_url(url: str) -> str | None:
    """Scrape a page and return its text (truncated), or None on failure."""
    from langchain_community.document_loaders import WebBaseLoader

    settings = get_settings()
    try:
        docs = WebBaseLoader(url).load()
        return docs[0].page_content[: settings.scrape_char_limit]
    except Exception as e:  # noqa: BLE001 — a bad page should not abort the question
        logger.warning("Scraping failed for %s: %s", url, e)
        return None
