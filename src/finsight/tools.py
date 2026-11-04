"""External tools: LLM factory, Tavily search, and web scraping."""
import json
import logging
from functools import lru_cache

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_tavily import TavilySearch

from .config import get_settings
from .costs import usage_event
from .domains import get_profile

logger = logging.getLogger(__name__)


@lru_cache
def get_llm(model_name: str = "") -> ChatNVIDIA:
    """Return a cached NVIDIA NIM chat model. An empty name uses Settings.model_name."""
    settings = get_settings()
    name = model_name or settings.model_name
    return ChatNVIDIA(
        model=name,
        api_key=settings.nvidia_api_key,
        max_completion_tokens=8192,
    )


def get_fast_llm() -> ChatNVIDIA:
    """Smaller NVIDIA NIM for domain classification and eval judging."""
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


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        content = "".join(parts)
    return str(content)


def _json_object(text: str) -> str:
    """Return the JSON object inside a model reply, ignoring fences or reasoning."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _invoke_json_schema(llm, schema, messages):
    """Ask the NIM for JSON and keep the chat message, which carries token usage.

    ChatNVIDIA.with_structured_output parses the message away and does not
    support include_raw, so planner, researcher, and critic calls were recorded
    as zero tokens.
    """
    payload = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        },
    }
    return llm.bind(response_format=payload).invoke(messages)


def invoke_structured(llm, schema, messages, node: str, model: str):
    """Call a structured-output model and keep token usage for the cost footer.

    Returns (parsed_model, usage_event). Raises if parsing fails so callers
    can apply their own fallback.
    """
    if hasattr(llm, "bind") and hasattr(schema, "model_json_schema"):
        message = _invoke_json_schema(llm, schema, messages)
        try:
            parsed = schema.model_validate_json(_json_object(_message_text(message)))
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{node} structured output was empty") from exc
        return parsed, usage_event(node, message, model)

    result = llm.with_structured_output(schema).invoke(messages)
    if isinstance(result, dict) and "parsed" in result:
        parsed = result.get("parsed")
        error = result.get("parsing_error")
        if parsed is None:
            raise error or RuntimeError(f"{node} structured output was empty")
        return parsed, usage_event(node, result.get("raw"), model)
    if result is None:
        raise RuntimeError(f"{node} structured output was empty")
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
