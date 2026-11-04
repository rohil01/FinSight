"""Graph construction and the public research entry points."""
import logging
from collections.abc import Iterator

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from .config import get_settings
from .costs import apply_cost_footer, summarize_usage
from .domains import get_profile
from .nodes import critic_node, planner_node, researcher_node, writer_node
from .prompts import classify_domain_prompt
from .schemas import DomainChoice
from .state import ResearchState
from .tools import get_fast_llm, invoke_structured

logger = logging.getLogger(__name__)

_STREAM_MODES = {"updates", "values", "custom", "debug", "messages"}


def should_continue_research(state: ResearchState) -> str:
    """Route from critic: loop back to planner or proceed to writer."""
    if state.get("should_continue", False):
        logger.info("Looping back to planner for additional research...")
        return "planner"
    logger.info("Proceeding to writer...")
    return "writer"


def build_graph():
    """Constructs and compiles the LangGraph state machine."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_conditional_edges(
        "critic",
        should_continue_research,
        {"planner": "planner", "writer": "writer"},
    )
    workflow.add_edge("writer", END)

    return workflow.compile()


def resolve_domain(query: str, domain: str | None) -> tuple:
    """Use an explicit domain, otherwise classify with the fast model.

    Returns (domain_name, usage_events). Classification failure falls back
    to general so a classifier outage still produces a report.
    """
    if domain:
        get_profile(domain)
        logger.info("Domain set explicitly: %s", domain)
        return domain, []

    logger.info("Classifying query domain")
    settings = get_settings()
    try:
        choice, event = invoke_structured(
            get_fast_llm(),
            DomainChoice,
            [HumanMessage(content=classify_domain_prompt(query))],
            "classifier",
            settings.fast_model_name,
        )
        logger.info("Classified domain: %s", choice.domain)
        return choice.domain, [event]
    except Exception as e:  # noqa: BLE001 — classifier outage falls back to general
        logger.error("Domain classification failed (%s); using general", e)
        return "general", []


def prepare_research(
    query: str,
    domain: str | None = None,
    max_iterations: int | None = None,
):
    """Validate keys, resolve the domain, and build the initial state."""
    get_settings().validate_keys()
    settings = get_settings()
    resolved, seed_usage = resolve_domain(query, domain)
    graph = build_graph()
    initial: ResearchState = {
        "query": query,
        "domain": resolved,
        "plan": [],
        "pending_questions": [],
        "notes": [],
        "usage": seed_usage,
        "iteration": 0,
        "max_iterations": max_iterations or settings.max_iterations,
        "critique": "",
        "final_report": "",
        "should_continue": True,
    }
    return graph, initial


def unpack_stream_item(item):
    """Normalize a LangGraph stream item to (mode, payload).

    A single stream mode yields the payload alone. A list of modes yields
    (mode, payload) tuples.
    """
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and item[0] in _STREAM_MODES
    ):
        return item
    return "updates", item


def _finalize(final_state: dict) -> dict:
    usage: list[dict] = list(final_state.get("usage") or [])
    report = apply_cost_footer(final_state.get("final_report") or "", usage)
    summary = summarize_usage(usage)
    logger.info(
        "Run finished: %s input tokens, %s output tokens, estimated $%.4f",
        summary["input_tokens"],
        summary["output_tokens"],
        summary["estimated_usd"],
    )
    return {"report": report, "usage": usage}


def iter_run(
    query: str,
    domain: str | None = None,
    max_iterations: int | None = None,
) -> Iterator[dict]:
    """Stream node updates, then a final report event that includes the cost footer."""
    graph, initial = prepare_research(query, domain, max_iterations)
    final_state = dict(initial)
    for item in graph.stream(initial, stream_mode=["updates", "values"]):
        mode, chunk = unpack_stream_item(item)
        if mode == "updates" and isinstance(chunk, dict) and chunk:
            name, update = next(iter(chunk.items()))
            update = update or {}
            iteration = update.get("iteration", final_state.get("iteration"))
            yield {"type": "node", "node": name, "iteration": iteration}
        elif mode == "values" and isinstance(chunk, dict):
            final_state = chunk
    yield {"type": "report", **_finalize(final_state)}


async def aiter_run(
    query: str,
    domain: str | None = None,
    max_iterations: int | None = None,
):
    """Async twin of iter_run, backed by graph.astream."""
    graph, initial = prepare_research(query, domain, max_iterations)
    final_state = dict(initial)
    async for item in graph.astream(initial, stream_mode=["updates", "values"]):
        mode, chunk = unpack_stream_item(item)
        if mode == "updates" and isinstance(chunk, dict) and chunk:
            name, update = next(iter(chunk.items()))
            update = update or {}
            iteration = update.get("iteration", final_state.get("iteration"))
            yield {"type": "node", "node": name, "iteration": iteration}
        elif mode == "values" and isinstance(chunk, dict):
            final_state = chunk
    yield {"type": "report", **_finalize(final_state)}


def run_research(
    query: str,
    domain: str | None = None,
    max_iterations: int | None = None,
) -> str:
    """Execute the research agent for a query and return the final report."""
    report = ""
    for event in iter_run(query, domain, max_iterations):
        if event["type"] == "report":
            report = event["report"]
    return report
