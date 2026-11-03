"""Graph nodes: planner, researcher, critic, writer.

Node contract: each node returns ONLY the state keys it changes. Keys with
an `operator.add` reducer (`plan`, `notes`, `usage`) receive just the NEW
items to append — returning `{**state, ...}` would duplicate accumulated values.
"""
import json
import logging

from langchain_core.messages import HumanMessage

from . import prompts
from .config import get_settings
from .costs import usage_event
from .schemas import Critique, ResearchPlan, SearchAnalysis
from .state import ResearchState
from .tools import get_llm, get_search_tool, invoke_structured, scrape_url

logger = logging.getLogger(__name__)


def _response_text(response) -> str:
    """Extract plain text from an LLM response, whatever shape content takes."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("text"):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def _model_name() -> str:
    return get_settings().model_name


def _domain(state: ResearchState) -> str:
    return state.get("domain") or "general"


# ==================== NODE 1: THE PLANNER ====================
def planner_node(state: ResearchState) -> dict:
    """Breaks the query into sub-questions; on re-entry, plans only the gaps."""
    domain = _domain(state)
    logger.info(
        "[PLANNER] Creating research plan (iteration %d, domain %s)",
        state.get("iteration", 0) + 1,
        domain,
    )

    query = state["query"]
    critique = state.get("critique", "")
    existing_plan = state.get("plan", [])

    if critique and existing_plan:
        prompt = prompts.replan_prompt(query, critique, existing_plan, domain)
    else:
        prompt = prompts.initial_plan_prompt(query, domain)

    model = _model_name()
    try:
        plan, event = invoke_structured(
            get_llm(),
            ResearchPlan,
            [HumanMessage(content=prompt)],
            "planner",
            model,
        )
        seen = set(existing_plan)
        new_questions = [question for question in plan.sub_questions if question not in seen]
        usage = [event]
    except Exception as e:  # noqa: BLE001 — model and parse failures must not crash the run
        logger.error("[PLANNER] Failed to generate plan: %s", e)
        new_questions = []
        usage = []

    logger.info("[PLANNER] %d new sub-questions", len(new_questions))
    for i, question in enumerate(new_questions, 1):
        logger.info("  %d. %s", i, question)

    return {
        "plan": new_questions,
        "pending_questions": new_questions,
        "iteration": state.get("iteration", 0) + 1,
        "usage": usage,
    }


# ==================== NODE 2: THE RESEARCHER ====================
def researcher_node(state: ResearchState) -> dict:
    """Researches ONLY the pending questions (not previously answered ones)."""
    domain = _domain(state)
    pending = state.get("pending_questions", [])
    logger.info(
        "[RESEARCHER] Gathering information for %d question(s) (domain %s)",
        len(pending),
        domain,
    )

    model = _model_name()
    llm = get_llm()
    search_tool = get_search_tool(domain)
    gathered_notes = []
    usage = []

    for idx, sub_question in enumerate(pending, 1):
        logger.info("[RESEARCHER] (%d/%d) %s", idx, len(pending), sub_question)

        try:
            search_results = search_tool.invoke(sub_question)
            analysis, event = invoke_structured(
                llm,
                SearchAnalysis,
                [
                    HumanMessage(
                        content=prompts.search_analysis_prompt(
                            sub_question,
                            json.dumps(search_results, indent=2, default=str),
                            domain,
                        )
                    )
                ],
                "researcher",
                model,
            )
            usage.append(event)

            note = {
                "question": sub_question,
                "search_results": search_results,
                "summary": analysis.summary,
                "sufficient": analysis.sufficient,
            }

            if analysis.needs_scraping and analysis.url_to_scrape:
                logger.info("[RESEARCHER] Scraping: %s", analysis.url_to_scrape)
                note["scraped_content"] = scrape_url(analysis.url_to_scrape)

            gathered_notes.append(note)
            logger.info("[RESEARCHER] Summary: %.100s...", analysis.summary)

        except Exception as e:  # noqa: BLE001 — search or model failures stay on the note
            logger.error("[RESEARCHER] Error on %r: %s", sub_question, e)
            gathered_notes.append(
                {"question": sub_question, "error": str(e), "sufficient": False}
            )

    return {
        "notes": gathered_notes,
        "pending_questions": [],
        "usage": usage,
    }


# ==================== NODE 3: THE CRITIC ====================
def critic_node(state: ResearchState) -> dict:
    """Checks completeness; loops back with feedback or approves."""
    domain = _domain(state)
    logger.info("[CRITIC] Evaluating research quality (domain %s)", domain)

    query = state["query"]
    notes = state["notes"]
    iteration = state["iteration"]
    max_iterations = state.get("max_iterations") or get_settings().max_iterations

    notes_summary = "\n\n".join(
        f"Question: {note['question']}\n"
        f"Summary: {note.get('summary', 'No summary')}\n"
        f"Sufficient: {note.get('sufficient', False)}"
        for note in notes
    )

    model = _model_name()
    try:
        critique, event = invoke_structured(
            get_llm(),
            Critique,
            [
                HumanMessage(
                    content=prompts.critique_prompt(query, iteration, notes_summary, domain)
                )
            ],
            "critic",
            model,
        )
        feedback = critique.feedback
        is_complete = critique.is_complete
        usage = [event]
    except Exception as e:  # noqa: BLE001 — a broken critique ends the loop instead of crashing
        logger.error("[CRITIC] Critique failed (%s); treating research as complete", e)
        feedback = "Critique unavailable — proceeding with gathered research."
        is_complete = True
        usage = []

    is_complete = is_complete or iteration >= max_iterations

    if is_complete:
        logger.info("[CRITIC] Research quality approved")
    else:
        logger.info("[CRITIC] Incomplete. Feedback: %s", feedback)

    return {
        "critique": feedback,
        "should_continue": not is_complete,
        "usage": usage,
    }


# ==================== NODE 4: THE WRITER ====================
def writer_node(state: ResearchState) -> dict:
    """Synthesizes all notes into a cited Markdown report."""
    domain = _domain(state)
    logger.info("[WRITER] Composing final report (domain %s)", domain)

    query = state["query"]
    notes = state["notes"]
    model = _model_name()

    research_content = ""
    for i, note in enumerate(notes, 1):
        research_content += f"\n## Research Point {i}\n"
        research_content += f"**Question:** {note['question']}\n"
        research_content += f"**Summary:** {note.get('summary', 'N/A')}\n"
        if note.get("search_results"):
            research_content += (
                f"**Sources:** {json.dumps(note['search_results'], indent=2, default=str)}\n"
            )
        if note.get("scraped_content"):
            research_content += f"**Additional Details:** {note['scraped_content'][:500]}...\n"

    try:
        response = get_llm().invoke(
            [
                HumanMessage(
                    content=prompts.writer_prompt(query, research_content, domain)
                )
            ]
        )
        final_report = _response_text(response)
        usage = [usage_event("writer", response, model)]
    except Exception as e:  # noqa: BLE001 — writer failure still returns the gathered summaries
        logger.error("[WRITER] Report generation failed: %s", e)
        final_report = (
            f"# Research Report (generation failed)\n\n"
            f"**Query:** {query}\n\n"
            f"Report synthesis failed ({e}). Raw research summaries:\n"
            + "\n".join(f"- **{n['question']}**: {n.get('summary', 'N/A')}" for n in notes)
        )
        usage = []

    logger.info("[WRITER] Report completed")

    return {
        "final_report": final_report,
        "should_continue": False,
        "usage": usage,
    }
