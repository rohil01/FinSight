"""Prompt builders for the research agent.

Each builder takes the active domain so regulatory and market runs ask for
the evidence those reports need. The general profile keeps the original
section list.
"""
from .domains import get_profile


def _guidance_block(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return f"\n\n{text}"


def classify_domain_prompt(query: str) -> str:
    """Cheap classifier prompt used when the caller does not pass --domain."""
    return f"""Classify this research query into exactly one domain.

- regulatory: law, regulation, RBI/SEBI/NPCI circulars, compliance, licensing
- market: market share, volumes, company news, competition, funding
- general: anything else

Query: {query}"""


def initial_plan_prompt(query: str, domain: str) -> str:
    profile = get_profile(domain)
    return (
        "You are a research planner. Break down this query into 3-5 specific, "
        "researchable sub-questions. Make them concrete and answerable.\n\n"
        f"Query: {query}"
        f"{_guidance_block(profile.planner_guidance)}"
    )


def replan_prompt(query: str, critique: str, existing_questions: list, domain: str) -> str:
    profile = get_profile(domain)
    listed = "\n".join(f"- {question}" for question in existing_questions)
    return (
        "You are a research planner. The previous research was incomplete.\n\n"
        f"Original Query: {query}\n"
        f"Critique: {critique}\n\n"
        "Questions already researched (do NOT repeat these):\n"
        f"{listed}\n\n"
        "Create 2-3 additional specific sub-questions to address the gaps."
        f"{_guidance_block(profile.planner_guidance)}"
    )


def search_analysis_prompt(sub_question: str, search_results: str, domain: str) -> str:
    profile = get_profile(domain)
    source_hint = ""
    if profile.include_domains:
        names = ", ".join(profile.include_domains)
        source_hint = f"\n\nPrefer evidence from these sources when the results include them: {names}."
    return (
        f'Analyze these search results for the question: "{sub_question}"\n\n'
        f"Search Results:\n{search_results}\n\n"
        "Assess whether the results sufficiently answer the question, summarize the findings, "
        "and decide whether one of the result URLs should be scraped for more detail."
        f"{source_hint}"
    )


def critique_prompt(query: str, iteration: int, notes_summary: str, domain: str) -> str:
    profile = get_profile(domain)
    return (
        "You are a research quality critic. Evaluate if the gathered research "
        "adequately answers the original query.\n\n"
        f"Original Query: {query}\n"
        f"Domain: {domain}\n"
        f"Iteration: {iteration}\n\n"
        f"Research Gathered:\n{notes_summary}\n\n"
        "Be strict but fair. Identify specific gaps if the research is incomplete."
        f"{_guidance_block(profile.critic_guidance)}"
    )


def writer_prompt(query: str, research_content: str, domain: str) -> str:
    profile = get_profile(domain)
    extra = ""
    if profile.writer_guidance:
        extra = f"\n{profile.writer_guidance}\n"
    return (
        "You are a research writer. Create a comprehensive, well-structured Markdown report.\n\n"
        f"Original Query: {query}\n"
        f"Domain: {domain}\n\n"
        f"Research Gathered:\n{research_content}\n\n"
        "Write a professional report with:\n"
        "1. Executive Summary\n"
        "2. Key Findings (organized by theme)\n"
        "3. Detailed Analysis\n"
        "4. Citations (include URLs from sources)\n"
        "5. Conclusion\n"
        f"{extra}\n"
        "Use proper Markdown formatting. Cite only sources that appear in the research notes."
    )
