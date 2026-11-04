"""Schema validation for structured LLM outputs."""
import pytest
from pydantic import ValidationError

from finsight.schemas import Critique, DomainChoice, ResearchPlan, SearchAnalysis


def test_research_plan_requires_a_question():
    plan = ResearchPlan(sub_questions=["What is the escrow rule?"])
    assert plan.sub_questions == ["What is the escrow rule?"]
    with pytest.raises(ValidationError):
        ResearchPlan(sub_questions=[])


def test_search_analysis_defaults():
    analysis = SearchAnalysis(sufficient=True, summary="Escrow is mandatory.")
    assert analysis.needs_scraping is False
    assert analysis.url_to_scrape is None


def test_critique_defaults_missing_aspects():
    critique = Critique(is_complete=True, feedback="Research is complete")
    assert critique.missing_aspects == []


def test_domain_choice_rejects_unknown_label():
    choice = DomainChoice(domain="regulatory")
    assert choice.domain == "regulatory"
    with pytest.raises(ValidationError):
        DomainChoice(domain="equities")
