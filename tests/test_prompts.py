"""Domain-specific prompt sections."""
from finsight.prompts import initial_plan_prompt, writer_prompt


def test_regulatory_writer_adds_implication_and_timeline_sections():
    prompt = writer_prompt("query", "notes", "regulatory")
    assert "Regulatory Implications" in prompt
    assert "Compliance Timeline" in prompt
    assert "circular" in prompt.lower()


def test_regulatory_planner_covers_dates_entities_and_timelines():
    prompt = initial_plan_prompt("RBI payment aggregator guidelines", "regulatory")
    lowered = prompt.lower()
    assert "effective date" in lowered
    assert "payment aggregator" in lowered
    assert "compliance timeline" in lowered


def test_market_writer_asks_for_dated_shares():
    prompt = writer_prompt("query", "notes", "market")
    lowered = prompt.lower()
    assert "market share" in lowered or "shares" in lowered
    assert "date" in lowered


def test_general_writer_keeps_the_original_sections():
    prompt = writer_prompt("query", "notes", "general")
    assert "Executive Summary" in prompt
    assert "Regulatory Implications" not in prompt
