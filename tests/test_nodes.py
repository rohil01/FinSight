"""Node contracts: partial updates, queue clearing, and fail-open fallbacks."""
from finsight.nodes import critic_node, planner_node, researcher_node, writer_node
from finsight.schemas import Critique, ResearchPlan, SearchAnalysis
from tests.fakes import FakeLLM, FakeSearch


def _state(**overrides):
    state = {
        "query": "What are RBI's payment aggregator guidelines?",
        "domain": "regulatory",
        "plan": [],
        "pending_questions": [],
        "notes": [],
        "usage": [],
        "iteration": 0,
        "max_iterations": 3,
        "critique": "",
        "final_report": "",
        "should_continue": True,
    }
    state.update(overrides)
    return state


def test_planner_filters_questions_already_in_the_plan(monkeypatch):
    def on_structured(schema, messages):
        assert schema is ResearchPlan
        return ResearchPlan(sub_questions=["What is the escrow rule?", "When does it take effect?"])

    monkeypatch.setattr("finsight.nodes.get_llm", lambda model_name="": FakeLLM(on_structured))
    result = planner_node(
        _state(
            plan=["What is the escrow rule?"],
            critique="Missing the effective date",
            iteration=1,
        )
    )

    assert result["plan"] == ["When does it take effect?"]
    assert result["pending_questions"] == ["When does it take effect?"]
    assert result["iteration"] == 2
    assert [event["node"] for event in result["usage"]] == ["planner"]
    assert "notes" not in result


def test_researcher_searches_only_pending_questions_and_clears_the_queue(monkeypatch):
    search = FakeSearch()

    def on_structured(schema, messages):
        assert schema is SearchAnalysis
        return SearchAnalysis(sufficient=True, summary="Escrow is required.", needs_scraping=False)

    monkeypatch.setattr("finsight.nodes.get_llm", lambda model_name="": FakeLLM(on_structured))
    monkeypatch.setattr("finsight.nodes.get_search_tool", lambda domain="general": search)

    result = researcher_node(
        _state(
            plan=["already answered", "only this"],
            pending_questions=["only this"],
            notes=[{"question": "already answered", "summary": "done", "sufficient": True}],
        )
    )

    assert search.calls == ["only this"]
    assert [note["question"] for note in result["notes"]] == ["only this"]
    assert result["pending_questions"] == []
    assert len(result["usage"]) == 1


def test_critic_fail_open_completes_instead_of_crashing(monkeypatch):
    def on_structured(schema, messages):
        raise RuntimeError("parse failed")

    monkeypatch.setattr("finsight.nodes.get_llm", lambda model_name="": FakeLLM(on_structured))
    result = critic_node(
        _state(
            iteration=1,
            notes=[{"question": "q", "summary": "s", "sufficient": True}],
        )
    )

    assert result["should_continue"] is False
    assert "proceeding" in result["critique"].lower()


def test_critic_iteration_cap_overrides_an_incomplete_verdict(monkeypatch):
    def on_structured(schema, messages):
        return Critique(
            is_complete=False,
            feedback="Still missing the timeline",
            missing_aspects=["timeline"],
        )

    monkeypatch.setattr("finsight.nodes.get_llm", lambda model_name="": FakeLLM(on_structured))
    result = critic_node(
        _state(
            iteration=3,
            max_iterations=3,
            notes=[{"question": "q", "summary": "s", "sufficient": False}],
        )
    )

    assert result["should_continue"] is False


def test_critic_loops_when_under_the_cap(monkeypatch):
    def on_structured(schema, messages):
        return Critique(is_complete=False, feedback="Need the timeline", missing_aspects=["timeline"])

    monkeypatch.setattr("finsight.nodes.get_llm", lambda model_name="": FakeLLM(on_structured))
    result = critic_node(
        _state(
            iteration=1,
            max_iterations=3,
            notes=[{"question": "q", "summary": "s", "sufficient": False}],
        )
    )

    assert result["should_continue"] is True


def test_writer_fail_open_returns_a_markdown_fallback(monkeypatch):
    def on_text(messages):
        raise RuntimeError("model down")

    monkeypatch.setattr("finsight.nodes.get_llm", lambda model_name="": FakeLLM(on_text=on_text))
    result = writer_node(
        _state(notes=[{"question": "What is the escrow rule?", "summary": "Escrow is required."}])
    )

    assert result["should_continue"] is False
    assert "generation failed" in result["final_report"].lower()
    assert "What is the escrow rule?" in result["final_report"]
