"""Graph routing and the note-duplication regression."""
from finsight.graph import build_graph, iter_run, resolve_domain, should_continue_research
from finsight.schemas import Critique, ResearchPlan, SearchAnalysis
from tests.fakes import FakeLLM, FakeSearch

Q1 = "What is the escrow rule for payment aggregators?"
Q2 = "Which entities does the circular bind?"
Q3 = "When does the compliance timeline start?"


def test_routing_follows_the_critic_flag():
    assert should_continue_research({"should_continue": True}) == "planner"
    assert should_continue_research({"should_continue": False}) == "writer"
    assert should_continue_research({}) == "writer"


def test_compiled_graph_has_the_four_nodes():
    names = set(build_graph().get_graph().nodes)
    assert {"planner", "researcher", "critic", "writer"} <= names


def test_iter_run_emits_node_events_and_a_cost_footer(monkeypatch):
    class _Graph:
        def stream(self, state, stream_mode=None):
            yield ("updates", {"planner": {"iteration": 1}})
            merged = dict(state)
            merged["final_report"] = "# Report"
            merged["usage"] = [
                {"node": "planner", "model": "m", "input_tokens": 10, "output_tokens": 2}
            ]
            yield ("values", merged)

    monkeypatch.setattr("finsight.graph.build_graph", lambda: _Graph())
    events = list(iter_run("RBI payment aggregator rules", domain="general", max_iterations=1))

    assert events[0] == {"type": "node", "node": "planner", "iteration": 1}
    assert events[-1]["type"] == "report"
    assert "## Run cost (estimated)" in events[-1]["report"]
    assert events[-1]["usage"][0]["node"] == "planner"


def test_classifier_failure_falls_back_to_general(monkeypatch):
    class _Down:
        def with_structured_output(self, schema, include_raw=True):
            raise RuntimeError("classifier down")

    monkeypatch.setattr("finsight.graph.get_fast_llm", lambda: _Down())
    domain, usage = resolve_domain("What is UPI?", None)
    assert domain == "general"
    assert usage == []


def test_two_loop_iterations_do_not_duplicate_notes_or_research_questions(monkeypatch):
    """Regression: two loop iterations must not re-append notes or re-search old questions.

    Nodes used to return ``{**state, ...}``. With ``operator.add`` on ``notes``,
    that appended the whole history again on every pass, and the researcher
    searched the full plan instead of the new questions.
    """
    search = FakeSearch()
    calls = {"planner": 0, "critic": 0}

    def on_structured(schema, messages):
        if schema is ResearchPlan:
            calls["planner"] += 1
            if calls["planner"] == 1:
                return ResearchPlan(sub_questions=[Q1, Q2])
            return ResearchPlan(sub_questions=[Q1, Q3])
        if schema is SearchAnalysis:
            return SearchAnalysis(sufficient=True, summary="Covered.", needs_scraping=False)
        if schema is Critique:
            calls["critic"] += 1
            if calls["critic"] == 1:
                return Critique(
                    is_complete=False,
                    feedback="Missing the compliance timeline",
                    missing_aspects=["timeline"],
                )
            return Critique(is_complete=True, feedback="Research is complete", missing_aspects=[])
        raise AssertionError(f"unexpected schema {schema}")

    monkeypatch.setattr("finsight.nodes.get_llm", lambda model_name="": FakeLLM(on_structured))
    monkeypatch.setattr("finsight.nodes.get_search_tool", lambda domain="general": search)

    final = build_graph().invoke(
        {
            "query": "How do RBI payment aggregator guidelines affect UPI-first companies?",
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
    )

    assert search.calls == [Q1, Q2, Q3]
    assert [note["question"] for note in final["notes"]] == [Q1, Q2, Q3]
    assert final["plan"] == [Q1, Q2, Q3]
    assert final["pending_questions"] == []
    assert final["iteration"] == 2
    assert final["should_continue"] is False
    # planner x2, researcher calls x3, critic x2, writer x1
    assert len(final["usage"]) == 8
    assert calls == {"planner": 2, "critic": 2}
