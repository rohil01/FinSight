"""POST /research streams node events and a final report."""
from fastapi.testclient import TestClient

from finsight.api import app


class _FakeGraph:
    async def astream(self, state, stream_mode=None):
        yield ("updates", {"planner": {"iteration": 1}})
        merged = dict(state)
        merged["final_report"] = "# Done\n\nSee https://rbi.org.in"
        merged["usage"] = [
            {"node": "writer", "model": "fake", "input_tokens": 100, "output_tokens": 50}
        ]
        merged["iteration"] = 1
        yield ("values", merged)


def test_research_endpoint_streams_node_and_report_events(monkeypatch):
    monkeypatch.setattr("finsight.graph.build_graph", lambda: _FakeGraph())
    client = TestClient(app)

    with client.stream(
        "POST",
        "/research",
        json={"query": "RBI payment aggregator guidelines", "domain": "general"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: node" in body
    assert '"node": "planner"' in body
    assert "event: report" in body
    assert "Run cost (estimated)" in body
    assert "https://rbi.org.in" in body
