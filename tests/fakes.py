"""Shared fakes for node and graph tests. No network."""
from types import SimpleNamespace


class FakeLLM:
    """LLM double whose structured output and free-text calls are scripted."""

    def __init__(self, on_structured=None, on_text=None):
        self.on_structured = on_structured
        self.on_text = on_text

    def with_structured_output(self, schema, include_raw=True):
        parent = self

        class _Bound:
            def invoke(self, messages, config=None):
                parsed = parent.on_structured(schema, messages)
                raw = SimpleNamespace(
                    usage_metadata={"input_tokens": 12, "output_tokens": 4},
                    content="",
                )
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return _Bound()

    def invoke(self, messages):
        if self.on_text is not None:
            return self.on_text(messages)
        return SimpleNamespace(
            content="# Report\n\nFindings.",
            usage_metadata={"input_tokens": 20, "output_tokens": 30},
        )


class FakeSearch:
    """Records the queries the researcher actually sends."""

    def __init__(self):
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        return {
            "results": [
                {"url": "https://rbi.org.in/example", "content": "Escrow account required."}
            ]
        }
