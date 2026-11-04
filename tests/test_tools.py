"""Structured calls keep the chat message so NVIDIA usage is not dropped."""
from types import SimpleNamespace

from pydantic import BaseModel

from finsight.tools import invoke_structured


class _Answer(BaseModel):
    answer: str


class _Bound:
    def __init__(self, message):
        self.message = message
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.message


class _BindLLM:
    def __init__(self, message):
        self.message = message
        self.bound = _Bound(message)
        self.response_format = None

    def bind(self, **kwargs):
        self.response_format = kwargs.get("response_format")
        return self.bound


def test_json_schema_call_keeps_token_usage():
    llm = _BindLLM(
        SimpleNamespace(
            content='{"answer": "ok"}',
            usage_metadata={"input_tokens": 21, "output_tokens": 8},
        )
    )

    parsed, event = invoke_structured(llm, _Answer, ["question"], "planner", "nvidia/nemotron-test")

    assert parsed.answer == "ok"
    assert event["node"] == "planner"
    assert event["input_tokens"] == 21
    assert event["output_tokens"] == 8
    assert llm.response_format["type"] == "json_schema"
