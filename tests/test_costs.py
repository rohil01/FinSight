"""Estimated cost footer."""
import pytest

from finsight.config import get_settings
from finsight.costs import apply_cost_footer, summarize_usage, usage_event


def test_usage_event_reads_token_metadata():
    class _Raw:
        def __init__(self):
            self.usage_metadata = {"input_tokens": 15, "output_tokens": 5}

    event = usage_event("planner", _Raw(), "nvidia/nemotron-test")
    assert event == {
        "node": "planner",
        "model": "nvidia/nemotron-test",
        "input_tokens": 15,
        "output_tokens": 5,
    }


def test_usage_event_reads_nvidia_token_usage():
    class _Raw:
        def __init__(self):
            self.usage_metadata = None
            self.response_metadata = {
                "token_usage": {"prompt_tokens": 40, "completion_tokens": 12}
            }

    event = usage_event("researcher", _Raw(), "nvidia/nemotron-test")
    assert event["input_tokens"] == 40
    assert event["output_tokens"] == 12


def test_missing_metadata_counts_as_zero():
    event = usage_event("writer", None, "nvidia/nemotron-test")
    assert event["input_tokens"] == 0
    assert event["output_tokens"] == 0


def test_footer_prices_tokens_from_settings(monkeypatch):
    monkeypatch.setenv("PRICE_INPUT_PER_MILLION", "2")
    monkeypatch.setenv("PRICE_OUTPUT_PER_MILLION", "4")
    get_settings.cache_clear()

    usage = [
        {"node": "planner", "model": "m", "input_tokens": 1_000_000, "output_tokens": 500_000}
    ]
    summary = summarize_usage(usage)
    assert summary["estimated_usd"] == pytest.approx(4.0)

    report = apply_cost_footer("# Report\n\nBody", usage)
    assert report.startswith("# Report")
    assert "## Run cost (estimated)" in report
    assert "$4.0000" in report
    assert "planner" in report


def test_every_report_gets_a_footer_even_with_no_usage():
    report = apply_cost_footer("# Empty", [])
    assert "## Run cost (estimated)" in report
    assert "$0.0000" in report
