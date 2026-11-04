"""The checked-in question file stays in the shape the harness expects."""
from pathlib import Path

import yaml

QUESTIONS = Path(__file__).resolve().parents[1] / "evals" / "questions.yaml"


def test_question_file_covers_regulatory_and_market():
    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    assert 6 <= len(questions) <= 8
    domains = {item["domain"] for item in questions}
    assert domains == {"regulatory", "market"}
    for item in questions:
        assert item["id"]
        assert item["question"].strip()
        assert len(item["keywords"]) >= 3

    demo = next(item for item in questions if item["id"] == "pa-guidelines-upi")
    assert "payment aggregator" in demo["question"].lower()
    assert "UPI" in demo["question"]
