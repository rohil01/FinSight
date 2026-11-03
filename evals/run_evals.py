"""Score the fintech question set with a live research run and a flash judge.

Writes evals/results.json and evals/results.md. Exits without writing scores
when GOOGLE_API_KEY or TAVILY_API_KEY is missing.
"""
import json
import sys
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finsight.config import get_settings
from finsight.graph import run_research
from finsight.tools import get_fast_llm, invoke_structured

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")
RESULTS_JSON = Path(__file__).with_name("results.json")
RESULTS_MD = Path(__file__).with_name("results.md")


class JudgeScore(BaseModel):
    """Scores for one generated report. Integers are on a 0–5 scale."""

    coverage: int = Field(ge=0, le=5, description="How much of the expected coverage is present")
    citation_presence: bool = Field(description="Whether the report cites sources with URLs or circular ids")
    citation_faithfulness: int = Field(
        ge=0, le=5, description="Whether cited claims stay within the supplied report text"
    )
    structure: int = Field(ge=0, le=5, description="Whether the report has the expected sections")
    rationale: str


def _judge_prompt(question: dict, report: str) -> str:
    keywords = ", ".join(question["keywords"])
    return (
        "You are grading a research report. Score only what is written in the report.\n\n"
        f"Question: {question['question']}\n"
        f"Domain: {question['domain']}\n"
        f"Expected-coverage keywords: {keywords}\n\n"
        "Score coverage by how many of those topics the report actually addresses (0 = none, 5 = all).\n"
        "citation_presence is true only if the report includes source URLs or circular numbers.\n"
        "citation_faithfulness scores whether claims are tied to those sources rather than unsourced (0–5).\n"
        "structure scores whether the report is organized into clear sections (0–5).\n\n"
        f"Report:\n{report}"
    )


def _markdown(rows: list) -> str:
    lines = [
        "# Eval results",
        "",
        "| Question | Domain | Coverage | Citations | Faithfulness | Structure |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        if row.get("error"):
            lines.append(
                f"| {row['id']} | {row['domain']} | — | — | — | — |"
            )
            continue
        citations = "yes" if row["citation_presence"] else "no"
        lines.append(
            f"| {row['id']} | {row['domain']} | {row['coverage']} | {citations} | "
            f"{row['citation_faithfulness']} | {row['structure']} |"
        )
    lines.extend(
        [
            "",
            "Scores are 0–5 from `gemini` via `FAST_MODEL_NAME`, except citation presence.",
            "Regenerate with `python evals/run_evals.py`.",
            "",
        ]
    )
    errors = [row for row in rows if row.get("error")]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for row in errors:
            lines.append(f"- **{row['id']}**: {row['error']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    try:
        get_settings().validate_keys()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        print("Refusing to write scores without API keys.", file=sys.stderr)
        raise SystemExit(1)

    questions = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    settings = get_settings()
    rows = []
    for question in questions:
        print(f"Running {question['id']}...", flush=True)
        try:
            report = run_research(question["question"], domain=question["domain"])
            score, _event = invoke_structured(
                get_fast_llm(),
                JudgeScore,
                [HumanMessage(content=_judge_prompt(question, report))],
                "judge",
                settings.fast_model_name,
            )
            rows.append(
                {
                    "id": question["id"],
                    "domain": question["domain"],
                    "question": question["question"],
                    "coverage": score.coverage,
                    "citation_presence": score.citation_presence,
                    "citation_faithfulness": score.citation_faithfulness,
                    "structure": score.structure,
                    "rationale": score.rationale,
                }
            )
        except Exception as exc:  # noqa: BLE001 — record the failure and keep scoring
            rows.append(
                {
                    "id": question["id"],
                    "domain": question["domain"],
                    "question": question["question"],
                    "error": str(exc),
                }
            )

    RESULTS_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    RESULTS_MD.write_text(_markdown(rows), encoding="utf-8")
    print(f"Wrote {RESULTS_JSON} and {RESULTS_MD}")
    if any(row.get("error") for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
