"""Shared LangGraph state for the research agent."""
import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict):
    """The shared state that flows through all nodes.

    Reducer semantics matter here:
    - `notes`, `plan`, and `usage` use `operator.add`, so nodes must return
      ONLY the new items to append — never the full accumulated list.
    - All other keys are plain overwrites; nodes return only the keys they
      actually change (never `{**state, ...}`).
    """

    query: str                                       # Original user query
    domain: str                                      # regulatory / market / general
    plan: Annotated[list[str], operator.add]         # All sub-questions so far (accumulates)
    pending_questions: list[str]                     # Work queue: questions not yet researched
    notes: Annotated[list[dict], operator.add]       # Research findings (accumulates)
    usage: Annotated[list[dict], operator.add]       # Per-call token records (accumulates)
    iteration: int                                   # Loop counter
    max_iterations: int                              # Cap copied onto state for this run
    critique: str                                    # Feedback from critic
    final_report: str                                # Final output
    should_continue: bool                            # Control flag
