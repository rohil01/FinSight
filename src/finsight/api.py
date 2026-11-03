"""FastAPI endpoint that streams LangGraph node events as server-sent events."""
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .graph import aiter_run

app = FastAPI(
    title="FinSight",
    description="Autonomous research analyst for Indian fintech.",
    version=__version__,
)


class ResearchRequest(BaseModel):
    """Body for POST /research."""

    query: str = Field(min_length=1)
    domain: Literal["regulatory", "market", "general"] | None = None
    max_iterations: int | None = Field(default=None, ge=1)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/research")
async def research(body: ResearchRequest) -> StreamingResponse:
    """Stream node updates, then a final report event with the cost footer."""

    async def events():
        try:
            async for event in aiter_run(body.query, body.domain, body.max_iterations):
                if event["type"] == "node":
                    yield _sse(
                        "node",
                        {"node": event["node"], "iteration": event.get("iteration")},
                    )
                elif event["type"] == "report":
                    yield _sse("report", {"report": event["report"]})
        except RuntimeError as exc:
            yield _sse("error", {"message": str(exc)})

    try:
        from .config import get_settings

        get_settings().validate_keys()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
