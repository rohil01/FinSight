"""Token totals and the estimated-cost footer appended to every report."""

from .config import get_settings


def usage_event(node: str, raw, model: str) -> dict[str, object]:
    """Build one usage record from an LLM response. Missing metadata counts as zero."""
    input_tokens, output_tokens = _token_counts(raw)
    return {
        "node": node,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def summarize_usage(usage: list[dict]) -> dict[str, object]:
    """Sum token counts and price them with the configured per-million rates."""
    settings = get_settings()
    input_tokens = sum(int(event.get("input_tokens") or 0) for event in usage)
    output_tokens = sum(int(event.get("output_tokens") or 0) for event in usage)
    estimated_usd = (
        input_tokens / 1_000_000 * settings.price_input_per_million
        + output_tokens / 1_000_000 * settings.price_output_per_million
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_usd": estimated_usd,
        "events": list(usage),
    }


def format_cost_footer(usage: list[dict]) -> str:
    """Markdown section with a per-call table and an estimated USD total."""
    settings = get_settings()
    summary = summarize_usage(usage)
    lines = [
        "",
        "---",
        "",
        "## Run cost (estimated)",
        "",
        "| Node | Model | Input tokens | Output tokens |",
        "| --- | --- | ---: | ---: |",
    ]
    if not usage:
        lines.append("| — | — | 0 | 0 |")
    else:
        for event in usage:
            lines.append(
                "| {node} | {model} | {input_tokens} | {output_tokens} |".format(
                    node=event.get("node", ""),
                    model=event.get("model", ""),
                    input_tokens=int(event.get("input_tokens") or 0),
                    output_tokens=int(event.get("output_tokens") or 0),
                )
            )
    lines.extend(
        [
            "",
            (
                f"**Total:** {summary['input_tokens']} input tokens, "
                f"{summary['output_tokens']} output tokens. "
                f"**Estimated cost:** ${summary['estimated_usd']:.4f}."
            ),
            "",
            (
                "Prices are estimates from `PRICE_INPUT_PER_MILLION` and "
                "`PRICE_OUTPUT_PER_MILLION` "
                f"(${settings.price_input_per_million} / 1M input, "
                f"${settings.price_output_per_million} / 1M output). "
                "Token counts come from provider usage metadata."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def apply_cost_footer(report: str, usage: list[dict]) -> str:
    """Append the cost section to a finished report."""
    return report.rstrip() + "\n" + format_cost_footer(usage)


def _token_counts(raw) -> tuple:
    """Read input and output token counts from a LangChain response.

    NVIDIA NIM messages expose usage_metadata. Older payloads keep the same
    numbers under response_metadata.token_usage as prompt_tokens and
    completion_tokens.
    """
    if raw is None:
        return 0, 0
    meta = getattr(raw, "usage_metadata", None)
    if meta is None and isinstance(raw, dict):
        meta = raw.get("usage_metadata")
    if not meta:
        response_meta = getattr(raw, "response_metadata", None)
        if response_meta is None and isinstance(raw, dict):
            response_meta = raw.get("response_metadata")
        if isinstance(response_meta, dict):
            meta = response_meta.get("token_usage") or response_meta.get("usage")
    if not meta:
        return 0, 0
    if not isinstance(meta, dict):
        meta = {
            "input_tokens": getattr(meta, "input_tokens", 0) or getattr(meta, "prompt_tokens", 0),
            "output_tokens": getattr(meta, "output_tokens", 0)
            or getattr(meta, "completion_tokens", 0),
        }
    input_tokens = meta.get("input_tokens") or meta.get("prompt_tokens") or 0
    output_tokens = meta.get("output_tokens") or meta.get("completion_tokens") or 0
    return int(input_tokens), int(output_tokens)
