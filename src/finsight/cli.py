"""Typer CLI for the FinSight research agent."""
import logging
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .costs import summarize_usage
from .domains import DOMAINS

app = typer.Typer(
    name="finsight",
    help="FinSight — an autonomous research analyst for Indian fintech.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root() -> None:
    """Autonomous research analyst for Indian fintech."""
    # A callback keeps this a command group. Without it, Typer flattens the
    # only command and `finsight research "..."` treats "research" as the query.
DEFAULT_OUTPUT = Path("research_report.md")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )
    if not verbose:
        for name in ("httpx", "urllib3", "langchain", "google"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _print_cost_table(usage: list) -> None:
    summary = summarize_usage(usage)
    table = Table(title="Estimated run cost")
    table.add_column("Node")
    table.add_column("Model")
    table.add_column("Input tokens", justify="right")
    table.add_column("Output tokens", justify="right")
    if not usage:
        table.add_row("—", "—", "0", "0")
    else:
        for event in usage:
            table.add_row(
                str(event.get("node", "")),
                str(event.get("model", "")),
                str(int(event.get("input_tokens") or 0)),
                str(int(event.get("output_tokens") or 0)),
            )
    table.add_row(
        "Total",
        "",
        str(summary["input_tokens"]),
        str(summary["output_tokens"]),
    )
    console.print(table)
    console.print(f"[dim]Estimated cost:[/dim] ${summary['estimated_usd']:.4f}")


@app.command()
def research(
    query: Annotated[str, typer.Argument(help="The research question to investigate")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path for the Markdown report"),
    ] = DEFAULT_OUTPUT,
    domain: Annotated[
        str | None,
        typer.Option(
            "--domain",
            help=f"Source profile: {', '.join(DOMAINS)}. Omit to classify the query.",
        ),
    ] = None,
    max_iterations: Annotated[
        int | None,
        typer.Option(
            "--max-iterations",
            help="Max plan-research-critique loops (default from config)",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Gemini model id (default from config)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show debug logging"),
    ] = False,
) -> None:
    """Run the research agent on QUERY and save a cited Markdown report."""
    load_dotenv()
    _setup_logging(verbose)

    if max_iterations is not None and max_iterations < 1:
        console.print("[bold red]Error:[/bold red] --max-iterations must be at least 1")
        raise typer.Exit(code=1)

    if domain is not None and domain not in DOMAINS:
        known = ", ".join(DOMAINS)
        console.print(f"[bold red]Error:[/bold red] --domain must be one of: {known}")
        raise typer.Exit(code=1)

    if model is not None:
        os.environ["MODEL_NAME"] = model

    from .config import get_settings
    from .graph import iter_run
    from .tools import get_llm, get_search_tool

    get_settings.cache_clear()
    get_llm.cache_clear()
    get_search_tool.cache_clear()

    try:
        get_settings().validate_keys()
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    domain_label = domain or "auto"
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "[bold]FinSight Research Agent[/bold]",
                    f"[dim]{query}[/dim]",
                    f"[dim]domain: {domain_label}[/dim]",
                ]
            ),
            border_style="cyan",
        )
    )

    report = ""
    usage: list = []
    with console.status("[cyan]Starting...", spinner="dots") as status:
        try:
            for event in iter_run(query, domain, max_iterations):
                if event["type"] == "node":
                    iteration = event.get("iteration")
                    label = f"iteration {iteration}" if iteration else "in progress"
                    status.update(f"[cyan]Running {event['node']} ({label})...")
                elif event["type"] == "report":
                    report = event["report"]
                    usage = event["usage"]
        except RuntimeError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1)

    console.print("\n[bold green]Research completed[/bold green]\n")
    console.print(Markdown(report))
    _print_cost_table(usage)

    output.write_text(report, encoding="utf-8")
    console.print(f"\n[dim]Report saved to[/dim] [bold]{output}[/bold]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
