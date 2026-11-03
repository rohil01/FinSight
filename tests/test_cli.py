"""The documented CLI shape is `finsight research QUERY`."""
from typer.testing import CliRunner

from finsight.cli import app

runner = CliRunner()


def test_research_is_a_subcommand():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "research" in result.stdout

    result = runner.invoke(app, ["research", "--help"])
    assert result.exit_code == 0
    assert "--domain" in result.stdout
    assert "--max-iterations" in result.stdout


def test_unknown_domain_is_rejected_before_any_run():
    result = runner.invoke(app, ["research", "RBI rules", "--domain", "equities"])
    assert result.exit_code == 1
    assert "regulatory" in result.stdout
