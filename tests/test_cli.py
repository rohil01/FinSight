"""The documented CLI shape is `finsight research QUERY`."""
import re

from typer.testing import CliRunner

from finsight.cli import app

runner = CliRunner()
# Rich styles each dash of a long option on its own, so "--domain" is not a
# contiguous substring of the colored help text.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def test_research_is_a_subcommand():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "research" in _plain(result.stdout)

    result = runner.invoke(app, ["research", "--help"])
    assert result.exit_code == 0
    help_text = _plain(result.stdout)
    assert "--domain" in help_text
    assert "--max-iterations" in help_text


def test_unknown_domain_is_rejected_before_any_run():
    result = runner.invoke(app, ["research", "RBI rules", "--domain", "equities"])
    assert result.exit_code == 1
    assert "regulatory" in result.stdout
