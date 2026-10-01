from typer.testing import CliRunner

from researchhelp.cli import app


def test_inspect_shows_sections_and_pages(widget_pdf):
    result = CliRunner().invoke(app, ["inspect", str(widget_pdf), "--page", "3"])
    assert result.exit_code == 0, result.output
    assert "Sparse Widget Networks for Tabular Data" in result.output
    assert "p3  [Experiments]" in result.output
    assert "p1  [" not in result.output  # page filter applied
