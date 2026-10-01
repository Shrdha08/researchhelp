from typer.testing import CliRunner

from researchhelp.cli import app


def test_inspect_shows_sections_and_pages(widget_pdf):
    result = CliRunner().invoke(app, ["inspect", str(widget_pdf), "--page", "3"])
    assert result.exit_code == 0, result.output
    assert "Sparse Widget Networks for Tabular Data" in result.output
    assert "p3  [Experiments]" in result.output
    assert "p1  [" not in result.output  # page filter applied


def test_relative_paths_resolve_against_repo_root():
    from researchhelp.config.settings import REPO_ROOT, Settings

    s = Settings(_env_file=None, qdrant_path="./data/qdrant_local", data_dir="data")
    assert s.qdrant_path == (REPO_ROOT / "data" / "qdrant_local").resolve()
    assert s.data_dir == (REPO_ROOT / "data").resolve()
