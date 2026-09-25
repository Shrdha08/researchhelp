import pytest

from tests.pdf_factory import gadget_paper, widget_paper


@pytest.fixture(scope="session")
def pdf_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("pdfs")


@pytest.fixture(scope="session")
def widget_pdf(pdf_dir):
    return widget_paper(pdf_dir / "widget.pdf")


@pytest.fixture(scope="session")
def gadget_pdf(pdf_dir):
    return gadget_paper(pdf_dir / "gadget.pdf")
