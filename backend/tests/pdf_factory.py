"""Builds small synthetic research-paper PDFs with PyMuPDF so tests are deterministic and need
no downloaded (licensed) papers."""

from pathlib import Path

import pymupdf

STYLES = {
    # kind: (fontname, fontsize)
    "title": ("hebo", 18),
    "heading": ("hebo", 12),
    "body": ("helv", 10),
    "header": ("helv", 8),
}
PAGE_W, PAGE_H = 612, 792
MARGIN = 72


def build_pdf(
    path: Path,
    pages: list[list[tuple[str, str]]],
    running_header: str | None = None,
    page_numbers: bool = True,
    title_metadata: str = "",
) -> Path:
    """``pages`` is a list of pages; each page is a list of (kind, text) items laid out top-down.
    Body text is wrapped inside a textbox; explicit "\\n" forces a line break."""
    doc = pymupdf.open()
    for page_no, items in enumerate(pages, start=1):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        if running_header:
            page.insert_text((MARGIN, 40), running_header, fontname="helv", fontsize=8)
        y = MARGIN
        for kind, text in items:
            font, size = STYLES[kind]
            rect = pymupdf.Rect(MARGIN, y, PAGE_W - MARGIN, PAGE_H - MARGIN)
            spare = page.insert_textbox(rect, text, fontname=font, fontsize=size)
            assert spare >= 0, f"text did not fit on page {page_no}: {text[:40]!r}"
            used = (PAGE_H - MARGIN - y) - spare
            y += used + size * 1.2
        if page_numbers:
            page.insert_text((PAGE_W / 2, PAGE_H - 30), str(page_no), fontname="helv", fontsize=8)
    if title_metadata:
        doc.set_metadata({"title": title_metadata})
    doc.save(path)
    doc.close()
    return path


LOREM = (
    "The proposed architecture processes each input through a sequence of layers that "
    "progressively refine the representation. We describe each component in detail and "
    "justify the design choices with ablation studies reported later in the paper. "
)


def widget_paper(path: Path) -> Path:
    """Three-page paper about 'Sparse Widget Networks' with known sections per page."""
    return build_pdf(
        path,
        running_header="Preprint. Under review.",
        pages=[
            [
                ("title", "Sparse Widget Networks for Tabular Data"),
                ("heading", "Abstract"),
                (
                    "body",
                    "We introduce Sparse Widget Networks (SWN), a model for tabular "
                    "classification that routes features through sparse widgets. " + LOREM,
                ),
                ("heading", "1 Introduction"),
                ("body", "Tabular data remains dominated by gradient boosted trees. " + LOREM * 2),
            ],
            [
                (
                    "body",
                    "Continuing the introduction, prior neural approaches underperform on "
                    "small tabular datasets. " + LOREM,
                ),
                ("heading", "2 Method"),
                (
                    "body",
                    "Each widget selects a sparse subset of features. The network is\n"
                    "trained end to end with stochastic optimi-\n"
                    "zation of the cross entropy loss. " + LOREM,
                ),
            ],
            [
                ("heading", "3 Experiments"),
                (
                    "body",
                    "We evaluate SWN on the UCI Adult dataset and the Covertype dataset. "
                    "Models are trained with the AdamW optimizer and a learning rate of "
                    "3e-4 for 50 epochs. SWN reaches 87.1% accuracy on Adult. " + LOREM,
                ),
                ("heading", "References"),
                (
                    "body",
                    "[1] T. Chen and C. Guestrin. XGBoost: A scalable tree boosting system. "
                    "In KDD, 2016.\n[2] A. Author. Adam optimizer for tabular data. 2020.",
                ),
            ],
        ],
    )


def gadget_paper(path: Path) -> Path:
    """Two-page paper on a different topic, used to check retrieval scoping."""
    return build_pdf(
        path,
        running_header="Workshop on Molecules 2024",
        title_metadata="Graph Gadgets for Molecular Property Prediction",
        pages=[
            [
                ("title", "Graph Gadgets for Molecular Property Prediction"),
                ("heading", "Abstract"),
                (
                    "body",
                    "We propose Graph Gadgets, a message passing network for predicting "
                    "molecular properties from molecular graphs. " + LOREM,
                ),
                ("heading", "1 Introduction"),
                ("body", "Molecular property prediction is central to drug discovery. " + LOREM),
            ],
            [
                ("heading", "2 Experiments"),
                (
                    "body",
                    "We evaluate Graph Gadgets on the QM9 dataset and the ZINC dataset "
                    "using the Adam optimizer with a learning rate of 1e-3. " + LOREM,
                ),
                ("heading", "3 Limitations"),
                (
                    "body",
                    "Graph Gadgets do not model 3D conformations and were only tested on "
                    "small molecules. " + LOREM,
                ),
            ],
        ],
    )
