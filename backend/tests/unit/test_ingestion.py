import pytest

from researchhelp.ingestion.chunking import chunk_segments
from researchhelp.ingestion.cleaning import remove_boilerplate
from researchhelp.ingestion.loaders.pdf_loader import Line, load_pdf
from researchhelp.ingestion.metadata.paper_meta import extract_title, paper_id_from_sha256
from researchhelp.ingestion.metadata.sections import (
    REFERENCES,
    Segment,
    heading_title,
    merge_split_numbers,
    segment_pages,
)
from researchhelp.ingestion.pipeline import parse_pdf


@pytest.mark.parametrize(
    ("text", "bold", "size", "expected"),
    [
        ("1 Introduction", True, 12, "Introduction"),
        ("3.2 Training Details", True, 10, "Training Details"),
        ("IV. RESULTS", False, 12, "Results"),
        ("A Hyperparameters", True, 10, "Hyperparameters"),
        ("Abstract", True, 10, "Abstract"),
        ("REFERENCES", False, 12, REFERENCES),
        ("Related Work:", True, 10, "Related Work"),
        # Not headings:
        ("1 Introduction", False, 10, None),  # body font, not bold
        ("We train the model for 10 epochs and report accuracy.", True, 10, None),
        ("3 The results are shown below.", True, 10, None),  # bold sentence
        ("Figure 3: Accuracy per epoch", False, 10, None),
        ("Model", True, 10, None),  # bold table header, unnumbered generic word
        ("7B Yes", True, 10, None),  # table cell
        ("1 Jurassic-1 (178B)", True, 10, None),  # table row
        ("N + 1[", True, 12, None),  # formula fragment
    ],
)
def test_heading_title(text, bold, size, expected):
    assert heading_title(Line(text, size=size, bold=bold, block=0), body_size=10.0) == expected


def test_paper_id_is_deterministic_prefix_of_hash():
    assert paper_id_from_sha256("ab" * 32) == "abababababababab"


def test_load_pdf_skips_nothing_important(widget_pdf):
    pdf = load_pdf(widget_pdf)
    assert pdf.num_pages == 3
    all_text = " ".join(line.text for page in pdf.pages for line in page.lines)
    assert "AdamW" in all_text


def test_title_from_largest_font_when_metadata_missing(widget_pdf):
    assert extract_title(load_pdf(widget_pdf)) == "Sparse Widget Networks for Tabular Data"


def test_title_prefers_metadata(gadget_pdf):
    assert extract_title(load_pdf(gadget_pdf)) == "Graph Gadgets for Molecular Property Prediction"


def test_parse_pdf_sections_and_pages(widget_pdf):
    paper = parse_pdf(widget_pdf)
    assert paper.num_pages == 3
    # The title-only front-matter chunk is below MIN_CHUNK_CHARS; the title lives in metadata.
    assert paper.sections == [
        "Abstract",
        "Introduction",
        "Method",
        "Experiments",
        REFERENCES,
    ]
    by_text = {c.text: c for c in paper.chunks}
    adamw = next(c for t, c in by_text.items() if "AdamW" in t)
    assert (adamw.page, adamw.section) == (3, "Experiments")
    # Section carries across the page break: page 2 starts mid-Introduction.
    continuing = next(c for t, c in by_text.items() if t.startswith("Continuing the introduction"))
    assert (continuing.page, continuing.section) == (2, "Introduction")


def test_parse_pdf_cleans_text(widget_pdf):
    text = " ".join(c.text for c in parse_pdf(widget_pdf).chunks)
    assert "optimization" in text and "optimi-" not in text
    assert "Preprint. Under review." not in text


def test_chunks_never_cross_pages_and_respect_size(widget_pdf):
    paper = parse_pdf(widget_pdf, chunk_size=200, chunk_overlap=40)
    segments = segment_pages(remove_boilerplate(load_pdf(widget_pdf).pages))
    page_text = {s.page: "" for s in segments}
    for s in segments:
        page_text[s.page] += s.text + "\n\n"
    assert len(paper.chunks) > 5
    for chunk in paper.chunks:
        assert len(chunk.text) <= 200
        # The whole chunk is found on the page it claims, i.e. it did not cross a page break.
        assert chunk.text in page_text[chunk.page]


def test_chunk_ids_are_stable_and_unique():
    segments = [Segment(1, "Intro", "a " * 300), Segment(2, "Method", "b " * 300)]
    first = chunk_segments(segments, "pid", "Title", chunk_size=200, chunk_overlap=0)
    second = chunk_segments(segments, "pid", "Title", chunk_size=200, chunk_overlap=0)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.point_id for c in first}) == len(first)
    assert first[0].chunk_id == "pid:p1:c0"
    assert [c.chunk_index for c in first] == list(range(len(first)))


def test_same_file_gives_same_paper_id(widget_pdf):
    assert parse_pdf(widget_pdf).paper_id == parse_pdf(widget_pdf).paper_id


def test_affiliation_marker_is_not_a_heading_in_front_matter():
    line = Line("1 Inria, Paris", size=12, bold=True, block=0)
    assert heading_title(line, body_size=10.0, in_front_matter=True) is None


def test_small_caps_heading_with_number_on_separate_line():
    # ICLR style: "1" and "INTRODUCTION" are separate lines; small caps start with a larger letter.
    lines = [
        Line("1", size=12, bold=False, block=0),
        Line("INTRODUCTION", size=12, bold=False, block=1),
        Line("Body text of the introduction.", size=10, bold=False, block=2),
    ]
    merged = merge_split_numbers(lines, body_size=10.0)
    assert [ln.text for ln in merged] == ["1 INTRODUCTION", "Body text of the introduction."]
    assert heading_title(merged[0], body_size=10.0) == "Introduction"
