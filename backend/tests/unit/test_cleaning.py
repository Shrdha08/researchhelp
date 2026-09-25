from researchhelp.ingestion.cleaning import join_lines, normalize_text, remove_boilerplate
from researchhelp.ingestion.loaders.pdf_loader import Line, PageContent


def line(text, block=0, size=10.0, bold=False):
    return Line(text=text, size=size, bold=bold, block=block)


def test_normalize_text_expands_ligatures_and_whitespace():
    assert normalize_text("eﬃcient  ﬁne-tuning­") == "efficient fine-tuning"


def test_join_lines_dehyphenates_within_block():
    text = join_lines([line("stochastic optimi-"), line("zation of the loss")])
    assert text == "stochastic optimization of the loss"


def test_join_lines_does_not_merge_hyphen_before_capital():
    # "BERT-" + "Large" is a compound, not a word split across lines.
    assert "BERTLarge" not in join_lines([line("BERT-"), line("Large model")])


def test_join_lines_separates_blocks_with_blank_line():
    assert join_lines([line("Heading", block=0), line("Body text", block=1)]) == (
        "Heading\n\nBody text"
    )


TOPICS = ["encoders", "decoders", "retrievers", "rerankers"]


def _page(n, header, footer):
    topic = TOPICS[n - 1]
    body = [line(f"Body line about {topic}, part {i} (see Table {n}).") for i in range(4)]
    body[1] = line("See Table 1 for details.")  # identical mid-page line on every page
    return PageContent(n, [line(header), *body, line(footer)])


def test_remove_boilerplate_drops_running_headers_and_page_numbers():
    pages = [_page(n, "Preprint. Under review.", str(n)) for n in range(1, 5)]
    cleaned = remove_boilerplate(pages)
    for page in cleaned:
        texts = [ln.text for ln in page.lines]
        assert len(texts) == 4
        assert "Preprint. Under review." not in texts and str(page.number) not in texts


def test_remove_boilerplate_masks_digits_in_page_edge_lines():
    pages = [_page(n, f"Page {n} of 4", "Conference 2024") for n in range(1, 5)]
    assert all(len(p.lines) == 4 for p in remove_boilerplate(pages))


def test_remove_boilerplate_keeps_repeated_body_lines():
    # A line repeated on every page but not at a page edge is content, not a header.
    pages = [_page(n, "Header", "Footer") for n in range(1, 5)]
    kept = [ln.text for p in remove_boilerplate(pages) for ln in p.lines]
    assert kept.count("See Table 1 for details.") == 4
