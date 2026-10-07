import pytest

from researchhelp.graph.routing import keyword_intent


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        # evidence: stated in the papers
        ("What dataset does this paper use?", "evidence"),
        ("What optimizer and learning rate were used?", "evidence"),
        ("Compare the methodologies of these papers.", "evidence"),
        ("What limitations did the authors identify?", "evidence"),
        ("What future work do the authors mention?", "evidence"),
        ("Which limitations are acknowledged in the papers?", "evidence"),
        ("What were the experimental results?", "evidence"),
        # research: inference beyond what is stated
        ("What are the limitations of these approaches?", "research"),
        ("What research gaps exist across these papers?", "research"),
        ("How could the retriever be improved?", "research"),
        ("Suggest a new research direction.", "research"),
        ("How could this approach be adapted to code search?", "research"),
        ("What experiments could validate an improvement?", "research"),
        ("What possible extensions could be explored?", "research"),
    ],
)
def test_keyword_intent(question, intent):
    assert keyword_intent(question)[0] == intent
