"""Intent routing: should a question go to evidence Q&A or to the research assistant?

Decision rule (ADR-13):
- evidence: what the papers state, including results, methods, datasets and the limitations
  the AUTHORS themselves identify ("What limitations did the authors identify?"), and
  comparisons of what papers did.
- research: requests for inferred gaps, weaknesses of the approaches, improvements,
  extensions, new directions, hypotheses, experiments to run, or adapting a method elsewhere.
  Generic "What are the limitations of these approaches?" counts as research: it asks for an
  assessment, not a lookup.

The LLM router (small, fast model) follows few-shot examples; a keyword router is the fallback
when the LLM call fails or returns an invalid label, and serves as the baseline in evaluation.
"""

import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from researchhelp.rag.common.structured import StructuredOutputError, invoke_json, json_mode

log = logging.getLogger(__name__)

_AUTHOR_STATED = re.compile(
    r"\b(authors?|papers?|they|study|studies)\b[^?]{0,40}\b("
    r"identif\w*|mention\w*|state\w*|report\w*|acknowledg\w*|discuss\w*|list\w*|note\w*|admit\w*"
    r")\b",
    re.IGNORECASE,
)
_RESEARCH = re.compile(
    r"\b("
    r"gaps?|improv\w*|extend\w*|extension\w*|future (research|work|directions?)|"
    r"new (research )?directions?|research directions?|research questions?|open problems?|"
    r"suggest\w*|propos\w*|hypothes\w*|novel|next steps?|could (we|this|it|they|be)|"
    r"how (might|could|would|can)|what experiments?|experiments? (could|should|to)|"
    r"adapt\w*|apply (this|these|it) to|weakness\w*|limitations? of (these|this|the|their)|"
    r"shortcomings?|critique|criticis\w*"
    r")\b",
    re.IGNORECASE,
)


def keyword_intent(question: str) -> tuple[str, str]:
    """Rule-based fallback router. Returns (intent, reason)."""
    if _AUTHOR_STATED.search(question):
        return "evidence", "asks what the papers/authors state"
    match = _RESEARCH.search(question)
    if match:
        return "research", f"asks for inference or new ideas ('{match.group(0)}')"
    return "evidence", "factual question about the papers"


class RouteDecision(BaseModel):
    intent: Literal["evidence", "research"]
    reason: str = ""


ROUTER_SYSTEM = """You route questions about research papers to one of two pipelines.

"evidence": the answer is stated in the papers. Results, methods, datasets, hyperparameters, \
comparisons of what papers did, and limitations or future work that the AUTHORS themselves \
mention.
"research": the user wants analysis beyond what is stated: research gaps, weaknesses of the \
approaches, how to improve or extend them, new research directions, hypotheses, experiments to \
run, or adapting a method to another problem.

Examples:
- "What dataset does this paper use?" -> evidence
- "Compare the methodologies of these papers." -> evidence
- "What limitations did the authors identify?" -> evidence
- "What future work do the authors mention?" -> evidence
- "What are the limitations of these approaches?" -> research
- "What research gaps exist across these papers?" -> research
- "How could the retriever be improved?" -> research
- "Suggest a new research direction based on these papers." -> research
- "How could this approach be adapted to code search?" -> research

Return JSON only: {"intent": "evidence" or "research", "reason": "<short reason>"}"""


def classify_intent(llm, question: str, config=None) -> tuple[str, str, str]:
    """Returns (intent, reason, source) where source is "llm" or "keyword"."""
    if llm is not None:
        try:
            decision = invoke_json(
                json_mode(llm),
                [SystemMessage(ROUTER_SYSTEM), HumanMessage(f"Question: {question}")],
                RouteDecision,
                config,
            )
            return decision.intent, decision.reason, "llm"
        except StructuredOutputError as exc:
            log.warning("router returned invalid output, using keyword fallback: %s", exc)
        except Exception as exc:  # network / rate-limit errors must not break answering
            log.warning("router call failed, using keyword fallback: %s", exc)
    intent, reason = keyword_intent(question)
    return intent, reason, "keyword"
