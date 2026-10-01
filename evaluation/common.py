"""Evaluation dataset loading, gold-label validation, and relevance matching.

Gold evidence is labelled as (paper, page, quote), not as chunk IDs. A retrieved chunk counts
as relevant to an evidence item when it is from the same paper and page AND contains the quote.
Because labels do not reference chunk IDs, they stay valid when chunking settings change.
"""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "evaluation" / "datasets"
RESULTS = ROOT / "evaluation" / "results"
PAPERS_DIR = ROOT / "data" / "papers"
TEST_FRACTION = 0.3


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("−", "-").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Evidence:
    paper: str  # short name from papers.yaml
    paper_id: str
    page: int
    quote: str


@dataclass
class Question:
    id: str
    type: str
    papers: list[str]
    paper_ids: list[str]
    question: str
    reference_answer: str
    evidence: list[Evidence] = field(default_factory=list)
    split: str = "dev"


def split_for(question_id: str) -> str:
    """Deterministic ~70/30 dev/test split by hashing the question id. K values and other
    hyperparameters are tuned on dev only; headline numbers are reported on test."""
    bucket = int(hashlib.md5(question_id.encode()).hexdigest(), 16) % 100
    return "test" if bucket < TEST_FRACTION * 100 else "dev"


def paper_ids() -> dict[str, str]:
    """short_name -> paper_id (sha256 prefix of the PDF, same as ingestion)."""
    corpus = yaml.safe_load((DATASETS / "papers.yaml").read_text(encoding="utf-8"))["papers"]
    ids = {}
    for p in corpus:
        pdf = PAPERS_DIR / f"{p['short_name']}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(f"{pdf} missing; run scripts/download_papers.py")
        ids[p["short_name"]] = hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]
    return ids


def load_questions(split: str | None = None) -> list[Question]:
    ids = paper_ids()
    questions = []
    for line in (DATASETS / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        q = Question(
            id=raw["id"],
            type=raw["type"],
            papers=raw["papers"],
            paper_ids=[ids[p] for p in raw["papers"]],
            question=raw["question"],
            reference_answer=raw["reference_answer"],
            evidence=[
                Evidence(e["paper"], ids[e["paper"]], e["page"], e["quote"])
                for e in raw["evidence"]
            ],
            split=split_for(raw["id"]),
        )
        if split in (None, "all") or q.split == split:
            questions.append(q)
    return questions


def chunk_matches(evidence: Evidence, paper_id: str, page: int, text: str) -> bool:
    return (
        paper_id == evidence.paper_id
        and page == evidence.page
        and norm(evidence.quote) in norm(text)
    )


def page_matches(evidence: Evidence, paper_id: str, page: int) -> bool:
    return paper_id == evidence.paper_id and page == evidence.page
