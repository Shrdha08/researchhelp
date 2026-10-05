"""Generation evaluation with an LLM judge (a different model family from the generator).

    uv run --project backend python -m evaluation.generation_eval --split test \
        --strategies semantic hybrid_rerank

Stage 1 runs the production evidence chain for each (strategy, question) and caches the answer
and the retrieved context. Stage 2 asks the judge for structured verdicts and computes:

  faithfulness       RAGAS definition: the judge splits the answer into atomic claims and marks
                     each as supported / not supported by the retrieved context.
                     score = supported / total claims.
  answer_relevance   RAGAS method: the judge writes 3 questions the answer would answer; score =
                     mean cosine similarity (bge-small) between them and the real question.
                     Measures whether the answer addresses the question, not whether it's true.
  context_relevance  the judge marks each retrieved chunk as relevant to the question or not;
                     score = relevant / retrieved.
  correctness        the judge compares the answer with the reference answer:
                     correct = 1, partial = 0.5, incorrect = 0.
                     For unanswerable questions "correct" means the system said it is not stated.

Every LLM output is cached under evaluation/results/generation_cache/, so a run interrupted by
Groq's free-tier limits can simply be restarted. A manual-review sheet is written so a human
can score a sample and check agreement with the judge.
"""

import argparse
import json
import math
import random
import statistics
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from evaluation.common import RESULTS, Question, load_questions
from researchhelp.config import get_settings
from researchhelp.rag.common.llm import get_chat_model
from researchhelp.rag.evidence.chain import build_evidence_chain
from researchhelp.retrieval.factory import get_store
from researchhelp.retrieval.reranking import CrossEncoderReranker
from researchhelp.retrieval.retriever import STRATEGIES, ScopedRetriever

CACHE = RESULTS / "generation_cache"

JUDGE_SYSTEM = (
    "You are a strict, careful evaluator of question-answering systems. "
    "Respond with a single JSON object and nothing else."
)

# One judge call per answer: Groq's free tier caps the judge at 1,000 output tokens per minute,
# so three separate calls per answer would triple the run time.
JUDGE_PROMPT = """QUESTION:
{question}

REFERENCE ANSWER (ground truth written by a human):
{reference}

RETRIEVED CONTEXT (numbered chunks shown to the system):
{chunks}

SYSTEM ANSWER:
{answer}

Do four things:
1. claims: break the SYSTEM ANSWER into short atomic factual claims (at most 15 words each; \
ignore citation markers like [S1] and statements that merely say information is missing). Mark \
a claim supported only if the RETRIEVED CONTEXT states it or it follows unambiguously from it.
2. questions: write 3 different questions that the SYSTEM ANSWER would be a good response to.
3. verdict: grade the SYSTEM ANSWER against the REFERENCE ANSWER: "correct" if it conveys the \
key facts of the reference without contradicting it, "partial" if it gets some key facts but \
misses or gets others wrong, "incorrect" otherwise. If the reference says the information is \
not stated, the answer is "correct" only if it says the papers/excerpts do not provide it.
4. relevant_chunks: for each numbered chunk, whether it is relevant to answering the QUESTION.

Return JSON only:
{{"claims": [{{"claim": "...", "supported": true}}], "questions": ["...", "...", "..."], \
"verdict": "correct|partial|incorrect", "reason": "<one sentence>", \
"relevant_chunks": [true, false]}}"""

JUDGE_MAX_TOKENS = 900


def _cache_path(*parts: str) -> Path:
    return CACHE.joinpath(*parts).with_suffix(".json")


def _cached(path: Path, compute):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return value


def _judge_json(judge, prompt: str) -> dict:
    """Call the judge in JSON mode; retry once on malformed output."""
    for _ in range(2):
        reply = judge.invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(prompt)])
        text = reply.content.strip()
        try:
            return json.loads(text[text.index("{") : text.rindex("}") + 1])
        except ValueError:
            continue
    raise ValueError(f"Judge returned invalid JSON: {text[:200]!r}")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def generate(strategy_name: str, chain, q: Question) -> dict:
    def run():
        result = chain.invoke({"question": q.question, "paper_ids": q.paper_ids})
        return {
            "answer": result.answer,
            "citations": [c.to_dict() for c in result.citations],
            "contexts": [
                {
                    "text": d.page_content,
                    "paper_id": d.metadata["paper_id"],
                    "page": d.metadata["page"],
                    "section": d.metadata["section"],
                }
                for d in result.sources
            ],
        }

    return _cached(_cache_path(strategy_name, q.id, "generation"), run)


def judge(strategy_name: str, q: Question, gen: dict, judge_llm, embedder) -> dict:
    chunks = "\n\n".join(f"[{i}] {c['text'][:1200]}" for i, c in enumerate(gen["contexts"], 1))
    out = _cached(
        _cache_path(strategy_name, q.id, "judge"),
        lambda: _judge_json(
            judge_llm,
            JUDGE_PROMPT.format(
                question=q.question,
                reference=q.reference_answer,
                chunks=chunks or "(nothing retrieved)",
                answer=gen["answer"],
            ),
        ),
    )
    verdict = str(out.get("verdict", "")).lower()
    flags = [bool(x) for x in out.get("relevant_chunks", [])][: len(gen["contexts"])]
    scores = {
        "correctness": {"correct": 1.0, "partial": 0.5}.get(verdict, 0.0),
        "context_relevance": (sum(flags) / len(gen["contexts"])) if gen["contexts"] else None,
        "faithfulness": None,
        "answer_relevance": None,
    }
    if q.type == "unanswerable" or not gen["contexts"]:
        return scores  # faithfulness/relevance are defined for substantive answers only

    claims = [c for c in out.get("claims", []) if isinstance(c, dict)]
    if claims:
        scores["faithfulness"] = sum(bool(c.get("supported")) for c in claims) / len(claims)
    generated = [g for g in out.get("questions", []) if isinstance(g, str) and g.strip()]
    if generated:
        q_vec = embedder.embed_documents([q.question])[0]
        sims = [_cosine(q_vec, v) for v in embedder.embed_documents(generated)]
        scores["answer_relevance"] = statistics.mean(sims)
    return scores


def summarise(name: str, rows: list[dict]) -> dict:
    def mean(key, subset):
        vals = [r[key] for r in subset if r[key] is not None]
        return round(statistics.mean(vals), 3) if vals else None

    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    return {
        "config": name,
        "n": len(rows),
        "faithfulness": mean("faithfulness", answerable),
        "answer_relevance": mean("answer_relevance", answerable),
        "context_relevance": mean("context_relevance", answerable),
        "correctness": mean("correctness", answerable),
        "refusal_accuracy": mean("correctness", unanswerable),
        "n_unanswerable": len(unanswerable),
    }


def write_manual_review(rows_by_config: dict, questions: dict, path: Path, n: int = 15):
    """Sample answers for a human to grade blind (no judge verdicts or config names shown); the
    judge's verdicts go to a separate *_key.md file for computing agreement afterwards."""
    pool = [(cfg, r) for cfg, rows in rows_by_config.items() for r in rows]
    random.Random(0).shuffle(pool)
    sheet = [
        "# Manual review sheet (blind)",
        "For each answer write: correct / partial / incorrect (against the reference), and "
        "whether every claim is supported by the cited pages (yes / no). Only open the key file "
        "after grading all items.\n",
    ]
    key = [
        "# Judge verdicts (key for the manual review sheet)\n",
        "| # | Question | Config | Judge correctness | Judge faithfulness |",
        "|---|---|---|--:|--:|",
    ]
    for i, (cfg, r) in enumerate(pool[:n], 1):
        q = questions[r["id"]]
        sheet += [
            f"## {i}. {r['id']}",
            f"**Q:** {q.question}",
            f"**Reference:** {q.reference_answer}",
            f"**System answer:**\n\n{r['answer']}\n",
            "Your grade: ______   All claims supported? ______\n",
        ]
        key.append(f"| {i} | {r['id']} | {cfg} | {r['correctness']} | {r['faithfulness']} |")
    path.write_text("\n".join(sheet), encoding="utf-8")
    path.with_name(path.stem + "_key.md").write_text("\n".join(key) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test", "all"])
    parser.add_argument(
        "--strategies", nargs="+", default=["semantic", "hybrid_rerank"], choices=STRATEGIES
    )
    parser.add_argument("--reranker", default="Xenova/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--k-candidates", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="First N questions only")
    args = parser.parse_args()

    settings = get_settings()
    questions = load_questions(args.split)[: args.limit]
    store = get_store()
    generator = get_chat_model()
    judge_llm = get_chat_model(model=settings.judge_model, temperature=0.0).bind(
        response_format={"type": "json_object"}, max_tokens=JUDGE_MAX_TOKENS
    )

    rows_by_config, summaries = {}, []
    for strategy in args.strategies:
        reranker = CrossEncoderReranker(args.reranker) if strategy == "hybrid_rerank" else None
        retriever = ScopedRetriever(
            store,
            strategy=strategy,
            reranker=reranker,
            k_final=settings.k_final,
            k_per_paper=settings.k_per_paper,
            k_candidates=args.k_candidates,
        )
        name = strategy if not reranker else f"hybrid_rerank[{args.reranker.split('/')[-1]}]"
        chain = build_evidence_chain(retriever, generator)
        rows = []
        for i, q in enumerate(questions, 1):
            print(f"[{name}] {i}/{len(questions)} {q.id}", flush=True)
            gen = generate(name, chain, q)
            scores = judge(name, q, gen, judge_llm, store.dense)
            rows.append({"id": q.id, "type": q.type, "answer": gen["answer"], **scores})
        rows_by_config[name] = rows
        summaries.append(summarise(name, rows))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = RESULTS / f"generation_{args.split}_{stamp}"
    base.with_suffix(".json").write_text(
        json.dumps({"summary": summaries, "rows": rows_by_config}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    table = [
        "| Config | Faithfulness | Answer relevance | Context relevance | Correctness | "
        "Refusal accuracy |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for s in summaries:
        table.append(
            f"| {s['config']} | {s['faithfulness']} | {s['answer_relevance']} | "
            f"{s['context_relevance']} | {s['correctness']} | {s['refusal_accuracy']} "
            f"(n={s['n_unanswerable']}) |"
        )
    md = (
        f"## Generation results ({args.split} split, {len(questions)} questions, "
        f"generator {settings.llm_model}, judge {settings.judge_model})\n\n" + "\n".join(table)
    )
    base.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    write_manual_review(
        rows_by_config, {q.id: q for q in questions}, RESULTS / f"manual_review_{stamp}.md"
    )
    print("\n" + md)


if __name__ == "__main__":
    main()
