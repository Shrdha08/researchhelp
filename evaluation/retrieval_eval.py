"""Retrieval evaluation: Recall@K and MRR for each retrieval strategy. No LLM involved.

    uv run --project backend python -m evaluation.retrieval_eval --split test
    uv run --project backend python -m evaluation.retrieval_eval --strategies hybrid_rerank \
        --reranker Xenova/ms-marco-MiniLM-L-6-v2 --reranker BAAI/bge-reranker-base

Metrics (computed on answerable questions; the scope is the question's own papers):
  Recall@K  fraction of a question's gold evidence items found in the top-K chunks, averaged
            over questions. A 3-paper comparison that only retrieves two papers' evidence
            scores 0.67.
  MRR       mean of 1/rank of the first relevant chunk (0 if none in the top 10).
  PageRecall@5  looser variant: right paper and page, quote not required.

The ranking evaluated is the one the production retriever returns with k_final=10; Recall@5 uses
its first five chunks.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime

from evaluation.common import RESULTS, Question, chunk_matches, load_questions, page_matches
from researchhelp.retrieval.factory import get_store
from researchhelp.retrieval.reranking import CrossEncoderReranker
from researchhelp.retrieval.retriever import STRATEGIES, ScopedRetriever

K_EVAL = 10


def score_question(q: Question, docs) -> dict:
    ranked = [(d.metadata["paper_id"], d.metadata["page"], d.page_content) for d in docs]

    def recall(k: int, matcher) -> float:
        top = ranked[:k]
        found = sum(any(matcher(ev, *hit) for hit in top) for ev in q.evidence)
        return found / len(q.evidence)

    first = next(
        (i for i, hit in enumerate(ranked, 1) if any(chunk_matches(ev, *hit) for ev in q.evidence)),
        None,
    )
    return {
        "recall@5": recall(5, chunk_matches),
        "recall@10": recall(10, chunk_matches),
        "mrr": 1 / first if first else 0.0,
        "page_recall@5": recall(5, lambda ev, pid, page, _t: page_matches(ev, pid, page)),
        "first_relevant_rank": first,
    }


def evaluate(name: str, retriever: ScopedRetriever, questions: list[Question]) -> dict:
    rows, latencies = [], []
    for q in questions:
        start = time.perf_counter()
        docs = retriever.retrieve(q.question, q.paper_ids)
        latencies.append(time.perf_counter() - start)
        rows.append({"id": q.id, "type": q.type, **score_question(q, docs)})

    def mean(key, subset=rows):
        return round(statistics.mean(r[key] for r in subset), 3) if subset else None

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    return {
        "config": name,
        "n": len(rows),
        "recall@5": mean("recall@5"),
        "recall@10": mean("recall@10"),
        "mrr": mean("mrr"),
        "page_recall@5": mean("page_recall@5"),
        "latency_s": round(statistics.median(latencies), 2),
        "by_type": {
            t: {"n": len(rs), "recall@5": mean("recall@5", rs), "mrr": mean("mrr", rs)}
            for t, rs in sorted(by_type.items())
        },
        "questions": rows,
    }


def markdown(results: list[dict], split: str) -> str:
    lines = [
        f"## Retrieval results ({split} split, {results[0]['n']} answerable questions)\n",
        "| Config | Recall@5 | Recall@10 | MRR | PageRecall@5 | Median latency (s) |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']} | {r['recall@5']:.3f} | {r['recall@10']:.3f} | "
            f"{r['mrr']:.3f} | {r['page_recall@5']:.3f} | {r['latency_s']:.2f} |"
        )
    types = sorted(results[0]["by_type"])
    lines += [
        "",
        "Recall@5 by question type:",
        "",
        "| Config | "
        + " | ".join(f"{t} (n={results[0]['by_type'][t]['n']})" for t in types)
        + " |",
        "|---|" + "--:|" * len(types),
    ]
    for r in results:
        lines.append(
            f"| {r['config']} | "
            + " | ".join(f"{r['by_type'][t]['recall@5']:.3f}" for t in types)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test", "all"])
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    parser.add_argument(
        "--reranker", action="append", help="Cross-encoder model(s) for hybrid_rerank (repeatable)"
    )
    parser.add_argument("--k-candidates", type=int, nargs="+", default=[20])
    parser.add_argument("--k-per-paper", type=int, default=8)
    parser.add_argument("--min-per-paper", type=int, default=1)
    parser.add_argument("--tag", default="", help="Suffix for the results file name")
    args = parser.parse_args()

    questions = [q for q in load_questions(args.split) if q.evidence]
    store = get_store()
    rerankers = args.reranker or ["Xenova/ms-marco-MiniLM-L-6-v2"]

    configs = []
    for strategy in args.strategies:
        common = dict(
            k_final=K_EVAL, k_per_paper=args.k_per_paper, min_per_paper=args.min_per_paper
        )
        if strategy != "hybrid_rerank":
            configs.append((strategy, ScopedRetriever(store, strategy=strategy, **common)))
            continue
        for model in rerankers:
            reranker = CrossEncoderReranker(model)
            for kc in args.k_candidates:
                name = f"hybrid_rerank[{model.split('/')[-1]}, k_cand={kc}]"
                configs.append(
                    (
                        name,
                        ScopedRetriever(
                            store, strategy=strategy, reranker=reranker, k_candidates=kc, **common
                        ),
                    )
                )

    results = []
    for name, retriever in configs:
        print(f"Evaluating {name} on {len(questions)} questions ...", flush=True)
        results.append(evaluate(name, retriever, questions))
        r = results[-1]
        print(
            f"  R@5={r['recall@5']:.3f} R@10={r['recall@10']:.3f} MRR={r['mrr']:.3f} "
            f"latency={r['latency_s']}s",
            flush=True,
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = RESULTS / f"retrieval_{args.split}_{stamp}{'_' + args.tag if args.tag else ''}"
    base.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = markdown(results, args.split)
    base.with_suffix(".md").write_text(table, encoding="utf-8")
    print("\n" + table + f"\nSaved {base.with_suffix('.md').name} and .json")


if __name__ == "__main__":
    main()
