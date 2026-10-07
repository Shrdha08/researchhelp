"""Router evaluation: accuracy and confusion matrix of the intent classifier.

    uv run --project backend python -m evaluation.router_eval

Compares the LLM router (ROUTER_MODEL) with the keyword-only router used as its fallback.
LLM decisions are cached in evaluation/results/router_cache.json.
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime

from evaluation.common import DATASETS, RESULTS
from researchhelp.config import get_settings
from researchhelp.graph.routing import ROUTER_SYSTEM, classify_intent, keyword_intent
from researchhelp.rag.common.llm import get_chat_model

LABELS = ("evidence", "research")
CACHE = RESULTS / "router_cache.json"


def load_queries() -> list[dict]:
    lines = (DATASETS / "router_queries.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def score(rows: list[dict], key: str) -> dict:
    confusion = Counter((r["intent"], r[key]) for r in rows)
    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r["intent"] == r[key])
    return {
        "accuracy": round(sum(r["intent"] == r[key] for r in rows) / len(rows), 3),
        "confusion": {
            f"{gold}->{pred}": confusion[(gold, pred)] for gold in LABELS for pred in LABELS
        },
        "by_tag": {t: round(sum(v) / len(v), 3) for t, v in sorted(by_tag.items())},
        "errors": [r["id"] for r in rows if r["intent"] != r[key]],
    }


def markdown(rows: list[dict], results: dict, model: str) -> str:
    lines = [
        f"## Router evaluation ({len(rows)} labelled queries, LLM router: {model})\n",
        "| Router | Accuracy | evidence->evidence | evidence->research | research->evidence "
        "| research->research |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for name, r in results.items():
        c = r["confusion"]
        lines.append(
            f"| {name} | {r['accuracy']:.3f} | {c['evidence->evidence']} | "
            f"{c['evidence->research']} | {c['research->evidence']} | {c['research->research']} |"
        )
    tags = list(next(iter(results.values()))["by_tag"])
    lines += [
        "",
        "Accuracy by query tag:",
        "",
        "| Router | " + " | ".join(tags) + " |",
        "|---|" + "--:|" * len(tags),
    ]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r['by_tag'][t]:.2f}" for t in tags) + " |")
    lines += ["", "Misrouted queries:", ""]
    by_id = {r["id"]: r for r in rows}
    for name, r in results.items():
        for qid in r["errors"]:
            row = by_id[qid]
            pred = row["llm" if name.startswith("LLM") else "keyword"]
            lines.append(f'- {name}: `{qid}` "{row["query"]}" (gold {row["intent"]}, got {pred})')
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = get_settings()
    llm = get_chat_model(model=settings.router_model)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    rows = []
    for q in load_queries():
        key = f"{settings.router_model}::{q['query']}"
        if key not in cache:
            intent, reason, source = classify_intent(llm, q["query"])
            cache[key] = {"intent": intent, "reason": reason, "source": source}
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        llm_out = cache[key]
        rows.append(
            {
                **q,
                "llm": llm_out["intent"],
                "llm_source": llm_out["source"],
                "llm_reason": llm_out["reason"],
                "keyword": keyword_intent(q["query"])[0],
            }
        )

    fallbacks = sum(r["llm_source"] != "llm" for r in rows)
    results = {
        "LLM router": score(rows, "llm"),
        "Keyword router (baseline)": score(rows, "keyword"),
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = RESULTS / f"router_{stamp}"
    base.with_suffix(".json").write_text(
        json.dumps(
            {"results": results, "llm_fallbacks": fallbacks, "rows": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    md = markdown(rows, results, settings.router_model)
    md += f"\nLLM calls that fell back to keywords (invalid output or API error): {fallbacks}\n"
    # Queries that appear verbatim as few-shot examples in the router prompt inflate accuracy;
    # report the score on the remaining (unseen) queries as well.
    in_prompt = {q.lower() for q in re.findall(r'- "(.+?)" ->', ROUTER_SYSTEM)}
    unseen = [r for r in rows if r["query"].lower() not in in_prompt]
    md += (
        f"\nOn the {len(unseen)} queries that are not few-shot examples in the router prompt: "
        f"LLM router {score(unseen, 'llm')['accuracy']:.3f}, "
        f"keyword router {score(unseen, 'keyword')['accuracy']:.3f}.\n"
    )
    base.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
