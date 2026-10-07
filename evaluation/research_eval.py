"""Research-assistant evaluation.

    uv run --project backend python -m evaluation.research_eval

There is no ground truth for "the right research gap", so the output is evaluated on what CAN be
checked:

  Structure (computed by code)
    completeness      share of answers with all three sections (evidence, analysis, directions)
    grounding         share of analysis/direction items linked to >= 1 evidence item, and of
                      evidence items with >= 1 valid page citation (1.0 by construction: the
                      pipeline removes items that fail; how many it removed is reported as
                      "dropped", i.e. how often the model invented sources or ungrounded claims)
  Evidence faithfulness (LLM judge, a different model family from the generator)
                      share of evidence claims supported by the excerpts they cite. Only the
                      evidence section is judged: analysis and directions are inference by design.
  Manual rubric       a sheet to grade specificity, grounding and plausibility (1-5) by hand.

Pipeline outputs and judge verdicts are cached in evaluation/results/research_cache/.
"""

import json
import statistics
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from evaluation.common import DATASETS, RESULTS, paper_ids
from researchhelp.config import get_settings
from researchhelp.rag.common.llm import get_chat_model
from researchhelp.rag.common.structured import invoke_json, json_mode
from researchhelp.rag.research.chain import build_research_chain
from researchhelp.retrieval.factory import get_retriever

CACHE = RESULTS / "research_cache"
JUDGE_MAX_TOKENS = 600

JUDGE_PROMPT = """EXCERPTS (full text, each shown once):
{excerpts}

CLAIMS (each lists the excerpts it cites):
{claims}

For each numbered CLAIM, decide whether it is directly supported by the excerpts it cites. \
Supported means the excerpts state it or it follows unambiguously from them; a claim that adds \
interpretation or a conclusion the excerpts do not state is NOT supported.

Return JSON only: {{"supported": [true, false, ...]}} with one boolean per claim, in order."""


class Verdicts(BaseModel):
    supported: list[bool]


def load_questions() -> list[dict]:
    ids = paper_ids()
    lines = (DATASETS / "research_questions.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    for r in rows:
        r["paper_ids"] = [ids[p] for p in r["papers"]]
    return rows


def cached(name: str, compute, cache_dir=None):
    cache_dir = cache_dir or CACHE
    path = cache_dir / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return value


def judge_evidence(judge, answer: dict) -> list[bool]:
    # Every cited chunk is shown once and in full (chunks are <= 1000 chars). Truncating per claim
    # hid facts near the end of a chunk and produced false "unsupported" verdicts.
    chunks = {s["chunk_id"]: s["text"] for s in answer["sources"]}
    cited = list(dict.fromkeys(c["chunk_id"] for e in answer["evidence"] for c in e["citations"]))
    label = {cid: f"X{i}" for i, cid in enumerate(cited, 1)}
    excerpts = "\n\n".join(f"[{label[cid]}] {chunks.get(cid, '')}" for cid in cited)
    claims = "\n".join(
        f"CLAIM {i}: {e['claim']} (cites {', '.join(label[c['chunk_id']] for c in e['citations'])})"
        for i, e in enumerate(answer["evidence"], 1)
    )
    verdicts = invoke_json(
        judge,
        [
            SystemMessage("You are a strict fact-checker. Respond with JSON only."),
            HumanMessage(JUDGE_PROMPT.format(excerpts=excerpts, claims=claims)),
        ],
        Verdicts,
    )
    return verdicts.supported[: len(answer["evidence"])]


def rubric_sheet(rows: list[dict]) -> str:
    lines = [
        "# Research-assistant rubric (grade by hand)",
        "Score each answer 1-5 on: **Specificity** (about these papers, not generic), "
        "**Grounding** (analysis really follows from the cited evidence), **Plausibility** "
        "(directions are sensible, testable research hypotheses).\n",
    ]
    for r in rows:
        a = r["answer"]
        lines += [
            f"## {r['id']}: {r['question']}",
            f"Papers: {', '.join(r['papers'])}\n",
            "**Evidence**",
        ]
        lines += [
            f"- {e['id']}. {e['claim']} ({e['paper_title']}, "
            f"p.{', '.join(str(c['page']) for c in e['citations'])})"
            for e in a["evidence"]
        ]
        lines += ["", "**Analysis (inferred)**"]
        lines += [
            f"- {x['statement']} [based on {', '.join(x['based_on'])}; {x['confidence']}]"
            for x in a["analysis"]
        ]
        lines += ["", "**Proposed directions (hypotheses)**"]
        lines += [
            f"- **{d['title']}**: {d['rationale']} *Test:* {d['validation_experiment']} "
            f"[based on {', '.join(d['based_on'])}]"
            for d in a["directions"]
        ]
        lines += ["", "Specificity: __  Grounding: __  Plausibility: __\n"]
    return "\n".join(lines)


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE,
        help="Where answers/verdicts are cached (re-judge old answers by "
        "pointing here and deleting their *_judge.json files)",
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    settings = get_settings()
    chain = build_research_chain(
        get_retriever(), get_chat_model(), settings.research_aux_query, settings.research_k_final
    )
    judge = json_mode(get_chat_model(model=settings.judge_model, temperature=0.0)).bind(
        max_tokens=JUDGE_MAX_TOKENS
    )

    rows = []
    for q in load_questions():
        print(f"{q['id']}: {q['question']}", flush=True)
        answer = cached(
            f"{q['id']}_answer",
            lambda q=q: chain.invoke(
                {"question": q["question"], "paper_ids": q["paper_ids"]}
            ).to_dict(),
            args.cache_dir,
        )
        supported = (
            cached(f"{q['id']}_judge", lambda a=answer: judge_evidence(judge, a), args.cache_dir)
            if answer["evidence"]
            else []
        )
        rows.append({**q, "answer": answer, "evidence_supported": supported})

    def share(values):
        values = list(values)
        return round(sum(values) / len(values), 3) if values else None

    answers = [r["answer"] for r in rows]
    inferred = [x for a in answers for x in a["analysis"] + a["directions"]]
    dropped = {
        k: sum(a.get("dropped", {}).get(k, 0) for a in answers)
        for k in ("evidence", "analysis", "directions")
    }
    summary = {
        "n_questions": len(rows),
        "status": {
            s: sum(a["status"] == s for a in answers) for s in ("ok", "no_evidence", "unstructured")
        },
        "completeness": share(
            bool(a["evidence"] and a["analysis"] and a["directions"]) for a in answers
        ),
        "mean_items": {
            k: round(statistics.mean(len(a[k]) for a in answers), 1)
            for k in ("evidence", "analysis", "directions")
        },
        "papers_covered": share(
            len({e["citations"][0]["paper_id"] for e in r["answer"]["evidence"]})
            == len(r["paper_ids"])
            for r in rows
        ),
        "inferred_items_linked": share(bool(x["based_on"]) for x in inferred),
        "evidence_with_citation": share(
            bool(e["citations"]) for a in answers for e in a["evidence"]
        ),
        "dropped_by_validation": dropped,
        "evidence_faithfulness": share(v for r in rows for v in r["evidence_supported"]),
        "n_evidence_claims": sum(len(a["evidence"]) for a in answers),
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    base = RESULTS / f"research_{stamp}{suffix}"
    base.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md = [
        f"## Research-assistant evaluation ({len(rows)} questions, generator "
        f"{settings.llm_model}, judge {settings.judge_model})",
        "",
        "| Metric | Value |",
        "|---|--:|",
    ]
    md += [f"| {k} | {v} |" for k, v in summary.items()]
    base.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RESULTS / f"research_rubric_{stamp}{suffix}.md").write_text(
        rubric_sheet(rows), encoding="utf-8"
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
