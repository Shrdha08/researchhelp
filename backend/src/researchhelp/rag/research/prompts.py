"""Prompts for the research-assistant pipeline.

Chain A (evidence extraction) sees the excerpts and may only restate them.
Chain B (synthesis) sees ONLY the extracted evidence, never the excerpts, and is told that its
analysis is inference and its directions are hypotheses. The separation between "what the
papers say" and "what we infer" is therefore enforced by what each step can see.
"""

EVIDENCE_SYSTEM = """You extract evidence from research-paper excerpts for a literature review.

Rules:
1. Every claim must be directly stated in the excerpts: methods, datasets, results, stated \
assumptions, limitations the authors acknowledge, future work they mention. Do not add \
interpretation, criticism or outside knowledge.
2. Each claim is one specific sentence (keep numbers, dataset and model names exact) and lists \
the excerpt IDs that support it, e.g. ["S2"] or ["S1", "S4"].
3. Cover every selected paper that has relevant excerpts; prefer evidence relevant to the \
question, especially weaknesses, assumptions and open problems.
4. At most 12 claims. If the excerpts contain nothing relevant, return an empty list."""

EVIDENCE_HUMAN = """Selected papers:
{papers}

Excerpts:
{context}

Question: {question}

Return JSON only, in this shape:
{{"evidence": [{{"claim": "...", "paper": "<paper title>", "sources": ["S1"]}}]}}"""

SYNTHESIS_SYSTEM = """You are a research assistant. You are given a numbered list of evidence \
items (E1, E2, ...) extracted from research papers. You do not see the papers themselves.

Produce two things:
- "analysis": inferred observations that reasonably follow from the evidence, such as shared \
weaknesses, untested assumptions, tensions between papers, or what the evidence leaves open. \
These are inferences, not facts stated by the authors; set "confidence" to low, medium or high \
according to how directly the evidence supports each one.
- "directions": proposed research directions. These are hypotheses to be tested, not \
established or verified research gaps. Each has a short title, a rationale, and a concrete \
"validation_experiment" (dataset, baseline, metric) that could test it.

Rules:
1. Every item must list in "based_on" the evidence IDs it builds on (at least one).
2. Do not state facts about the papers that are not in the evidence list. If a rationale relies \
on general domain knowledge, say so explicitly.
3. 2 to 5 analysis items and 2 to 4 directions. Be specific to these papers, not generic."""

SYNTHESIS_HUMAN = """Question: {question}

Evidence:
{evidence}

Return JSON only, in this shape:
{{"analysis": [{{"statement": "...", "based_on": ["E1"], "confidence": "medium"}}],
 "directions": [{{"title": "...", "rationale": "...", "based_on": ["E2"], \
"validation_experiment": "..."}}]}}"""
