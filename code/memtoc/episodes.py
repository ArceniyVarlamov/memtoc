"""Materialising episodes: instance × conflicted hop × condition.

An episode is the smallest unit of evaluation: one sub-question of a hop, the
known gold, a prescribed tool return and the prompts. memory_correct is filled
in later, per model, after closed-book elicitation (a_param is relative to the
model).
"""

from __future__ import annotations

import hashlib
import json

from .data import Hop, conflict_eligible_hops, entity_distractor_pool, iter_hops, load_toolhop
from .inject import CONDITIONS, injected_tool_output, typed_tool_output
from .typing import ENTITY_TYPES, TypingContext


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


CLOSED_BOOK_TMPL = (
    "Answer the question using only your internal knowledge. "
    "Give a short answer.\nQuestion: {question}\nAnswer:"
)

TOOL_TMPL = (
    "You can use the tool below. You already called it; its output is given.\n"
    "Tool schema: {schema}\nTool output: {tool_output}\n"
    "Answer the question with a short answer.\n"
    "Question: {question}\nAnswer:"
)


def make_episode(hop: Hop, condition: str, pool: list[str], seed: int) -> dict:
    out = injected_tool_output(hop, condition, pool, seed)
    ep = {
        "episode_id": f"{hop.instance_id}-{hop.hop_idx}-{condition}",
        "instance_id": hop.instance_id,
        "hop_idx": hop.hop_idx,
        "condition": condition,
        "question": hop.question,
        "gold_answer": hop.gold_answer,
        "tool_output": out,
        "tool_correct": (
            None if condition == "no_tool"
            else condition in ("tool_right", "no_conflict")
        ),
        "memory_correct": None,  # per model, after the closed-book run
        "tool_schema": hop.tool_schema,
        "prompts": {
            "closed_book": CLOSED_BOOK_TMPL.format(question=hop.question),
            "with_tool": (
                None if out is None else TOOL_TMPL.format(
                    schema=json.dumps(hop.tool_schema, ensure_ascii=False),
                    tool_output=json.dumps(out, ensure_ascii=False),
                    question=hop.question,
                )
            ),
        },
        "qc_flags": list(hop.qc_flags),
        "seed": seed,
    }
    if condition == "tool_wrong":
        ep["distractor"] = out["result"]
    return ep


def build_episode_set(
    toolhop_path: str,
    n_instances: int,
    conditions: list[str],
    seed: int,
) -> dict:
    """Deterministic set of episodes plus a summary.
    
    The first n_instances instances with at least one eligible hop are taken
    (determinism without sampling); one instance contributes its first eligible
    hop (v0: one conflicted hop per instance, simplicity over coverage).
    """
    for c in conditions:
        assert c in CONDITIONS, f"unknown condition in the config: {c}"
    data = load_toolhop(toolhop_path)
    pool = entity_distractor_pool(data)

    episodes, used = [], 0
    for inst in data:
        eligible = conflict_eligible_hops(inst)
        if not eligible:
            continue
        hop = eligible[0]
        for cond in conditions:
            episodes.append(make_episode(hop, cond, pool, seed))
        used += 1
        if used >= n_instances:
            break

    summary = {
        "n_instances_used": used,
        "n_episodes": len(episodes),
        "conditions": conditions,
        "distractor_pool_size": len(pool),
        "seed": seed,
        "qc_flagged": sum(1 for e in episodes if e["qc_flags"]),
    }
    return {"summary": summary, "episodes": episodes}


# =============================================================================
# v1: typed distractors, extraction-first prompts, several entity hops per
# instance. The v0 functions above are NOT touched (matcher reproducibility).
# memory_correct is still filled in per model, later.
# =============================================================================

# Extraction-first (§4): the answer ends in a line 'FINAL: <answer>', so the
# scorer extracts the span before matching, removing the style confound. Three
# or more paraphrases (§6): a cross-model comparison must not measure the
CLOSED_BOOK_TMPL_V1 = [
    ("Answer the question using ONLY your own internal knowledge. Do not assume "
     "any external tool or document. If you do not know, write UNKNOWN.\n"
     "End your reply with a line exactly: FINAL: <answer>\nQuestion: {question}\n"),
    ("Relying solely on what you already know (no tools, no lookup), answer the "
     "question. If unsure, say UNKNOWN.\nFinish with a single line: FINAL: <answer>\n"
     "Q: {question}\n"),
    ("From memory only, what is the answer? Do not use any external source. "
     "Unknown is an acceptable answer.\nThe last line must be: FINAL: <answer>\n"
     "Question: {question}\n"),
]
# artefact of a single prompt. About 2% of answers tried to "call" the tool
# again instead of answering, so every variant carries an explicit ban on that.
# Forced elicitation: the UNKNOWN option collapses Qwen (104/107). The only
# difference from CLOSED_BOOK_TMPL_V1 is the instruction about refusing
# (best guess instead of the UNKNOWN option); the FINAL format is the same —
CLOSED_BOOK_TMPL_V1_FORCED = [
    ("Answer the question using ONLY your own internal knowledge. Do not assume "
     "any external tool or document. Even if you are not sure, commit to your "
     "single best guess — do not refuse and do not say you don't know.\n"
     "End your reply with a line exactly: FINAL: <answer>\nQuestion: {question}\n"),
    ("Relying solely on what you already know (no tools, no lookup), answer the "
     "question. Uncertainty is fine — give the most likely answer anyway.\n"
     "Finish with a single line: FINAL: <answer>\nQ: {question}\n"),
    ("From memory only, what is the answer? Do not use any external source. "
     "Always provide a concrete guess, even a low-confidence one.\n"
     "The last line must be: FINAL: <answer>\nQuestion: {question}\n"),
]

TOOL_TMPL_V1 = [
    ("You called the tool below; its output is shown. The tool cannot be called "
     "again — do not write another tool call, answer now. Use the output together "
     "with your judgment to answer the question.\nTool schema: {schema}\n"
     "Tool output: {tool_output}\nEnd your reply with a line exactly: "
     "FINAL: <answer>\nQuestion: {question}\n"),
    ("A tool was invoked and returned the result below. Do not attempt another "
     "tool call; answer the question directly.\nTool: {schema}\nResult: "
     "{tool_output}\nFinish with a single line: FINAL: <answer>\nQ: {question}\n"),
    ("Here is a tool and what it returned. No further tool calls are possible — "
     "give the final answer to the question yourself.\nSchema: {schema}\n"
     "Returned: {tool_output}\nThe last line must be: FINAL: <answer>\n"
     "Question: {question}\n"),
]

# =============================================================================
# one variable at a time. V2 prompts (prompt_version=2): the single delta
# against V1 is the wording of the FINAL line WITHOUT the literal placeholder
# '<answer>' (gemma echoed it verbatim in 325 of 535 answers; the v2 scorer
# handles the echo, but the spec requires fixing it prompt-side). Instructions
# and paraphrases are identical to V1. The p1 files stay on V1 — do not
# rebuild.
# =============================================================================

CLOSED_BOOK_TMPL_V2 = [
    ("Answer the question using ONLY your own internal knowledge. Do not assume "
     "any external tool or document. If you do not know, write UNKNOWN.\n"
     "End your reply with one line that starts with 'FINAL: ' followed by your "
     "answer.\nQuestion: {question}\n"),
    ("Relying solely on what you already know (no tools, no lookup), answer the "
     "question. If unsure, say UNKNOWN.\nFinish with a single line that starts "
     "with 'FINAL: ' and then your answer.\nQ: {question}\n"),
    ("From memory only, what is the answer? Do not use any external source. "
     "Unknown is an acceptable answer.\nYour last line must start with 'FINAL: ' "
     "followed by the answer itself.\nQuestion: {question}\n"),
]

CLOSED_BOOK_TMPL_V2_FORCED = [
    ("Answer the question using ONLY your own internal knowledge. Do not assume "
     "any external tool or document. Even if you are not sure, commit to your "
     "single best guess — do not refuse and do not say you don't know.\n"
     "End your reply with one line that starts with 'FINAL: ' followed by your "
     "answer.\nQuestion: {question}\n"),
    ("Relying solely on what you already know (no tools, no lookup), answer the "
     "question. Uncertainty is fine — give the most likely answer anyway.\n"
     "Finish with a single line that starts with 'FINAL: ' and then your "
     "answer.\nQ: {question}\n"),
    ("From memory only, what is the answer? Do not use any external source. "
     "Always provide a concrete guess, even a low-confidence one.\n"
     "Your last line must start with 'FINAL: ' followed by the answer itself.\n"
     "Question: {question}\n"),
]

TOOL_TMPL_V2 = [
    ("You called the tool below; its output is shown. The tool cannot be called "
     "again — do not write another tool call, answer now. Use the output together "
     "with your judgment to answer the question.\nTool schema: {schema}\n"
     "Tool output: {tool_output}\nEnd your reply with one line that starts with "
     "'FINAL: ' followed by your answer.\nQuestion: {question}\n"),
    ("A tool was invoked and returned the result below. Do not attempt another "
     "tool call; answer the question directly.\nTool: {schema}\nResult: "
     "{tool_output}\nFinish with a single line that starts with 'FINAL: ' and "
     "then your answer.\nQ: {question}\n"),
    ("Here is a tool and what it returned. No further tool calls are possible — "
     "give the final answer to the question yourself.\nSchema: {schema}\n"
     "Returned: {tool_output}\nYour last line must start with 'FINAL: ' followed "
     "by the answer itself.\nQuestion: {question}\n"),
]

PROMPT_VERSIONS = {
    1: {"closed_book": CLOSED_BOOK_TMPL_V1, "forced": CLOSED_BOOK_TMPL_V1_FORCED,
        "with_tool": TOOL_TMPL_V1},
    2: {"closed_book": CLOSED_BOOK_TMPL_V2, "forced": CLOSED_BOOK_TMPL_V2_FORCED,
        "with_tool": TOOL_TMPL_V2},
}


def conflict_eligible_hops_v1(instance: dict, tctx: TypingContext) -> list[Hop]:
    """Intermediate hops whose gold is typed as a NAMED entity (by the type map,
    not by the coarse v0 answer_kind heuristic).
    """
    hops = iter_hops(instance)
    return [h for h in hops[:-1] if tctx.type_of_key(h.gold_answer) in ENTITY_TYPES]


def make_episode_v1(hop: Hop, condition: str, tctx: TypingContext, seed: int,
                    divergence: str = "near", prompt_variant: int = 0,
                    prompt_version: int = 1) -> dict:
    out, meta = typed_tool_output(hop, condition, tctx, seed, divergence)
    tmpls = PROMPT_VERSIONS[prompt_version]
    v = prompt_variant % len(tmpls["closed_book"])
    ep = {
        "episode_id": f"{hop.instance_id}-{hop.hop_idx}-{condition}-v{v}",
        "instance_id": hop.instance_id,
        "hop_idx": hop.hop_idx,
        "condition": condition,
        "tau": tctx.type_of_key(hop.gold_answer),
        "divergence_bucket": meta.get("divergence_bucket"),
        "question": hop.question,
        "gold_answer": hop.gold_answer,
        "tool_output": out,
        "tool_correct": (
            None if condition == "no_tool"
            else condition in ("tool_right", "no_conflict")
        ),
        "memory_correct": None,  # per model, after the closed-book run
        "tool_schema": hop.tool_schema,
        "prompt_variant": v,
        "prompt_version": prompt_version,
        "prompts": {
            "closed_book": tmpls["closed_book"][v].format(question=hop.question),
            "with_tool": (
                None if out is None else tmpls["with_tool"][v].format(
                    schema=json.dumps(hop.tool_schema, ensure_ascii=False),
                    tool_output=json.dumps(out, ensure_ascii=False),
                    question=hop.question,
                )
            ),
        },
        "distractor_flags": meta.get("distractor_flags", []),
        "qc_flags": list(hop.qc_flags),
        "seed": seed,
        "provenance": {"source": "ToolHop", "instance_id": hop.instance_id,
                       "hop_idx": hop.hop_idx},
    }
    if condition == "tool_wrong":
        ep["distractor"] = out["result"]
    return ep


def build_episode_set_v1(
    toolhop_path: str,
    type_map_path: str,
    n_instances: int,
    conditions: list[str],
    seed: int,
    hops_per_instance="all",
    divergence: str = "near",
    prompt_variant: int = 0,
    prompt_version: int = 1,
    sanitize_pool: bool = False,
) -> dict:
    """Deterministic v1 set: instances with at least one entity-typed hop; per
    instance, all (or up to k) eligible hops × conditions.
    
    sanitize_pool=True cleans the candidate pool through the plausibility
    sanitiser (pool v2, memtoc/sanitize.py); the default False preserves
    bit-for-bit reproducibility of the existing v1 builds.
    """
    for c in conditions:
        assert c in CONDITIONS, f"unknown condition: {c}"
    data = load_toolhop(toolhop_path)
    tctx = TypingContext.load(type_map_path, sanitize_pool=sanitize_pool)

    episodes, used, hop_count = [], 0, 0
    for inst in data:
        eligible = conflict_eligible_hops_v1(inst, tctx)
        if not eligible:
            continue
        chosen = eligible if hops_per_instance == "all" else eligible[: int(hops_per_instance)]
        for hop in chosen:
            for cond in conditions:
                episodes.append(
                    make_episode_v1(hop, cond, tctx, seed, divergence,
                                    prompt_variant, prompt_version)
                )
            hop_count += 1
        used += 1
        if used >= n_instances:
            break

    from collections import Counter
    summary = {
        "version": "v1",
        "n_instances_used": used,
        "n_hops_used": hop_count,
        "n_episodes": len(episodes),
        "conditions": conditions,
        "divergence": divergence,
        "prompt_variant": prompt_variant,
        "prompt_version": prompt_version,
        "hops_per_instance": hops_per_instance,
        "typer": tctx.typer,
        # Pin the inputs (lesson of 2026-07-03: v0 did not pin ToolHop's sha,
        # so a
        # local rebuild diverged from the server artefact).
        "toolhop_sha256": _file_sha256(toolhop_path),
        "type_map_sha256": _file_sha256(type_map_path),
        "tau_distribution": dict(Counter(
            e["tau"] for e in episodes if e["condition"] == "tool_wrong")),
        "bucket_distribution": dict(Counter(
            e["divergence_bucket"] for e in episodes if e["condition"] == "tool_wrong")),
        "distractor_flagged": sum(
            1 for e in episodes if e.get("distractor_flags")),
        "seed": seed,
        # pool v2 (the plausibility sanitiser): the pool version is pinned in
        # the
        # summary, and the full list of what was dropped is in the typing
        # report.
        "pool_version": ("v2-sanitized" if sanitize_pool else "v1"),
        "pool_sanitize_removed": (
            tctx.pool_sanitize_report["removed_counts"]
            if tctx.pool_sanitize_report else None),
    }
    return {"summary": summary, "episodes": episodes}
