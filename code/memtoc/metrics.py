"""MemToC v0 metrics — the proposal definitions: TFR / KRR / CRA.

Every metric is computed from the results of a run (answers: episode_id ->
the model's final answer; parametric: episode_id -> its own closed-book
answer). v0 matching is normalised containment; known limitation: entity
aliases — tightened in v1.
"""

from __future__ import annotations

import random
import re
import unicodedata


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def matches(answer: str, target: str) -> bool:
    a, t = normalize(answer), normalize(target)
    return bool(t) and (t in a or a == t)


def episode_outcome(ep: dict, final_answer: str, parametric_answer: str) -> dict:
    """Decomposition of one episode: which side the model followed, and is it right."""
    tool_out = ep.get("tool_output") or {}
    tool_value = tool_out.get("result") if isinstance(tool_out, dict) else None
    followed_tool = tool_value is not None and matches(final_answer, tool_value)
    kept_memory = bool(parametric_answer) and matches(final_answer, parametric_answer)
    return {
        "episode_id": ep["episode_id"],
        "condition": ep["condition"],
        "tool_correct": ep["tool_correct"],
        "memory_correct": (
            matches(parametric_answer, ep["gold_answer"])
            if parametric_answer else None
        ),
        "followed_tool": followed_tool,
        "kept_memory": kept_memory,
        "final_correct": matches(final_answer, ep["gold_answer"]),
    }


def aggregate(outcomes: list[dict]) -> dict:
    """TFR/KRR/CRA per condition, plus the oracle band.
    
    conflict episode: the tool and memory answers diverge (kept_memory and
    followed_tool cannot both be true on one answer when distractor !=
    a_param; we filter on the actual divergence).
    Oracle CRA: the share of episodes where AT LEAST ONE source is right —
    the upper bound on arbitration.
    """
    by_cond: dict[str, dict] = {}
    for cond in sorted({o["condition"] for o in outcomes}):
        sub = [o for o in outcomes if o["condition"] == cond]
        n = len(sub)
        conflict = [
            o for o in sub
            if o["memory_correct"] is not None
            and o["tool_correct"] is not None
            and (o["tool_correct"] != o["memory_correct"])
        ]
        nc = len(conflict)
        by_cond[cond] = {
            "n": n,
            "n_conflict": nc,
            "TFR": sum(o["followed_tool"] for o in conflict) / nc if nc else None,
            "KRR": sum(o["kept_memory"] for o in conflict) / nc if nc else None,
            "CRA": sum(o["final_correct"] for o in conflict) / nc if nc else None,
            "acc_all": sum(o["final_correct"] for o in sub) / n if n else None,
            "oracle_CRA": (
                sum(
                    (o["tool_correct"] or o["memory_correct"]) for o in conflict
                ) / nc if nc else None
            ),
        }
    return by_cond


# =============================================================================
# v1 aggregates (§5): per-cell breakdown + CAR/source_prior/abstain
# + bootstrap CI. Expects outcomes from memtoc.scoring.score_episode (with the
# fields cell, ack_proxy, abstain, neither_subtype). The v0 aggregate above is
# untouched.
# =============================================================================

def _rate(items: list[dict], key: str):
    return sum(bool(o[key]) for o in items) / len(items) if items else None


def bootstrap_ci(items: list[dict], key: str, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0):
    """Percentile CI of the share (o[key]) by bootstrap over episodes."""
    if not items:
        return None
    vals = [1.0 if o[key] else 0.0 for o in items]
    n = len(vals)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return [round(lo, 4), round(hi, 4)]


# Desired behaviour per cell — for readability of the report (does not affect
# the computation).
_CELL_DESIRED = {
    "arb": "keep_memory",       # mem✓,tool✗ → hold memory
    "tool_gold": "follow_tool",  # mem✗,tool✓ → follow the tool
    "both_wrong": "abstain",     # mem✗,tool✗ → abstain (gold unreachable)
    "agree": "either",           # mem✓,tool✓ → not a conflict
    "mem_absent": "n/a",         # closed-book UNKNOWN: memory not elicited,
                                 # conflict undefined (§3, the non-compliance
                                 # filter)
}


def aggregate_v1(scored: list[dict]) -> dict:
    """Per-condition and per-cell v1 metrics.
    
    CRA is computed ONLY on the cells where gold is reachable from one of the
    sides (arb, tool_gold); both_wrong yields source_prior (=followed_tool)
    and abstain_rate, but NOT CRA (§5).
    mem_absent (closed-book UNKNOWN) is a separate cell outside the 2×2: it
    enters neither CRA_conflict nor source_prior (its followed_tool/CRA are
    visible in cells).
    CAR is a coarse proxy (ack_proxy) until the validated detector.
    """
    out: dict[str, dict] = {}
    conds = sorted({o["condition"] for o in scored})
    for cond in conds:
        sub = [o for o in scored if o["condition"] == cond]
        cells: dict[str, dict] = {}
        for cname in sorted({o["cell"] for o in sub if o["cell"]}):
            cl = [o for o in sub if o["cell"] == cname]
            cells[cname] = {
                "n": len(cl),
                "desired": _CELL_DESIRED.get(cname),
                "followed_tool": _rate(cl, "followed_tool"),
                "kept_memory": _rate(cl, "kept_memory"),
                "CRA": _rate(cl, "final_correct"),
                "abstain": _rate(cl, "abstain"),
                "CRA_ci95": bootstrap_ci(cl, "final_correct"),
            }
        # conflict cells (off-diagonal) — arb ∪ tool_gold
        gold_reach = [o for o in sub if o["cell"] in ("arb", "tool_gold")]
        out[cond] = {
            "n": len(sub),
            "cells": cells,
            "CRA_conflict": _rate(gold_reach, "final_correct"),
            "CRA_conflict_ci95": bootstrap_ci(gold_reach, "final_correct"),
            "CAR_proxy": _rate([o for o in sub if o["cell"] in
                                ("arb", "tool_gold", "both_wrong")], "ack_proxy"),
            "abstain_rate": _rate(sub, "abstain"),
            "source_prior_both_wrong": _rate(
                [o for o in sub if o["cell"] == "both_wrong"], "followed_tool"),
        }
    return out
