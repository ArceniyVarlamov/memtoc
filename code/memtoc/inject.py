"""Conflict injection: what the tool returns on a conflicted hop.

The injection is protocol-level: the function's return is substituted at the
observation layer in the runner; the dataset's own functions are never edited.
Whether the tool is right is controlled by construction:
  tool_right  -> the tool returns gold (a conflict is possible only if the
                 model's memory is wrong);
  tool_wrong  -> the tool returns a plausible distractor of the same nature;
  tool_error  -> a broken API (a structured error);
  no_tool     -> control without a tool (closed-book);
  no_conflict -> control: the tool returns gold and the question has no catch
                 (identical to tool_right in v0, split out for clarity).
"""

from __future__ import annotations

import hashlib
import random

from .data import Hop
from .typing import ENTITY_TYPES, TypingContext, normalize

CONDITIONS = ("tool_right", "tool_wrong", "tool_error", "no_tool", "no_conflict")

TOOL_ERROR_PAYLOAD = {
    "error": "InternalServiceError",
    "message": "upstream database temporarily unavailable; cached value may be stale",
    "code": 503,
}


def _stable_rng(seed: int, *keys) -> random.Random:
    h = hashlib.sha256(("|".join(map(str, keys)) + f"|{seed}").encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def pick_distractor(hop: Hop, pool: list[str], seed: int) -> str:
    """Same-kind distractor for tool_wrong.
    
    v0: a random other entity from the global pool (deterministic in the seed
    and the identity of the hop). Known limitation: the entity type
    (person/place/org) is not matched — the case/length heuristic only screens
    out junk; tightened in v1 (flagged on the episode).
    """
    rng = _stable_rng(seed, hop.instance_id, hop.hop_idx, "distractor")
    gold = hop.gold_answer.strip().lower()
    candidates = [p for p in pool if p.strip().lower() != gold and 2 < len(p) < 80]
    assert candidates, "empty distractor pool"
    return rng.choice(candidates)


def injected_tool_output(hop: Hop, condition: str, pool: list[str], seed: int):
    """The tool return on a conflicted hop, for the given condition."""
    if condition in ("tool_right", "no_conflict"):
        return {"result": hop.gold_answer}
    if condition == "tool_wrong":
        return {"result": pick_distractor(hop, pool, seed)}
    if condition == "tool_error":
        return dict(TOOL_ERROR_PAYLOAD)
    if condition == "no_tool":
        return None
    raise ValueError(f"unknown condition: {condition}")


# --- v1: typed distractors (§2) ----------------------------------------------

def _bucket_candidates(tctx: TypingContext, ttype: str, tdomains: set,
                       bucket: str, gold_key: str) -> list[str]:
    """Candidate distractors for the requested divergence bucket."""
    goldl = gold_key.lower()
    def ok(v: str) -> bool:
        return v.lower() != goldl
    if bucket in ("near", "far"):
        out = []
        for item in tctx.pools.get(ttype, []):
            if not ok(item["value"]):
                continue
            shares = bool(tdomains & item["domains"])
            if (bucket == "near") == shares:
                out.append(item["value"])
        return out
    if bucket == "off_type":
        out = []
        for t in ENTITY_TYPES:
            if t == ttype:
                continue
            out += [it["value"] for it in tctx.pools.get(t, []) if ok(it["value"])]
        return out
    # "any" — anything of the same type
    return [it["value"] for it in tctx.pools.get(ttype, []) if ok(it["value"])]


# Fallback order for an empty bucket (e.g. a type seen exactly once).
_FALLBACK = {
    "near": ["near", "far", "any"],
    "far": ["far", "near", "any"],
    "off_type": ["off_type"],
    "any": ["any"],
}


def _choice_sanitized(rng, cands: list[str], junk, flags: list[str]) -> str | None:
    """Selection with a rejection redraw over pool v2 (memtoc/sanitize.py).
    
    The first draw is over the FULL pool: hops whose original choice is clean
    keep their substitution bit-for-bit with the v1 builds; a redraw happens
    only when junk comes up (QC flag pool_v2_redraw). None means the bucket is
    empty after screening (the outer loop moves to the next fallback bucket).
    """
    d = rng.choice(sorted(cands))
    if not junk or d not in junk:
        return d
    clean = [c for c in cands if c not in junk]
    if not clean:
        flags.append("pool_v2_bucket_emptied")
        return None
    flags.append("pool_v2_redraw")
    return rng.choice(sorted(clean))


def pick_typed_distractor(hop: Hop, tctx: TypingContext, seed: int,
                          divergence: str = "near") -> tuple[str, str, list[str]]:
    """Same-type distractor for tool_wrong (v1). Deterministic in the seed and hop.
    
    Returns (distractor, actual_bucket, flags). If the requested bucket is
    empty there is a soft fallback (near→far→any) with a QC flag; the entity
    type is always preserved except on the explicit off_type arm. With pool v2
    enabled (tctx.pool_junk) a junk choice is redrawn inside the bucket (see
    _choice_sanitized).
    """
    gold_key = normalize(hop.gold_answer)
    ttype = tctx.type_of_key(gold_key)
    tdomains = tctx.domains_of(gold_key)
    rng = _stable_rng(seed, hop.instance_id, hop.hop_idx, "typed", divergence)
    flags: list[str] = []
    junk = getattr(tctx, "pool_junk", frozenset())
    if ttype not in ENTITY_TYPES:
        flags.append(f"target_not_entity_typed:{ttype}")
    for b in _FALLBACK.get(divergence, ["any"]):
        cands = _bucket_candidates(tctx, ttype, tdomains, b, gold_key)
        if cands:
            d = _choice_sanitized(rng, cands, junk, flags)
            if d is None:
                continue  # the bucket emptied after screening — next fallback
            if b != divergence:
                flags.append(f"fallback_{divergence}_to_{b}")
            return d, b, flags
    # last resort: the type is empty altogether — take any entity of another
    # type
    flags.append("no_same_type_distractor")
    glob = _bucket_candidates(tctx, ttype, tdomains, "off_type", gold_key)
    assert glob, "empty global pool of typed distractors"
    d = _choice_sanitized(rng, glob, junk, flags)
    assert d is not None, "the global pool emptied after pool v2 screening"
    return d, "off_type", flags


def typed_tool_output(hop: Hop, condition: str, tctx: TypingContext, seed: int,
                      divergence: str = "near"):
    """v1 tool return plus typing metadata. Returns (output|None, meta)."""
    if condition in ("tool_right", "no_conflict"):
        return {"result": hop.gold_answer}, {}
    if condition == "tool_wrong":
        d, bucket, flags = pick_typed_distractor(hop, tctx, seed, divergence)
        meta = {"divergence_bucket": bucket, "distractor_flags": flags,
                "target_type": tctx.type_of_key(hop.gold_answer)}
        return {"result": d}, meta
    if condition == "tool_error":
        return dict(TOOL_ERROR_PAYLOAD), {}
    if condition == "no_tool":
        return None, {}
    raise ValueError(f"unknown condition: {condition}")
