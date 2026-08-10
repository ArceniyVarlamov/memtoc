"""Distractor-pool sanitiser — "pool v2" (step 1 of the plausibility ladder).

Why: human QC on v1 catalogued junk in the pool that no real tool could ever
return: truncated fragments ("of", "Duke", "College", "al-Dawla"), reversed
strings ("anhsirkamar" <- Ramakrishna), sorted scrambles ("aeeehknnrst" —
letters in alphabetical order; ToolHop's fictitious values), timezones
("Europe/Paris"). An offline measurement showed that implausible substitutions
push the model's answers into neither-noise and shift the headline arbitration
metrics, so the pool is cleaned BEFORE episodes are generated. This is the
mechanical step 1 (high precision, low recall); step 2 — a substitution that
fits by construction (the same tool's output on a neighbouring input) — is
separate work.

Discipline: only MEMBERSHIP IN THE CANDIDATE POOL is filtered (what may be
injected as a tool_wrong substitution). The typing of targets (type_of_key of
the golds) is untouched — a hop with a junk gold stays typeable and is
excluded by other machinery (QC broken-hop), not by this.

Rules (each returns a machine-checkable reason):
- len2      — the normalised value is shorter than 3 characters (one- and
              two-letter stubs: "A", "mn", "EC", "of").
- slash     — contains "/" (timezones America/Chicago, paths) — not an entity.
- scramble  — lowercase only, length >= 5, letters sorted non-decreasing
              ("aahnor", "aeeehknnrst") — an artefact of ToolHop's fictitious
              values; no word looks like that.
- reversed  — a value with no capitals whose reversal (of the whole string OR
              of the letters only — this catches hyphens and spaces:
              "ryc-tnias" <- Saint-Cyr, "erdnaxela-siuol" <- Louis-Alexandre)
              matches another pool value case-insensitively: "anhsirkamar"
              while "Ramakrishna" is present. The reversal is removed and the
              original stays.
- stub      — a curated list of fragments FROM THE QC CATALOGUE (verbatim what
              the annotators found; extend only through new QC, never by eye):
              Duke, College, al-Dawla (+ of/the — function words; "of" is also
              caught by len2, and the duplication is deliberate).
"""

from __future__ import annotations

import re
from collections import defaultdict

from .typing import normalize

# Verbatim from the QC junk catalogue (human QC v1); lowercase.
QC_STUBS = frozenset({"of", "the", "duke", "college", "al-dawla"})

_LOWER_ALPHA = re.compile(r"[a-z]+")
_ANY_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def _letters_lower(s: str) -> str:
    return "".join(ch for ch in s.lower() if _ANY_LETTER.match(ch))


def junk_reason(value: str, pool_lower: frozenset[str] | set[str],
                pool_letters: frozenset[str] | set[str] | None = None) -> str | None:
    """Reason a value is screened out of the candidate pool, or None (it is fine).
    
    pool_lower — every pool value (of every type) in lowercase;
    pool_letters — the same values reduced to letters (for reversals with
    hyphens and spaces). The reversed rule looks across the whole pool: a pair
    can live in different types (the typer labels a scramble independently of
    its original).
    """
    s = normalize(value)
    if len(s) <= 2:
        return "len2"
    if "/" in s:
        return "slash"
    if s.lower() in QC_STUBS:
        return "stub"
    if s == s.lower() and _ANY_LETTER.search(s):  # no capitals — names are not written that
                                                  # way
        if _LOWER_ALPHA.fullmatch(s) and len(s) >= 5 and list(s) == sorted(s):
            return "scramble"
        rev = s[::-1]
        if len(s) >= 3 and rev != s and rev in pool_lower:
            return "reversed"
        letters = _letters_lower(s)
        lrev = letters[::-1]
        if pool_letters is not None and len(letters) >= 3 and lrev != letters \
                and lrev in pool_letters:
            return "reversed"
    return None


def pool_junk_keys(pools: dict[str, list[dict]]):
    """The set of screened-out pool keys plus a report (build_typed_pools format).
    
    Returns (junk: frozenset[str], report), report = {
      "pool_version": "v2-sanitized",
      "n_before"/"n_after": int,
      "removed": {type: {reason: [values...]}},   # the full list, for audit
      "removed_counts": {type: {reason: n}},
    }. Deterministic (no RNG). The pools themselves are NOT filtered: the
    generator (inject.pick_typed_distractor) applies a rejection redraw — the
    first draw is over the full pool, so hops with a clean substitution keep it
    bit-for-bit and only the episodes that drew junk are redrawn (the smallest
    possible re-run).
    """
    pool_lower = frozenset(
        normalize(it["value"]).lower() for items in pools.values() for it in items
    )
    pool_letters = frozenset(
        _letters_lower(normalize(it["value"]))
        for items in pools.values() for it in items
    )
    junk: set[str] = set()
    removed: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    n_before = 0
    for typ, items in pools.items():
        for it in items:
            n_before += 1
            reason = junk_reason(it["value"], pool_lower, pool_letters)
            if reason is not None:
                junk.add(it["value"])
                removed[typ][reason].append(it["value"])
    report = {
        "pool_version": "v2-sanitized",
        "n_before": n_before,
        "n_after": n_before - sum(len(vs) for d in removed.values() for vs in d.values()),
        "removed": {t: {r: sorted(vs) for r, vs in d.items()} for t, d in removed.items()},
        "removed_counts": {t: {r: len(vs) for r, vs in d.items()} for t, d in removed.items()},
    }
    return frozenset(junk), report


def sanitize_typed_pools(pools: dict[str, list[dict]]):
    """Filtered pools plus the report — for audit and reporting scripts.
    
    The episode generator does NOT use this path (see pool_junk_keys: a
    rejection redraw instead of filtering, so clean substitutions do not move).
    """
    junk, report = pool_junk_keys(pools)
    clean = {t: [it for it in items if it["value"] not in junk]
             for t, items in pools.items()}
    return clean, report
