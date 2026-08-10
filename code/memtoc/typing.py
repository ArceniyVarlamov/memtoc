"""Typing of ToolHop entity golds, and typed distractor pools (v1).

Why: the v0 defect — the tool_wrong distractor was drawn from the GLOBAL
entity pool with no regard for the entity type, so 58% of substitutions did
not match the gold's type (a person's name replacing a park), which
understated absolute TFR_tw. v1 (§2): a substitution of the SAME type, plus a
divergence grading near / far / off_type.

Typing is DECOUPLED from the generator through the artefact
data/entity_types.json:
  - the production typer is a NER model: scripts/build_type_map.py;
  - the dependency-free fallback is HeuristicTyper (below): ToolHop's
    ground-truth previous_answer_type (high-precision seeds) plus structural
    rules, for checking the pipe locally without a model.
The generator (inject.py / episodes.py) consumes only the finished type_map —
which typer built it is of no concern to the generator. Empirically (checked
2026-07-03): previous_answer_type yields only person(498)/place(44)/org(28)
and does NOT map cleanly onto a specific hop (chains branch), so as the SOLE
source of type it is insufficient — hence NER over all hops.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

# Type vocabulary. Entity types are the targets of conflict; temporal/number
# are not (parametric conflict lives on named entities).
ENTITY_TYPES = ("person", "place", "organization", "work")
TEMPORAL_TYPES = ("date", "year")
OTHER = "other"
ALL_TYPES = ENTITY_TYPES + TEMPORAL_TYPES + (OTHER,)

# ToolHop's previous_answer_type → our vocabulary (ground-truth seeds).
PREV_TYPE_MAP = {
    "person": "person",
    "place": "place",
    "organization": "organization",
    "date": "date",
    "year": "year",
}

_NUM_OR_BRACKET = re.compile(r"^[\d,\s.\-\[\]()]+$")
_HAS_ALPHA = re.compile(r"[A-Za-z]")
_YEAR = re.compile(r"^\[?\s*-?\d{3,4}\s*\]?$")

# Structural rules (low recall, high precision — for the fallback).
_ORG_CUES = re.compile(
    r"\b(College|University|Company|Corporation|Inc|Ltd|Institute|Museum|"
    r"Academy|Society|Association|Club|Party|Commission|Department|Council|"
    r"School|Hospital|Church|Foundation|Orchestra|Studios?|Records|"
    r"Committee|Bureau|Agency|League|Union|Bank|Group)\b",
    re.IGNORECASE,
)
_PLACE_CUES = re.compile(
    r"\b(Park|Lake|River|Mountain|Mount|Sea|Ocean|Island|Bay|City|County|"
    r"Province|Kingdom|Republic|Castle|Palace|Bridge|Street|Road|Station|"
    r"Airport|Harbou?r|Valley|Forest|Gardens?|Hill|Beach|Cape|Fort)\b",
    re.IGNORECASE,
)


_WRAPS = {"[": "]", "(": ")"}


def _fully_wrapped(s: str) -> bool:
    """A leading bracket is closed by exactly the LAST character of the string."""
    if len(s) < 2 or s[0] not in _WRAPS or s[-1] != _WRAPS[s[0]]:
        return False
    op, cl, depth = s[0], _WRAPS[s[0]], 0
    for i, ch in enumerate(s):
        if ch == op:
            depth += 1
        elif ch == cl:
            depth -= 1
            if depth == 0:
                return i == len(s) - 1
    return False


def normalize(text: str) -> str:
    """Canonical form of a gold, for pool keys and matching.
    
    Only PAIRED wrappers spanning the whole string are stripped
    ("[foo]" -> "foo"); a bare strip("[]()") cut the trailing bracket off
    Wikipedia-style suffixes ("X (band)" -> "X (band") — 14 of 807 pool keys.
    """
    s = re.sub(r"\s+", " ", str(text).strip())
    while _fully_wrapped(s):
        s = s[1:-1].strip()
    return s


def is_entityish(text: str) -> bool:
    """The string looks like a named entity (not a bare number/list/bracket)."""
    s = normalize(text)
    if not s or len(s) < 2 or len(s) > 80:
        return False
    if _NUM_OR_BRACKET.match(s):
        return False
    return bool(_HAS_ALPHA.search(s))


def collect_entity_golds(data: list[dict]) -> dict[str, dict]:
    """Collect every entity-ish gold across all hops, with context.
    
    Returns normalized_gold -> {"raw": <first raw form>, "domains": set[str],
    "prev_votes": Counter}. prev_votes accumulates ground-truth type votes only
    where the gold matched the penultimate answer and the instance has a
    previous_answer_type in PREV_TYPE_MAP (a high-precision but incomplete
    signal).
    """
    out: dict[str, dict] = {}
    for inst in data:
        domain = str(inst.get("domain", "")).strip().lower()
        subs = list(inst.get("sub_task", {}).values())
        penult = normalize(subs[-2]) if len(subs) >= 2 else None
        pat = PREV_TYPE_MAP.get(str(inst.get("previous_answer_type")))
        for raw in subs:
            if not is_entityish(raw):
                continue
            key = normalize(raw)
            rec = out.setdefault(
                key, {"raw": str(raw), "domains": set(), "prev_votes": Counter()}
            )
            if domain:
                rec["domains"].add(domain)
            if pat and key == penult and pat in ("person", "place", "organization"):
                rec["prev_votes"][pat] += 1
    return out


def build_gazetteers(golds: dict[str, dict]) -> dict[str, set]:
    """High-precision gazetteers from the ground-truth prev_votes.
    
    The person gazetteer is additionally expanded into tokens (given names and
    surnames occur one word at a time: 'Thomas', 'Jennings').
    """
    gaz: dict[str, set] = {t: set() for t in ("person", "place", "organization")}
    person_tokens: set[str] = set()
    for key, rec in golds.items():
        if not rec["prev_votes"]:
            continue
        t = rec["prev_votes"].most_common(1)[0][0]
        gaz[t].add(key.lower())
        if t == "person":
            for tok in key.split():
                if len(tok) > 2 and tok[:1].isupper():
                    person_tokens.add(tok.lower())
    gaz["person_tokens"] = person_tokens
    return gaz


class HeuristicTyper:
    """Dependency-free typer (the fallback). Precision over recall; provisional.
    
    Order: explicit prior (previous_answer_type for a known hop) → gazetteer →
    structural rules → person by name tokens → other.
    Real typing at scale is the NER model (build_type_map.py).
    """

    name = "heuristic"

    def __init__(self, data: list[dict]):
        self._golds = collect_entity_golds(data)
        self._gaz = build_gazetteers(self._golds)

    def type_of(self, text: str, prior: str | None = None) -> str:
        s = normalize(text)
        if not is_entityish(s):
            if _YEAR.match(str(text).strip()):
                return "year"
            return OTHER
        low = s.lower()
        if prior in ENTITY_TYPES or prior in TEMPORAL_TYPES:
            return prior
        for t in ("organization", "place", "person"):
            if low in self._gaz[t]:
                return t
        if _ORG_CUES.search(s):
            return "organization"
        if _PLACE_CUES.search(s):
            return "place"
        toks = [t for t in s.split() if t[:1].isupper()]
        if toks and all(t.lower() in self._gaz["person_tokens"] for t in toks):
            return "person"
        if 1 <= len(s.split()) <= 3 and s[:1].isupper():
            # many golds are short proper names; conservatively person
            return "person"
        return OTHER

    def type_map(self) -> dict[str, str]:
        """Type every collected entity-ish gold."""
        return {key: self.type_of(rec["raw"]) for key, rec in self._golds.items()}


def build_typed_pools(
    type_map: dict[str, str], golds: dict[str, dict]
) -> dict[str, list[dict]]:
    """Distractor pools by type: type -> [{value, domains}]."""
    pools: dict[str, list[dict]] = defaultdict(list)
    for key, t in type_map.items():
        rec = golds.get(key, {})
        pools[t].append({"value": key, "domains": set(rec.get("domains", set()))})
    return pools


def divergence_bucket(
    target_key: str,
    target_type: str,
    target_domains: set,
    distractor_key: str,
    type_map: dict[str, str],
    golds: dict[str, dict],
) -> str:
    """near = same type and a shared domain; far = same type, no domain overlap;
    off_type = a different type. For the QC audit (scripts/qc_type_match.py).
    """
    dt = type_map.get(distractor_key, OTHER)
    if dt != target_type:
        return "off_type"
    d_domains = set(golds.get(distractor_key, {}).get("domains", set()))
    return "near" if (target_domains & d_domains) else "far"


class TypingContext:
    """The loaded typing artefact: type/domains by key, plus pools by type.
    
    A single entry point for the generator (inject.py/episodes.py) and for QC.
    The key is normalize().
    """

    def __init__(self, golds: dict[str, dict], typer: str = "?", meta: dict | None = None,
                 sanitize_pool: bool = False):
        # golds: key -> {"type","domains"(set),"raw"}
        self.golds = golds
        self.typer = typer
        self.meta = meta or {}
        self.type_map = {k: v["type"] for k, v in golds.items()}
        self.pools = build_typed_pools(self.type_map, golds)
        # pool v2: junk screened out against the human-QC catalogue
        # (memtoc/sanitize.py).
        # The pools are NOT filtered — the generator does a rejection redraw
        # against
        # the junk set, so clean substitutions stay bit-for-bit; type_of_key is
        # untouched.
        self.pool_junk = frozenset()
        self.pool_sanitize_report = None
        if sanitize_pool:
            from .sanitize import pool_junk_keys
            self.pool_junk, self.pool_sanitize_report = pool_junk_keys(self.pools)

    @classmethod
    def load(cls, path: str, sanitize_pool: bool = False) -> "TypingContext":
        with open(path, encoding="utf-8") as f:
            art = json.load(f)
        golds = {
            k: {"type": v["type"], "domains": set(v.get("domains", [])), "raw": v.get("raw", k)}
            for k, v in art["golds"].items()
        }
        return cls(golds, typer=art.get("typer", "?"), meta=art.get("meta", {}),
                   sanitize_pool=sanitize_pool)

    def type_of_key(self, key: str) -> str:
        return self.type_map.get(normalize(key), OTHER)

    def domains_of(self, key: str) -> set:
        return set(self.golds.get(normalize(key), {}).get("domains", set()))

    def bucket(self, target_key: str, distractor_key: str) -> str:
        tk = normalize(target_key)
        return divergence_bucket(
            tk, self.type_of_key(tk), self.domains_of(tk),
            normalize(distractor_key), self.type_map, self.golds,
        )
