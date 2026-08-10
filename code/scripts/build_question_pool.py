"""Question pool v2: deduplicating ToolHop and screening out computational hops.

Motivation (audit of 2026-07-20). Canon v1 took 170 instances x ALL eligible
hops and produced 276 tool_wrong episodes from 139 unique question texts:
ToolHop's sub-questions are reused across chains, so "300 hops" is not "300
items". On top of that 74 of 276 (27%) were the template "What is the first
name of X?", where the gold is printed in the question itself: that is not a
memory-versus-tool conflict but a conflict with visible context, and it
accounts for 43-45% of the arbitration cell.

What is built here is a POOL OF QUESTIONS (not of episodes): each row is a
unique question text, exactly once, with the canonical (instance_id, hop_idx) =
its minimal occurrence. Distractor injection and episode assembly are a
separate step.

Screening: ToolHop is two worlds — facts ("Who is the father of X?") and
deterministic operations over the previous answer ("What is the date 10 days
before April 1, 1949?", "alphabetical order of the letters in Seldom"). The
second kind is invalid as an instrument (there is no parametric memory there)
but useful as a TRAINING signal for DPO/SFT — so it is not discarded, it is
written to a separate file.

The classifier is a pre-filter ahead of exhaustive manual annotation, not the
final truth: both the accepted and the screened-out rows are exported (with the
reason), so annotators can catch both junk inside the pool and false screening.

Run (deterministic, no GPU):
  python -m scripts.build_question_pool
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from memtoc.data import load_toolhop, iter_hops
from memtoc.inject_nonentity import parse_date
from memtoc.typing import TypingContext

ROOT = Path(__file__).resolve().parent.parent

# --- screening out computational / string hops ------------------------------
# Every pattern is named: the name goes into drop_reason and into the audit
# export.
_NUMW = r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)"
_ORD = r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)"

OP_PATTERNS: list[tuple[str, str]] = [
    ("name_from_string", r"\b(first|last|middle) name of\b"),
    ("letter_count", r"letters|number of (characters|words|vowels|consonants|digits|syllables)"),
    ("letter_pick", rf"\b{_ORD} (letter|vowel|consonant|digit|character|word)\b"),
    ("letter_stats", r"\bvowels?\b|\bconsonants?\b|syllabl|combinations"),
    ("string_sort", r"alphabetical order"),
    ("string_op", r"\breverse[sd]?\b|uppercase|lowercase|capitaliz|concatenat|substring|character at|without spaces"
                  r"|(longest|shortest) word"),
    # An unfilled template of ToolHop itself: "... of the place
    # #organization?".
    # Catch # followed by a lowercase word; "#32" in "jersey number to #32" is
    # valid text.
    ("placeholder", r"#[a-z]{3,}"),
    ("initials", r"\binitials of\b|\babbreviation of\b"),
    ("date_arith", rf"\bdate {_NUMW} |\byear {_NUMW} |year (following|preceding|before|after)|what is the year of \d"),
    ("date_part", r"^what is the (month|day|year|digit)\b.*\d|\(in number\)|^what is the \w+ digit|\bday of the week\b"),
    ("timezone", r"time difference|\bUTC\b|\btime zone\b|\bAoE\b|submission deadline|\bin (seconds|minutes|hours)\b"),
    ("arithmetic", r"result of (multiplying|adding|substracting|subtracting|dividing)|\bsum of\b|\bproduct of\b"
                   r"|\b(cube|square|power) of\b|prime factors|\bmodulo\b|square root|factorial|\bby how much\b"
                   r"|closest palindrome"),
    ("counting", r"\bhow many\b"),
    ("compare", r"difference between|^what is the (difference|ratio|average|median)"),
    ("encoding", r"binary code|\bASCII\b|hexadecimal|roman numeral|convert"),
    ("length", r"\blength of\b|rounded to"),
    ("numeric_stem", r"^what is \d"),
]
_OP_RE = [(name, re.compile(pat, re.I)) for name, pat in OP_PATTERNS]

# Knowledge templates that the op patterns would otherwise swallow. "What is
# the
# time zone of Encino?" -> "America/Los_Angeles" is a fact about a place, not
# arithmetic on UTC; the screening audit of 2026-07-20 returned 19 rows this
# way.
KNOWLEDGE_WHITELIST = [re.compile(r"^what is the time zone of ", re.I)]


def normalize(text: str) -> str:
    """Dedup key: case, punctuation and repeated spaces do not distinguish questions."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001 — provenance must not bring the build
                       # down
        return "unknown"


def drop_reason(question: str) -> str | None:
    """Name of the first op pattern that fired, or None for a knowledge question."""
    for rx in KNOWLEDGE_WHITELIST:
        if rx.search(question):
            return None
    for name, rx in _OP_RE:
        if rx.search(question):
            return name
    return None


def gold_kind(gold: str, tctx: TypingContext) -> str:
    """Gold type for distractor injection: entity by type map, otherwise by format.
    
    Order matters: year is checked before general number, otherwise "1914"
    falls through to number and gets a numeric distractor instead of a year.
    """
    g = gold.strip()
    if re.fullmatch(r"(1[0-9]|20)\d{2}", g):
        return "year"
    # The "date" type is decided by THE SAME parser that later builds the
    # distractor:
    # a regex over month names desynchronises from the injector and strays into
    # people's dates ("Christian August, Duke of..." — August is a real word
    # here,
    # and \\b boundaries do not help). If it parsed, it is a date.
    parsed = parse_date(g)
    if parsed is not None and parsed[0] is not None:
        return "date"
    if re.fullmatch(r"-?\d+(\.\d+)?", g):
        return "number"
    if re.fullmatch(r"[A-Za-z_]+/[A-Za-z_/]+", g):  # IANA timezone
        return "timezone"
    # type_of_key normalises on its own (memtoc.typing.normalize preserves case
    # and
    # punctuation); our dedup normalize must not be substituted here — the key
    # would miss.
    tau = tctx.type_of_key(g)
    return tau if tau in ("person", "place", "organization", "work") else "other_string"


def question_family(question: str) -> str:
    """Coarse template family — for controlling diversity during composition."""
    q = question.strip().lower()
    for name, pat in [
        ("genealogy", r"^who is the (father|mother|child|spouse|husband|wife|sibling|son|daughter)"),
        ("birth_death_date", r"^what is the date of (birth|death)"),
        ("birth_death_place", r"^what is the place of (birth|death)"),
        ("creator_role", r"^who is the (director|author|publisher|creator|composer|producer|cast member)"),
    ]:
        if re.search(pat, q):
            return name
    return "free_form"


def subject_of(question: str) -> str | None:
    """Subject entity of templated "... of X?" questions (a cluster by topic)."""
    m = re.search(r"\bof (?:the )?([A-Z][^?]*?)\s*\??$", question)
    return normalize(m.group(1)) if m else None


STOP_TOKENS = set(
    "the a an of in on at is are was were and or to for with which who what when "
    "where by from as that his her its their".split())


def gold_echo_ratio(question: str, gold: str) -> float:
    """Share of the gold's meaningful tokens already present in the question.
    
    A high value means TWO different defects, and only a human can tell them
    apart: (a) broken ToolHop gold (an actress was asked for, the gold is a
    film title); (b) a dynastic question "the father of the 6th Earl of X" ->
    "the 5th Earl of X", where the gold is correct but is guessable from the
    template rather than from memory. So this is measurement only; the decision
    belongs to the annotator.
    """
    gt = set(normalize(gold).split()) - STOP_TOKENS
    if not gt:
        return 0.0
    qt = set(normalize(question).split()) - STOP_TOKENS
    return len(gt & qt) / len(gt)


_ORDINAL_RE = re.compile(r"\b(\d+(st|nd|rd|th)|[IVXL]+)\b", re.I)


def dynasty_like(question: str, gold: str) -> bool:
    """Title-and-ordinal question: an ordinal in both the question and the gold."""
    return bool(_ORDINAL_RE.search(question) and _ORDINAL_RE.search(gold))


def subject_linked_clusters(rows: list[dict]) -> dict[str, int]:
    """Connected components of "one question's gold is another's subject".
    
    Genealogical chains (father of X -> Y, mother of Y -> Z) are not
    duplicates, but neither are they independent items: knowing the family
    solves them in a batch. For a cluster bootstrap that is one cluster, not N
    observations.
    """
    by_subject = defaultdict(list)
    for r in rows:
        if r.get("subject"):
            by_subject[r["subject"]].append(r["qkey"])
    parent = {r["qkey"]: r["qkey"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        gold_key = normalize(r["gold_answer"])
        for other in by_subject.get(gold_key, []):
            a, b = find(r["qkey"]), find(other)
            if a != b:
                parent[b] = a
    sizes = Counter(find(k) for k in parent)
    return {k: sizes[find(k)] for k in parent}


def near_duplicate_groups(rows: list[dict], threshold: float = 0.90) -> dict[str, list[str]]:
    """Near-identical texts WITH MATCHING gold -> a group id per row.
    
    The condition on the gold is essential: without it SequenceMatcher glues
    whole template families together ("In which administrative territorial
    entity is X located?" with different X and different answers) — those are
    not duplicates but a normal family of questions. The full square over ~600
    rows is cheap; blocking by length only makes it faster.
    """
    keys = [r["qkey"] for r in rows]
    gold_of = {r["qkey"]: normalize(r["gold_answer"]) for r in rows}
    parent = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if abs(len(a) - len(b)) > 25 or gold_of[a] != gold_of[b]:
                continue
            if difflib.SequenceMatcher(None, a, b).ratio() >= threshold:
                union(a, b)
    groups = defaultdict(list)
    for k in keys:
        groups[find(k)].append(k)
    return {k: v for k, v in groups.items() if len(v) > 1}


def build(toolhop_path: Path, type_map_path: Path) -> tuple[dict, dict]:
    data = load_toolhop(str(toolhop_path))
    tctx = TypingContext.load(str(type_map_path), sanitize_pool=True)

    # Dedup: the canonical occurrence is the minimal (instance_id, hop_idx);
    # the
    # other occurrences are kept, they are needed to reuse answers already
    # computed.
    seen: dict[str, dict] = {}
    for inst in data:
        hops = iter_hops(inst)
        for pos, hop in enumerate(hops):
            key = normalize(hop.question)
            row = seen.get(key)
            if row is None:
                seen[key] = {
                    "qkey": key,
                    "question": hop.question,
                    "gold_answer": hop.gold_answer,
                    "instance_id": inst["id"],
                    "hop_idx": hop.hop_idx,
                    "is_final_hop": pos == len(hops) - 1,
                    "domain": inst.get("domain", ""),
                    "occurrences": [[inst["id"], hop.hop_idx]],
                }
                continue
            row["occurrences"].append([inst["id"], hop.hop_idx])
            # the gold of one text must agree — otherwise the question is
            # ambiguous
            if normalize(row["gold_answer"]) != normalize(hop.gold_answer):
                row.setdefault("gold_conflict", []).append(hop.gold_answer)

    kept, dropped = [], []
    for row in seen.values():
        reason = drop_reason(row["question"])
        if reason:
            dropped.append({**row, "drop_reason": reason})
            continue
        row["gold_kind"] = gold_kind(row["gold_answer"], tctx)
        row["family"] = question_family(row["question"])
        row["subject"] = subject_of(row["question"])
        row["n_occurrences"] = len(row["occurrences"])
        kept.append(row)

    # Gold inside the question itself — the same defect as the name template,
    # but
    # caught on substance rather than on wording.
    for row in kept:
        g = normalize(row["gold_answer"])
        row["gold_in_question"] = bool(g) and g in row["qkey"]
        row["gold_echo"] = round(gold_echo_ratio(row["question"], row["gold_answer"]), 2)
        row["dynasty_like"] = dynasty_like(row["question"], row["gold_answer"])

    ndg = near_duplicate_groups(kept)
    member = {k: gid for gid, ks in ndg.items() for k in ks}
    for row in kept:
        row["near_dup_group"] = member.get(row["qkey"])

    subj = Counter(r["subject"] for r in kept if r["subject"])
    linked = subject_linked_clusters(kept)
    for row in kept:
        row["subject_cluster_size"] = subj.get(row["subject"], 0) if row["subject"] else 0
        row["knowledge_cluster_size"] = linked.get(row["qkey"], 1)

    kept.sort(key=lambda r: (r["instance_id"], r["hop_idx"]))
    summary = {
        "pool_version": "v2-questions",
        "built_by": "scripts/build_question_pool.py",
        "code_commit": code_commit(),
        "toolhop_sha256": file_sha256(toolhop_path),
        "type_map": str(type_map_path.relative_to(ROOT)),
        "type_map_sha256": file_sha256(type_map_path),
        "n_subquestions_total": sum(len(iter_hops(i)) for i in data),
        "n_unique_texts": len(seen),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "drop_reasons": dict(Counter(d["drop_reason"] for d in dropped).most_common()),
        "gold_kind": dict(Counter(r["gold_kind"] for r in kept).most_common()),
        "family": dict(Counter(r["family"] for r in kept).most_common()),
        "n_domains": len(set(r["domain"] for r in kept)),
        "n_instances_covered": len(set(r["instance_id"] for r in kept)),
        "n_gold_in_question": sum(1 for r in kept if r["gold_in_question"]),
        "n_gold_echo_ge_060": sum(1 for r in kept if r["gold_echo"] >= 0.6),
        "n_dynasty_like": sum(1 for r in kept if r["dynasty_like"]),
        "n_in_knowledge_cluster": sum(1 for r in kept if r["knowledge_cluster_size"] > 1),
        "max_knowledge_cluster": max((r["knowledge_cluster_size"] for r in kept), default=0),
        "n_gold_conflict": sum(1 for r in kept if r.get("gold_conflict")),
        "n_near_dup_groups": len(ndg),
        "n_in_near_dup_group": sum(1 for r in kept if r["near_dup_group"]),
        "n_final_hop": sum(1 for r in kept if r["is_final_hop"]),
    }
    return {"summary": summary, "questions": kept}, {"summary": summary, "dropped": dropped}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolhop", default=str(ROOT / "data" / "ToolHop.json"))
    ap.add_argument("--type-map", default=str(ROOT / "data" / "entity_types_v2.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "question_pool.json"))
    ap.add_argument("--out-dropped", default=str(ROOT / "data" / "question_pool_dropped.json"))
    args = ap.parse_args()

    pool, dropped = build(Path(args.toolhop), Path(args.type_map))
    Path(args.out).write_text(json.dumps(pool, ensure_ascii=False, indent=1))
    Path(args.out_dropped).write_text(json.dumps(dropped, ensure_ascii=False, indent=1))
    print(json.dumps(pool["summary"], ensure_ascii=False, indent=1))
    print("pool:", args.out, "sha256", file_sha256(Path(args.out))[:12])
    print("dropped:", args.out_dropped, "sha256", file_sha256(Path(args.out_dropped))[:12])


if __name__ == "__main__":
    main()
