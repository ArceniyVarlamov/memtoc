"""Freeze the near-distractor mapping from two independent annotations.

The two XLSX files are first imported with artifact-tool into JSON snapshots
(`scripts/e095_annotation_workbook.mjs`). This script performs the research
merge without any spreadsheet dependency and fails closed on count, identity,
gold-collision, alternative-correct-answer, or unresolved-index errors.

The merge does *not* treat disagreement as consensus. It preserves both raw
answers and records the deterministic adjudication rule used for every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ANNOTATOR_A = ROOT / "results/construction/annotation/authoring_annotator_a.json"
ANNOTATOR_B = ROOT / "results/construction/annotation/authoring_annotator_b.json"
MACHINE = ROOT / "results/construction/distractors_v2_wikidata.json"
GOLD_VERIFICATION = ROOT / "results/construction/gold_verification_wikidata.json"
ADJUDICATION_OUT = ROOT / "results/construction/annotation/authoring_adjudication.json"
FROZEN_OUT = ROOT / "results/construction/frozen_distractors_v2.json"


REPAIRS = {
    "2.1": {
        "final_near": "David Bowers",
        "reason": "the machine candidate Bill Tarmey does not match the role",
        "evidence": "earlier spot-check: David Bowers was marked suitable by a human",
    },
    "24.0": {
        "final_near": "Dylan McDermott",
        "reason": "Michael Loceff is not a plausible acting answer",
        "evidence": "earlier spot-check: Dylan McDermott was marked suitable by a human",
    },
    "46.0": {
        "final_near": "The Children Act (film)",
        "reason": "We Live in Public is a documentary of a different class",
        "evidence": "same period and a British adaptation of Ian McEwan; the director is not Dominic Cooke",
    },
    "63.0": {
        "final_near": "AFC East",
        "reason": "Southwest Division belongs to a different sporting system",
        "evidence": "a real division of the same NFL/AFC, but the Cleveland Browns are not in it",
    },
    "268.0": {
        "final_near": "Susan Seidelman",
        "reason": "Sheena Iyengar is not a film director",
        "evidence": "an American woman director born in 1952, like Melanie Mayron",
    },
    "294.0": {
        "final_near": "Prince Ranieri, Duke of Castro",
        "reason": "Adolphe Pegoud has no connection to the house of Bourbon-Two Sicilies",
        "evidence": "same dynasty and title; this is the grandfather of Princess Beatrice, not her father",
    },
    "297.0": {
        "final_near": "Fíachu Finnolach",
        "reason": "Dan Neville is a contemporary politician, not a figure of Irish tradition",
        "evidence": "another legendary High King of Ireland, not the father of Dui Finn",
    },
}


RULES = [
    {
        "code": "independent_agreement",
        "when": "A and B gave the same value once the candidate number was resolved",
        "choice": "shared value",
        "rationale": "direct independent agreement",
    },
    {
        "code": "custom_over_candidate",
        "when": "one answer is the annotator's own, the other was picked from the machine short list",
        "choice": "own answer",
        "rationale": "the short list is only a coarse Wikidata filter; manual context is more precise",
    },
    {
        "code": "lower_ranked_candidate",
        "when": "A and B picked different items from the short list",
        "choice": "the candidate with the lower number",
        "rationale": "the short list is pre-sorted from less to more famous",
    },
    {
        "code": "deterministic_a_tie_break",
        "when": "A and B gave different answers of their own",
        "choice": "answer A",
        "rationale": "a deterministic tie-break with no post-hoc selection; the disagreement is preserved",
    },
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def item_id(row: dict) -> str:
    return f'{row["instance_id"]}.{row["hop_idx"]}'


def candidates(text: object) -> list[str]:
    out: list[str] = []
    for line in str(text or "").splitlines():
        match = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", line)
        if match:
            assert int(match.group(1)) == len(out) + 1, f"broken candidate numbering: {text!r}"
            out.append(match.group(2))
    return out


def resolve_annotation(row: dict) -> dict:
    raw = row["near (no. or your own)"]
    shortlist = candidates(row["Wikidata candidates (pick no.)"])
    raw_text = str(raw).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", raw_text):
        index = int(float(raw_text))
        assert 1 <= index <= len(shortlist), f'{row["id"]}: candidate index {index} out of range'
        return {"raw": raw, "resolved": shortlist[index - 1], "kind": "candidate", "candidate_index": index}
    assert raw_text, f'{row["id"]}: empty near'
    return {"raw": raw, "resolved": raw_text, "kind": "custom", "candidate_index": None}


def choose(a: dict, b: dict) -> tuple[str, str, bool]:
    if norm(a["resolved"]) == norm(b["resolved"]):
        return a["resolved"], "independent_agreement", True
    if a["kind"] != b["kind"]:
        winner = a if a["kind"] == "custom" else b
        return winner["resolved"], "custom_over_candidate", False
    if a["kind"] == "candidate":
        winner = a if a["candidate_index"] < b["candidate_index"] else b
        return winner["resolved"], "lower_ranked_candidate", False
    return a["resolved"], "deterministic_a_tie_break", False


def known_values_by_question(gold: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in gold["results"]:
        values = {norm(v) for v in row.get("wikidata_values", []) if norm(v)}
        if values:
            out[norm(row["question"])] = values
    return out


def validate_near(near: str, gold_answer: str, far: str | None, known_values: set[str]) -> dict:
    problems = []
    near_n = norm(near)
    if not near_n:
        problems.append("empty")
    if near_n == norm(gold_answer):
        problems.append("equals_gold")
    if far and near_n == norm(far):
        problems.append("equals_far")
    if near_n in known_values:
        problems.append("known_property_value")
    return {"status": "PASS" if not problems else "FAIL", "note": ", ".join(problems)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, default=ANNOTATOR_A)
    parser.add_argument("--annotator-b", type=Path, default=ANNOTATOR_B)
    parser.add_argument("--machine", type=Path, default=MACHINE)
    parser.add_argument("--gold-verification", type=Path, default=GOLD_VERIFICATION)
    parser.add_argument("--adjudication-out", type=Path, default=ADJUDICATION_OUT)
    parser.add_argument("--frozen-out", type=Path, default=FROZEN_OUT)
    args = parser.parse_args()

    source_a = json.loads(args.annotator_a.read_text())
    source_b = json.loads(args.annotator_b.read_text())
    machine = json.loads(args.machine.read_text())
    gold = json.loads(args.gold_verification.read_text())
    known = known_values_by_question(gold)

    rows_a = {str(row["id"]): row for row in source_a["rows"]}
    rows_b = {str(row["id"]): row for row in source_b["rows"]}
    assert len(rows_a) == len(rows_b) == 152
    assert set(rows_a) == set(rows_b)

    machine_by_id = {item_id(row): row for row in machine["results"].values()}
    assert len(machine_by_id) == 605

    manual_adjudication = []
    final_manual: dict[str, str] = {}
    for iid in rows_a:
        ra, rb = rows_a[iid], rows_b[iid]
        for field in ("Question", "Gold (correct)", "Type", "Family", "Why manual",
                      "Wikidata candidates (pick no.)"):
            assert ra[field] == rb[field], f"{iid}: source mismatch in {field}"
        assert iid in machine_by_id, f"{iid}: missing from machine pool"
        base = machine_by_id[iid]
        assert norm(ra["Question"]) == norm(base["question"])
        assert norm(ra["Gold (correct)"]) == norm(base["gold_answer"])
        aa, bb = resolve_annotation(ra), resolve_annotation(rb)
        final_near, rule, agreement = choose(aa, bb)
        validation = validate_near(
            final_near, base["gold_answer"], base.get("far"), known.get(norm(base["question"]), set())
        )
        manual_adjudication.append({
            "id": iid,
            "instance_id": base["instance_id"],
            "hop_idx": base["hop_idx"],
            "question": base["question"],
            "gold_answer": base["gold_answer"],
            "gold_kind": base["gold_kind"],
            "family": base["family"],
            "reason": ra["Why manual"],
            "annotator_a": aa,
            "annotator_b": bb,
            "agreement": agreement,
            "decision_rule": rule,
            "final_near": final_near,
            "validation": validation,
        })
        final_manual[iid] = final_near

    repair_rows = []
    for iid, repair in REPAIRS.items():
        assert iid in machine_by_id and iid not in final_manual
        base = machine_by_id[iid]
        validation = validate_near(
            repair["final_near"], base["gold_answer"], base.get("far"),
            known.get(norm(base["question"]), set()),
        )
        assert validation["status"] == "PASS", f"repair {iid}: {validation}"
        repair_rows.append({
            "id": iid,
            "question": base["question"],
            "gold_answer": base["gold_answer"],
            "old_near": base["near"],
            **repair,
            "validation": validation,
        })

    final_rows, excluded_rows = [], []
    for base in machine["results"].values():
        iid = item_id(base)
        if base.get("dropped"):
            excluded_rows.append({
                "id": iid,
                "instance_id": base["instance_id"],
                "hop_idx": base["hop_idx"],
                "question": base["question"],
                "gold_answer": base["gold_answer"],
                "reason": base.get("reason", "broken_gold"),
            })
            continue
        if iid in final_manual:
            final_near = final_manual[iid]
            source, method = "dual_human_adjudication", "human_authorized"
        elif iid in REPAIRS:
            final_near = REPAIRS[iid]["final_near"]
            source, method = "manual_spotcheck_repair", "machine_repaired"
        else:
            final_near = base.get("near")
            source, method = "wikidata_or_deterministic_generator", base["method"]
        assert isinstance(final_near, str) and final_near.strip(), f"{iid}: no final near"
        validation = validate_near(
            final_near, base["gold_answer"], base.get("far"), known.get(norm(base["question"]), set())
        )
        final_rows.append({
            "id": iid,
            "instance_id": base["instance_id"],
            "hop_idx": base["hop_idx"],
            "qkey": base["qkey"],
            "question": base["question"],
            "gold_answer": base["gold_answer"],
            "gold_kind": base["gold_kind"],
            "family": base["family"],
            "final_near": final_near,
            "far": base.get("far"),
            "source": source,
            "method": method,
            "validation_status": validation["status"],
            "validation_note": validation["note"],
            "machine_provenance": base.get("provenance", {}),
        })

    decision_counts = Counter(row["decision_rule"] for row in manual_adjudication)
    invalid = [row for row in final_rows if row["validation_status"] != "PASS"]
    assert len(final_rows) == 587
    assert len(excluded_rows) == 18
    assert len([r for r in final_rows if r["source"] == "dual_human_adjudication"]) == 152
    assert len([r for r in final_rows if r["source"] == "manual_spotcheck_repair"]) == 7
    assert not invalid, f"invalid final near values: {[(r['id'], r['validation_note']) for r in invalid]}"

    sources = {
        "annotator_a": {
            "path": source_a["source_path"],
            "sha256": source_a["source_sha256"],
            "json_snapshot": str(args.annotator_a.relative_to(ROOT)),
        },
        "annotator_b": {
            "path": source_b["source_path"],
            "sha256": source_b["source_sha256"],
            "json_snapshot": str(args.annotator_b.relative_to(ROOT)),
        },
        "machine": {"path": str(args.machine.relative_to(ROOT)), "sha256": file_sha256(args.machine)},
        "gold_verification": {
            "path": str(args.gold_verification.relative_to(ROOT)),
            "sha256": file_sha256(args.gold_verification),
        },
    }
    summary = {
        "manual_rows": 152,
        "manual_agreements": sum(row["agreement"] for row in manual_adjudication),
        "manual_disagreements": sum(not row["agreement"] for row in manual_adjudication),
        "manual_decision_rules": dict(decision_counts),
        "machine_rows": 435,
        "machine_repairs": len(REPAIRS),
        "final_rows": len(final_rows),
        "excluded_broken_gold": len(excluded_rows),
        "invalid_final_near": len(invalid),
    }
    adjudication = {
        "schema_version": "memtoc-authoring-adjudication-v1",
        "status": "frozen",
        "built_by": "scripts/merge_distractor_authoring.py",
        "built_on": date.today().isoformat(),
        "code_commit": code_commit(),
        "builder_sha256": file_sha256(Path(__file__)),
        "sources": sources,
        "summary": summary,
        "adjudication_rules": RULES,
        "manual_adjudication": manual_adjudication,
        "machine_repairs": repair_rows,
        "final_rows": final_rows,
        "excluded_rows": excluded_rows,
    }
    frozen = {
        "schema_version": "memtoc-frozen-distractors-v2",
        "status": "frozen",
        "built_by": "scripts/merge_distractor_authoring.py",
        "built_on": date.today().isoformat(),
        "code_commit": code_commit(),
        "builder_sha256": file_sha256(Path(__file__)),
        "sources": sources,
        "summary": summary,
        "rows": final_rows,
        "excluded_rows": excluded_rows,
    }
    args.adjudication_out.parent.mkdir(parents=True, exist_ok=True)
    args.frozen_out.parent.mkdir(parents=True, exist_ok=True)
    args.adjudication_out.write_text(json.dumps(adjudication, ensure_ascii=False, indent=1) + "\n")
    args.frozen_out.write_text(json.dumps(frozen, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("->", args.adjudication_out)
    print("->", args.frozen_out)


if __name__ == "__main__":
    main()
