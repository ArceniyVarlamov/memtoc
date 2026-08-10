"""Apply the 39 adjudicated distractor repairs and rebuild all benchmark arms.

Reads the frozen v2 mapping and `results/quality_control/repair_list_39.json`, writes the
v3 mapping, rebuilds canonical pv0/pv1/pv2 and all derived arms into fresh
directories, verifies them with the existing independent verifiers, asserts
that exactly the 39 repaired tool_wrong rows differ from v2 (and nothing
else), and emits per-arm patch files that contain only the changed episodes.

The v2 files and every response already collected on them are left untouched;
patch runs are overlaid at analysis time by response key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CANON_ARMS = [f"episodes_question_v2_near_pv{pv}.json" for pv in (0, 1, 2)]
DERIVED_ARMS = [
    f"{kind}_{name}_pv{pv}.json"
    for kind, names in (("presentation", ("toolns", "ragsnip")), ("control", ("warn", "prior", "flag")))
    for name in names
    for pv in (0, 1, 2)
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    print("[repair-v3] run:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def norm(ep: dict, drop_payload: bool) -> str:
    clone = json.loads(json.dumps(ep, ensure_ascii=False))
    clone["provenance"]["distractor_source"] = None
    clone["provenance"]["distractor_method"] = None
    if drop_payload:
        clone["tool_output"] = None
        clone.pop("distractor", None)
        clone["prompts"]["with_tool"] = None
    return json.dumps(clone, ensure_ascii=False, sort_keys=True)


def diff_arm(old_path: Path, new_path: Path, repaired_ids: set[tuple[int, int]]) -> dict:
    old = {e["episode_id"]: e for e in json.loads(old_path.read_text())["episodes"]}
    new = {e["episode_id"]: e for e in json.loads(new_path.read_text())["episodes"]}
    assert old.keys() == new.keys(), f"episode id sets differ: {new_path.name}"
    material, metadata_only = [], []
    for eid, e_new in new.items():
        e_old = old[eid]
        if json.dumps(e_old, sort_keys=True) == json.dumps(e_new, sort_keys=True):
            continue
        key = (e_new["instance_id"], e_new["hop_idx"])
        assert key in repaired_ids, f"unexpected change outside repair set: {eid}"
        if norm(e_old, drop_payload=False) == norm(e_new, drop_payload=False):
            # provenance relabel only; prompt and payload are byte-identical,
            # existing v2 responses stay valid for this episode
            metadata_only.append(eid)
            continue
        assert e_new["condition"] == "tool_wrong", f"non-tool_wrong material change: {eid}"
        assert norm(e_old, drop_payload=True) == norm(e_new, drop_payload=True), (
            f"change leaks outside payload/prompt/provenance fields: {eid}"
        )
        material.append(eid)
    return {
        "n_changed": len(material),
        "n_metadata_only": len(metadata_only),
        "changed_ids": sorted(material),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-v2", type=Path, default=ROOT / "results/construction/distractor_mapping_575.json")
    parser.add_argument("--repairs", type=Path, default=ROOT / "results/quality_control/repair_list_39.json")
    parser.add_argument("--mapping-v3", type=Path, default=ROOT / "results/quality_control/frozen_distractors_v3_575.json")
    parser.add_argument("--toolhop", type=Path, default=ROOT / "data/ToolHop.json")
    parser.add_argument("--v2-dir", type=Path, default=ROOT / "data_e095_575")
    parser.add_argument("--v2-derived-dir", type=Path, default=ROOT / "data_e096_derived")
    parser.add_argument("--v3-dir", type=Path, default=ROOT / "data_e096_v3")
    parser.add_argument("--v3-stage-dir", type=Path, default=ROOT / "episodes_e096_v3_stage")
    parser.add_argument("--v3-derived-dir", type=Path, default=ROOT / "data_e096_v3_derived")
    parser.add_argument("--patch-dir", type=Path, default=ROOT / "data_e096_patch")
    parser.add_argument("--report", type=Path, default=ROOT / "results/quality_control/repair_v3_verification.json")
    args = parser.parse_args()

    python = sys.executable

    mapping = json.loads(args.mapping_v2.read_text())
    repairs_doc = json.loads(args.repairs.read_text())
    repairs = {r["qkey"]: r for r in repairs_doc["repairs"]}
    assert len(repairs) == 39, len(repairs)

    applied = 0
    repaired_ids: set[tuple[int, int]] = set()
    for row in mapping["rows"]:
        rep = repairs.get(row["qkey"])
        if rep is None:
            continue
        assert row["final_near"] == rep["old_near"], f"old distractor mismatch: {row['qkey']}"
        assert rep["new_near"] != row["gold_answer"], f"repair equals gold: {row['qkey']}"
        assert (row["instance_id"], row["hop_idx"]) == (rep["instance_id"], rep["hop_idx"])
        row["final_near"] = rep["new_near"]
        row["source"] = "qc_adjudication_repair"
        row["method"] = "assistant_draft_human_approved"
        row["validation_status"] = "PASS"
        row["validation_note"] = f"repair ({rep['audit_id']}): {rep['repair_reason']}"
        repaired_ids.add((row["instance_id"], row["hop_idx"]))
        applied += 1
    assert applied == 39, applied

    mapping["schema_version"] = "memtoc-frozen-distractors-v3"
    mapping["correction"] = {
        "row": "memtoc-v1",
        "supersedes": "results/construction/distractor_mapping_575.json (v2, sha256 "
        + file_sha256(args.mapping_v2) + ")",
        "applied": "39 distractor repairs from the human semantic-QC adjudication",
        "repairs_input": {
            "path": str(args.repairs.relative_to(ROOT)),
            "sha256": file_sha256(args.repairs),
        },
        "journal_row": "memtoc-repair-round",
        "unchanged": "question, gold_answer and the 536 non-repaired distractors are byte-identical to v2; the 33 EXCLUDE questions remain in the file and are dropped at analysis per the QC freeze",
    }
    canon = json.dumps(
        [
            {k: row[k] for k in ("qkey", "question", "gold_answer", "final_near")}
            for row in sorted(mapping["rows"], key=lambda r: r["qkey"])
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    mapping["content_sha256_575"] = hashlib.sha256(canon.encode()).hexdigest()
    args.mapping_v3.parent.mkdir(parents=True, exist_ok=True)
    args.mapping_v3.write_text(json.dumps(mapping, ensure_ascii=False, indent=1) + "\n")
    print(f"[repair-v3] mapping v3 written: {args.mapping_v3} sha256={file_sha256(args.mapping_v3)}")

    run([python, "-m", "scripts.build_episodes",
         "--mapping", str(args.mapping_v3), "--out-dir", str(args.v3_dir),
         "--stage-dir", str(args.v3_stage_dir)])
    run([python, "-m", "scripts.verify_episodes",
         "--mapping", str(args.mapping_v3), "--data-dir", str(args.v3_dir)])
    run([python, "-m", "scripts.build_derived_arms",
         "--episodes-dir", str(args.v3_dir), "--out-dir", str(args.v3_derived_dir)])
    run([python, "-m", "scripts.verify_e096_derived_arms",
         "--episodes-dir", str(args.v3_dir), "--derived-dir", str(args.v3_derived_dir),
         "--report", str(ROOT / "results/quality_control/derived_arms_verification.json")])

    report = {
        "schema_version": "memtoc-repair-v3-verification",
        "mapping_v2_sha256": file_sha256(args.mapping_v2),
        "mapping_v3_sha256": file_sha256(args.mapping_v3),
        "repairs_sha256": file_sha256(args.repairs),
        "arms": {},
    }
    args.patch_dir.mkdir(parents=True, exist_ok=True)
    pairs = [(args.v2_dir / n, args.v3_dir / n, f"canonical_pv{i}") for i, n in enumerate(CANON_ARMS)]
    pairs += [(args.v2_derived_dir / n, args.v3_derived_dir / n, n.removesuffix(".json")) for n in DERIVED_ARMS]
    for old_path, new_path, tag in pairs:
        d = diff_arm(old_path, new_path, repaired_ids)
        assert d["n_changed"] == 39, f"{tag}: expected 39 changed, got {d['n_changed']}"
        doc = json.loads(new_path.read_text())
        changed = set(d["changed_ids"])
        patch_doc = {
            "summary": {
                **doc["summary"],
                "patch_of": new_path.name,
                "patch_kind": "memtoc-repair-39",
                "n_episodes": 39,
            },
            "episodes": [e for e in doc["episodes"] if e["episode_id"] in changed],
        }
        assert len(patch_doc["episodes"]) == 39
        patch_path = args.patch_dir / f"patch_{tag}.json"
        patch_path.write_text(json.dumps(patch_doc, ensure_ascii=False, indent=1) + "\n")
        report["arms"][tag] = {
            "n_changed": d["n_changed"],
            "n_metadata_only": d["n_metadata_only"],
            "v3_sha256": file_sha256(new_path),
            "patch_path": str(patch_path.relative_to(ROOT)),
            "patch_sha256": file_sha256(patch_path),
        }
        print(f"[repair-v3] {tag}: 39 changed -> {patch_path.name}")

    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(f"[repair-v3] report: {args.report} sha256={file_sha256(args.report)}")
    print("[repair-v3] PASS")


if __name__ == "__main__":
    main()
