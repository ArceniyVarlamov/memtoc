"""Independent verifier for the frozen split and the cross-fit inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FOLDS = ("A", "B")
CONDITIONS = ("tool_right", "tool_wrong", "tool_error", "no_tool", "no_conflict")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=ROOT / "results/construction/distractor_mapping_575.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "results/quality_control/crossfit_folds.json",
    )
    parser.add_argument("--episodes-dir", type=Path, default=ROOT / "data_e095_575")
    parser.add_argument("--crossfit-dir", type=Path, default=ROOT / "data_e096_crossfit")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/quality_control/crossfit_verification.json",
    )
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text())
    split = json.loads(args.split.read_text())
    manifest = json.loads((args.crossfit_dir / "manifest.json").read_text())
    rows = mapping["rows"]
    row_by_qkey = {row["qkey"]: row for row in rows}
    assert len(rows) == len(row_by_qkey) == 575
    assert split["status"] == "frozen"
    assert split["source"]["mapping_sha256"] == sha256(args.mapping)
    assert manifest["sources"]["mapping_sha256"] == sha256(args.mapping)
    assert manifest["sources"]["split_sha256"] == sha256(args.split)

    split_rows = split["rows"]
    assert len(split_rows) == 575
    assert {row["qkey"] for row in split_rows} == set(row_by_qkey)
    split_content = {"assignments": split["assignments"], "rows": split_rows}
    assert split["split_content_sha256"] == canonical_sha256(split_content)
    fold_by_qkey = {row["qkey"]: row["fold"] for row in split_rows}
    assert set(fold_by_qkey.values()) == set(FOLDS)

    fold_by_instance = {}
    for row in split_rows:
        original = row_by_qkey[row["qkey"]]
        assert row["instance_id"] == original["instance_id"]
        assert row["hop_idx"] == original["hop_idx"]
        previous = fold_by_instance.setdefault(row["instance_id"], row["fold"])
        assert previous == row["fold"], f"split instance: {row['instance_id']}"
    instances = {
        fold: {instance for instance, value in fold_by_instance.items() if value == fold}
        for fold in FOLDS
    }
    assert instances["A"].isdisjoint(instances["B"])

    base_arms = {}
    base_by_pv_qkey_condition = {}
    for pv in (0, 1, 2):
        path = args.episodes_dir / f"episodes_question_v2_near_pv{pv}.json"
        data = json.loads(path.read_text())
        assert manifest["sources"]["episodes"][f"pv{pv}"]["sha256"] == sha256(path)
        base_arms[pv] = data
        lookup = {}
        for episode in data["episodes"]:
            key = (episode["provenance"]["qkey"], episode["condition"])
            assert key not in lookup
            lookup[key] = episode
        assert len(lookup) == 575 * len(CONDITIONS)
        base_by_pv_qkey_condition[pv] = lookup

    eval_union = {pv: set() for pv in (0, 1, 2)}
    task_reports = {}
    for train_fold, eval_fold in (("A", "B"), ("B", "A")):
        task_name = f"train_{train_fold}_eval_{eval_fold}"
        task_dir = args.crossfit_dir / task_name
        meta = json.loads((task_dir / "meta.json").read_text())
        task_manifest = manifest["tasks"][task_name]
        assert task_manifest["meta_sha256"] == sha256(task_dir / "meta.json")
        assert meta["train_fold"] == train_fold and meta["eval_fold"] == eval_fold
        train_qkeys = {qkey for qkey, fold in fold_by_qkey.items() if fold == train_fold}
        eval_qkeys = {qkey for qkey, fold in fold_by_qkey.items() if fold == eval_fold}
        train_instances = {row_by_qkey[qkey]["instance_id"] for qkey in train_qkeys}
        eval_instances = {row_by_qkey[qkey]["instance_id"] for qkey in eval_qkeys}
        assert train_instances.isdisjoint(eval_instances)

        pairs_path = task_dir / "dpo_sft_pairs.jsonl"
        assert meta["pairs"]["sha256"] == sha256(pairs_path)
        pairs = jsonl(pairs_path)
        assert len(pairs) == 2 * len(train_qkeys)
        assert Counter(pair["kind"] for pair in pairs) == Counter(
            {"resist_wrong_tool": len(train_qkeys), "follow_right_tool": len(train_qkeys)}
        )
        seen_pair_keys = set()
        for pair in pairs:
            qkey = pair["qkey"]
            assert qkey in train_qkeys
            row = row_by_qkey[qkey]
            assert pair["instance_id"] == row["instance_id"]
            assert pair["hop_idx"] == row["hop_idx"]
            assert pair["chosen"] == f'FINAL: {row["gold_answer"]}'
            assert pair["rejected"] == f'FINAL: {row["final_near"]}'
            condition = "tool_wrong" if pair["kind"] == "resist_wrong_tool" else "tool_right"
            source = base_by_pv_qkey_condition[0][(qkey, condition)]
            assert pair["prompt"] == source["prompts"]["with_tool"]
            key = (qkey, pair["kind"])
            assert key not in seen_pair_keys
            seen_pair_keys.add(key)

        for pv in (0, 1, 2):
            path = task_dir / f"eval_question_v2_near_pv{pv}.json"
            data = json.loads(path.read_text())
            assert meta["eval_files"][f"pv{pv}"]["sha256"] == sha256(path)
            episodes = data["episodes"]
            assert len(episodes) == len(eval_qkeys) * len(CONDITIONS)
            counts = Counter(episode["condition"] for episode in episodes)
            assert counts == Counter({condition: len(eval_qkeys) for condition in CONDITIONS})
            seen = set()
            for episode in episodes:
                qkey = episode["provenance"]["qkey"]
                assert qkey in eval_qkeys
                key = (qkey, episode["condition"])
                assert key not in seen
                seen.add(key)
                assert episode == base_by_pv_qkey_condition[pv][key]
            eval_union[pv].update(qkey for qkey, _ in seen)

        task_reports[task_name] = {
            "n_train_questions": len(train_qkeys),
            "n_eval_questions": len(eval_qkeys),
            "n_pairs": len(pairs),
            "instance_overlap": 0,
            "eval_episodes_per_pv": len(eval_qkeys) * len(CONDITIONS),
        }

    for pv in (0, 1, 2):
        assert eval_union[pv] == set(row_by_qkey)

    report = {
        "schema_version": "memtoc-crossfit-verification-v1",
        "status": "PASS",
        "mapping_sha256": sha256(args.mapping),
        "split_sha256": sha256(args.split),
        "crossfit_manifest_sha256": sha256(args.crossfit_dir / "manifest.json"),
        "n_questions": len(rows),
        "n_instances": len(fold_by_instance),
        "zero_instance_leakage": True,
        "out_of_fold_eval_union": 575,
        "tasks": task_reports,
        "verifier_sha256": sha256(Path(__file__)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
