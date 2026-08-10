"""Build leakage-free DPO/SFT pairs and out-of-fold eval episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FOLDS = ("A", "B")
CONDITIONS = ("tool_right", "tool_wrong", "tool_error", "no_tool", "no_conflict")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_commit() -> str:
    override = os.environ.get("KCB_CODE_COMMIT")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def load_arm(path: Path) -> dict:
    data = json.loads(path.read_text())
    assert len(data["episodes"]) == data["summary"]["n_episodes"]
    return data


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
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data_e096_crossfit")
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text())
    split = json.loads(args.split.read_text())
    assert split["status"] == "frozen"
    assert split["source"]["mapping_sha256"] == sha256(args.mapping)
    row_by_qkey = {row["qkey"]: row for row in mapping["rows"]}
    fold_by_qkey = {row["qkey"]: row["fold"] for row in split["rows"]}
    assert set(row_by_qkey) == set(fold_by_qkey)

    arms = {
        pv: load_arm(args.episodes_dir / f"episodes_question_v2_near_pv{pv}.json")
        for pv in (0, 1, 2)
    }
    for arm in arms.values():
        assert arm["summary"]["mapping_sha256"] == sha256(args.mapping)

    canonical_by_qkey: dict[str, dict[str, dict]] = {}
    for episode in arms[0]["episodes"]:
        qkey = episode["provenance"]["qkey"]
        canonical_by_qkey.setdefault(qkey, {})[episode["condition"]] = episode
    assert set(canonical_by_qkey) == set(row_by_qkey)

    report = {
        "schema_version": "memtoc-crossfit-inputs-v1",
        "dataset_row": "memtoc-v1",
        "seed": args.seed,
        "sources": {
            "mapping": str(args.mapping.resolve().relative_to(ROOT)),
            "mapping_sha256": sha256(args.mapping),
            "split": str(args.split.resolve().relative_to(ROOT)),
            "split_sha256": sha256(args.split),
            "episodes": {
                f"pv{pv}": {
                    "path": str(
                        (args.episodes_dir / f"episodes_question_v2_near_pv{pv}.json")
                        .resolve()
                        .relative_to(ROOT)
                    ),
                    "sha256": sha256(
                        args.episodes_dir / f"episodes_question_v2_near_pv{pv}.json"
                    ),
                }
                for pv in (0, 1, 2)
            },
        },
        "tasks": {},
        "code_commit": code_commit(),
        "builder_sha256": sha256(Path(__file__)),
    }

    for train_fold, eval_fold in (("A", "B"), ("B", "A")):
        task_name = f"train_{train_fold}_eval_{eval_fold}"
        task_dir = args.out_dir / task_name
        train_qkeys = {qkey for qkey, fold in fold_by_qkey.items() if fold == train_fold}
        eval_qkeys = {qkey for qkey, fold in fold_by_qkey.items() if fold == eval_fold}
        train_instances = {row_by_qkey[qkey]["instance_id"] for qkey in train_qkeys}
        eval_instances = {row_by_qkey[qkey]["instance_id"] for qkey in eval_qkeys}
        assert train_qkeys.isdisjoint(eval_qkeys)
        assert train_instances.isdisjoint(eval_instances)
        assert train_qkeys | eval_qkeys == set(row_by_qkey)

        pairs = []
        for qkey in sorted(train_qkeys):
            row = row_by_qkey[qkey]
            conditions = canonical_by_qkey[qkey]
            assert set(conditions) == set(CONDITIONS)
            wrong = conditions["tool_wrong"]
            right = conditions["tool_right"]
            gold = row["gold_answer"]
            distractor = row["final_near"]
            common = {
                "qkey": qkey,
                "instance_id": row["instance_id"],
                "hop_idx": row["hop_idx"],
                "train_fold": train_fold,
            }
            pairs.append(
                {
                    **common,
                    "kind": "resist_wrong_tool",
                    "prompt": wrong["prompts"]["with_tool"],
                    "chosen": f"FINAL: {gold}",
                    "rejected": f"FINAL: {distractor}",
                }
            )
            pairs.append(
                {
                    **common,
                    "kind": "follow_right_tool",
                    "prompt": right["prompts"]["with_tool"],
                    "chosen": f"FINAL: {gold}",
                    "rejected": f"FINAL: {distractor}",
                }
            )
        random.Random(args.seed + (0 if train_fold == "A" else 1)).shuffle(pairs)
        pairs_path = task_dir / "dpo_sft_pairs.jsonl"
        write_jsonl(pairs_path, pairs)

        eval_files = {}
        for pv, arm in arms.items():
            episodes = [
                episode
                for episode in arm["episodes"]
                if episode["provenance"]["qkey"] in eval_qkeys
            ]
            counts = Counter(episode["condition"] for episode in episodes)
            assert counts == Counter({condition: len(eval_qkeys) for condition in CONDITIONS})
            summary = dict(arm["summary"])
            summary.update(
                {
                    "schema_version": "memtoc-crossfit-eval-v1",
                    "n_questions": len(eval_qkeys),
                    "n_episodes": len(episodes),
                    "crossfit": {
                        "train_fold": train_fold,
                        "eval_fold": eval_fold,
                        "split_sha256": sha256(args.split),
                        "zero_instance_leakage": True,
                    },
                    "code_commit": code_commit(),
                    "builder_sha256": sha256(Path(__file__)),
                }
            )
            path = task_dir / f"eval_question_v2_near_pv{pv}.json"
            write_json(path, {"summary": summary, "episodes": episodes})
            eval_files[f"pv{pv}"] = {
                "path": str(path.resolve().relative_to(ROOT)),
                "sha256": sha256(path),
                "n_episodes": len(episodes),
            }

        task_meta = {
            "train_fold": train_fold,
            "eval_fold": eval_fold,
            "n_train_questions": len(train_qkeys),
            "n_eval_questions": len(eval_qkeys),
            "n_train_instances": len(train_instances),
            "n_eval_instances": len(eval_instances),
            "instance_overlap": 0,
            "n_pairs": len(pairs),
            "pair_kinds": dict(sorted(Counter(pair["kind"] for pair in pairs).items())),
            "pairs": {
                "path": str(pairs_path.resolve().relative_to(ROOT)),
                "sha256": sha256(pairs_path),
            },
            "eval_files": eval_files,
        }
        write_json(task_dir / "meta.json", task_meta)
        task_meta["meta_sha256"] = sha256(task_dir / "meta.json")
        report["tasks"][task_name] = task_meta

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "manifest.json", report)
    print(json.dumps(report["tasks"], ensure_ascii=False, indent=2))
    print(f"manifest_sha256={sha256(args.out_dir / 'manifest.json')}")


if __name__ == "__main__":
    main()
