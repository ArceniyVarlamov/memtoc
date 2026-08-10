"""Build the frozen instance-group split used by all in-domain reruns.

The split unit is a ToolHop ``instance_id``.  Questions from one instance can
therefore never appear in both training and evaluation.  The deterministic
greedy allocator balances question count, family, answer type, distractor
source, and their joint strata across two folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FOLDS = ("A", "B")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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


def source_bucket(row: dict) -> str:
    return "human" if "human" in row["source"] else "machine"


def row_features(row: dict) -> list[str]:
    family = row["family"]
    tau = row["gold_kind"]
    source = source_bucket(row)
    return [
        "all",
        f"family:{family}",
        f"tau:{tau}",
        f"source:{source}",
        f"stratum:{family}|{tau}|{source}",
    ]


def feature_weight(name: str) -> float:
    if name == "all":
        return 6.0
    if name.startswith("stratum:"):
        return 2.0
    return 1.0


def distribution(rows: list[dict]) -> dict:
    return {
        "n_questions": len(rows),
        "n_instances": len({row["instance_id"] for row in rows}),
        "family": dict(sorted(Counter(row["family"] for row in rows).items())),
        "tau": dict(sorted(Counter(row["gold_kind"] for row in rows).items())),
        "source_bucket": dict(sorted(Counter(source_bucket(row) for row in rows).items())),
        "stratum": dict(
            sorted(
                Counter(
                    f'{row["family"]}|{row["gold_kind"]}|{source_bucket(row)}'
                    for row in rows
                ).items()
            )
        ),
    }


def build(mapping: dict, mapping_path: Path, seed: int) -> dict:
    rows = mapping["rows"]
    assert mapping["status"] == "frozen"
    assert mapping.get("correction", {}).get("row") == "memtoc-v1"
    assert len(rows) == 575
    assert len({row["qkey"] for row in rows}) == len(rows)

    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["instance_id"]].append(row)

    group_features: dict[int, Counter] = {}
    totals: Counter = Counter()
    for instance_id, group_rows in groups.items():
        counts = Counter(
            feature for row in group_rows for feature in row_features(row)
        )
        group_features[instance_id] = counts
        totals.update(counts)

    rng = random.Random(seed)
    tie_key = {instance_id: rng.random() for instance_id in groups}

    def rarity(instance_id: int) -> float:
        return sum(
            count / totals[feature]
            for feature, count in group_features[instance_id].items()
        )

    order = sorted(
        groups,
        key=lambda instance_id: (
            -rarity(instance_id),
            -len(groups[instance_id]),
            tie_key[instance_id],
            instance_id,
        ),
    )

    fold_features = {fold: Counter() for fold in FOLDS}
    fold_instances = {fold: [] for fold in FOLDS}
    fold_questions = Counter()

    def imbalance(candidate_fold: str, instance_id: int) -> float:
        projected = {fold: fold_features[fold].copy() for fold in FOLDS}
        projected[candidate_fold].update(group_features[instance_id])
        cost = 0.0
        for feature, total in totals.items():
            delta = projected["A"][feature] - projected["B"][feature]
            cost += feature_weight(feature) * (delta * delta / total)
        n_a = len(fold_instances["A"]) + (candidate_fold == "A")
        n_b = len(fold_instances["B"]) + (candidate_fold == "B")
        cost += 0.5 * ((n_a - n_b) ** 2 / len(groups))
        return cost

    for instance_id in order:
        costs = {fold: imbalance(fold, instance_id) for fold in FOLDS}
        best_cost = min(costs.values())
        candidates = [fold for fold in FOLDS if abs(costs[fold] - best_cost) < 1e-12]
        if len(candidates) > 1:
            candidates.sort(
                key=lambda fold: (
                    fold_questions[fold], len(fold_instances[fold]), fold
                )
            )
        fold = candidates[0]
        fold_instances[fold].append(instance_id)
        fold_questions[fold] += len(groups[instance_id])
        fold_features[fold].update(group_features[instance_id])

    instance_to_fold = {
        instance_id: fold
        for fold, instance_ids in fold_instances.items()
        for instance_id in instance_ids
    }
    assert set(instance_to_fold) == set(groups)
    assert set(fold_instances["A"]).isdisjoint(fold_instances["B"])

    split_rows = [
        {
            "id": row["id"],
            "qkey": row["qkey"],
            "instance_id": row["instance_id"],
            "hop_idx": row["hop_idx"],
            "fold": instance_to_fold[row["instance_id"]],
        }
        for row in rows
    ]
    fold_rows = {
        fold: [row for row in rows if instance_to_fold[row["instance_id"]] == fold]
        for fold in FOLDS
    }
    assert sum(len(fold_rows[fold]) for fold in FOLDS) == 575

    assignments = [
        {
            "instance_id": instance_id,
            "fold": instance_to_fold[instance_id],
            "n_questions": len(groups[instance_id]),
            "qkeys": [row["qkey"] for row in groups[instance_id]],
        }
        for instance_id in sorted(groups)
    ]
    split_content = {"assignments": assignments, "rows": split_rows}
    return {
        "schema_version": "memtoc-instance-group-split-v1",
        "status": "frozen",
        "dataset_row": "memtoc-v1",
        "seed": seed,
        "algorithm": {
            "name": "deterministic-greedy-two-fold",
            "group_unit": "ToolHop instance_id",
            "balanced_features": [
                "question_count",
                "family",
                "gold_kind",
                "human_vs_machine_distractor_source",
                "family_x_gold_kind_x_source",
            ],
        },
        "source": {
            "mapping": str(mapping_path.resolve().relative_to(ROOT)),
            "mapping_sha256": sha256(mapping_path),
            "mapping_content_sha256": mapping["content_sha256_575"],
        },
        "summary": {
            "n_questions": len(rows),
            "n_instances": len(groups),
            "zero_instance_leakage": True,
            "folds": {fold: distribution(fold_rows[fold]) for fold in FOLDS},
        },
        "assignments": assignments,
        "rows": split_rows,
        "split_content_sha256": canonical_sha256(split_content),
        "code_commit": code_commit(),
        "builder_sha256": sha256(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=ROOT / "results/construction/distractor_mapping_575.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/quality_control/crossfit_folds.json",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    result = build(json.loads(args.mapping.read_text()), args.mapping, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"split_sha256={sha256(args.out)}")


if __name__ == "__main__":
    main()
