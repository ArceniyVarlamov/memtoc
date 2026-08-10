"""Materialize the episodes from the corrected 575-question distractor mapping.

Unlike the old instance-slice builders, this one uses one canonical occurrence of
every unique factual ToolHop subquestion admitted by question_pool. The near
tool output is read only from the mapping supplied with ``--mapping``; the
canonical default is `results/construction/distractor_mapping_575.json`.

Outputs: three prompt paraphrase arms (pv0/pv1/pv2), each containing the same
575 questions x 5 conditions = 2,875 episodes. Only prompt wording differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from memtoc.data import iter_hops, load_toolhop
from memtoc.episodes import PROMPT_VERSIONS
from memtoc.inject import CONDITIONS, TOOL_ERROR_PAYLOAD

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONDITIONS = ["tool_right", "tool_wrong", "tool_error", "no_tool", "no_conflict"]


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


def tool_output(condition: str, gold: str, near: str) -> dict | None:
    if condition in ("tool_right", "no_conflict"):
        return {"result": gold}
    if condition == "tool_wrong":
        return {"result": near}
    if condition == "tool_error":
        return dict(TOOL_ERROR_PAYLOAD)
    if condition == "no_tool":
        return None
    raise AssertionError(condition)


def make_episode(
    row: dict,
    hop,
    condition: str,
    prompt_variant: int,
    seed: int,
    dataset_row: str,
) -> dict:
    templates = PROMPT_VERSIONS[2]
    output = tool_output(condition, row["gold_answer"], row["final_near"])
    with_tool = None
    if output is not None:
        with_tool = templates["with_tool"][prompt_variant].format(
            schema=json.dumps(hop.tool_schema, ensure_ascii=False),
            tool_output=json.dumps(output, ensure_ascii=False),
            question=row["question"],
        )
    episode = {
        "episode_id": f'{row["instance_id"]}-{row["hop_idx"]}-{condition}-v{prompt_variant}',
        "instance_id": row["instance_id"],
        "hop_idx": row["hop_idx"],
        "condition": condition,
        "tau": row["gold_kind"],
        "family": row["family"],
        "divergence_bucket": "near" if condition == "tool_wrong" else None,
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "tool_output": output,
        "tool_correct": None if condition == "no_tool" else condition in ("tool_right", "no_conflict"),
        "memory_correct": None,
        "tool_schema": hop.tool_schema,
        "prompt_variant": prompt_variant,
        "prompt_version": 2,
        "prompts": {
            "closed_book": templates["closed_book"][prompt_variant].format(question=row["question"]),
            "with_tool": with_tool,
        },
        "distractor_flags": [],
        "qc_flags": list(hop.qc_flags),
        "seed": seed,
        "provenance": {
            "source": f"ToolHop/{dataset_row}-question-v2",
            "instance_id": row["instance_id"],
            "hop_idx": row["hop_idx"],
            "qkey": row["qkey"],
            "distractor_source": row["source"],
            "distractor_method": row["method"],
        },
    }
    if condition == "tool_wrong":
        episode["distractor"] = row["final_near"]
    return episode


def build(mapping: dict, toolhop_path: Path, prompt_variant: int, seed: int) -> dict:
    assert mapping["status"] == "frozen"
    n_questions = len(mapping["rows"])
    dataset_row = mapping.get("correction", {}).get("row", "memtoc-v0")
    assert prompt_variant in (0, 1, 2)
    hops = {
        (hop.instance_id, hop.hop_idx): hop
        for instance in load_toolhop(str(toolhop_path))
        for hop in iter_hops(instance)
    }
    episodes = []
    for row in mapping["rows"]:
        key = (row["instance_id"], row["hop_idx"])
        assert key in hops, f"missing ToolHop occurrence: {key}"
        hop = hops[key]
        assert hop.question == row["question"], f"question mismatch: {key}"
        assert hop.gold_answer == row["gold_answer"], f"gold mismatch: {key}"
        assert hop.tool_schema, f"missing tool schema: {key}"
        for condition in DEFAULT_CONDITIONS:
            assert condition in CONDITIONS
            episodes.append(
                make_episode(row, hop, condition, prompt_variant, seed, dataset_row)
            )

    ids = [episode["episode_id"] for episode in episodes]
    assert len(episodes) == n_questions * len(DEFAULT_CONDITIONS)
    assert len(ids) == len(set(ids))
    tool_wrong = [episode for episode in episodes if episode["condition"] == "tool_wrong"]
    assert len(tool_wrong) == n_questions
    assert all(episode["tool_output"]["result"] == episode["distractor"] for episode in tool_wrong)
    assert all(episode["tool_output"]["result"] != episode["gold_answer"] for episode in tool_wrong)

    summary = {
        "schema_version": f"{dataset_row.lower()}-episodes-question-v2",
        "dataset_row": dataset_row,
        "version": "question-v2",
        "n_questions": n_questions,
        "n_episodes": len(episodes),
        "conditions": DEFAULT_CONDITIONS,
        "prompt_variant": prompt_variant,
        "prompt_version": 2,
        "divergence": "near",
        "seed": seed,
        "mapping_status": mapping["status"],
        "mapping_sha256": mapping["_sha256"],
        "toolhop_sha256": file_sha256(toolhop_path),
        "tau_distribution": dict(Counter(episode["tau"] for episode in tool_wrong)),
        "family_distribution": dict(Counter(episode["family"] for episode in tool_wrong)),
        "distractor_source": dict(Counter(
            episode["provenance"]["distractor_source"] for episode in tool_wrong
        )),
        "distractor_method": dict(Counter(
            episode["provenance"]["distractor_method"] for episode in tool_wrong
        )),
        "qc_flagged": sum(bool(episode["qc_flags"]) for episode in episodes),
        "config": {
            "mapping": mapping["_path"],
            "toolhop": str(toolhop_path.relative_to(ROOT)),
            "conditions": DEFAULT_CONDITIONS,
            "prompt_variant": prompt_variant,
            "prompt_version": 2,
            "seed": seed,
        },
        "code_commit": code_commit(),
        "builder_sha256": file_sha256(Path(__file__)),
    }
    return {"summary": summary, "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=ROOT / "results/construction/distractor_mapping_575.json",
    )
    parser.add_argument("--toolhop", type=Path, default=ROOT / "data/ToolHop.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data_e095_575")
    parser.add_argument("--stage-dir", type=Path, default=ROOT / "episodes_e095_575_stage")
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text())
    mapping["_sha256"] = file_sha256(args.mapping)
    mapping["_path"] = str(args.mapping.resolve().relative_to(ROOT))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    report = {"mapping_sha256": mapping["_sha256"], "arms": {}}
    for prompt_variant in (0, 1, 2):
        result = build(mapping, args.toolhop, prompt_variant, args.seed)
        name = f"episodes_question_v2_near_pv{prompt_variant}.json"
        out_path = args.out_dir / name
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
        shutil.copy2(out_path, args.stage_dir / name)
        report["arms"][f"pv{prompt_variant}"] = {
            "path": str(out_path.relative_to(ROOT)),
            "sha256": file_sha256(out_path),
            "n_episodes": result["summary"]["n_episodes"],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
