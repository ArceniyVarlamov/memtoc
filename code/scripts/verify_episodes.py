"""Independent structural verifier for the question-v2 episode files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from memtoc.data import iter_hops, load_toolhop
from memtoc.episodes import PROMPT_VERSIONS
from memtoc.inject import TOOL_ERROR_PAYLOAD

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ["tool_right", "tool_wrong", "tool_error", "no_tool", "no_conflict"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_output(condition: str, gold: str, near: str):
    if condition in ("tool_right", "no_conflict"):
        return {"result": gold}
    if condition == "tool_wrong":
        return {"result": near}
    if condition == "tool_error":
        return TOOL_ERROR_PAYLOAD
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=ROOT / "results/construction/distractor_mapping_575.json",
    )
    parser.add_argument("--toolhop", type=Path, default=ROOT / "data/ToolHop.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_e095_575")
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text())
    row_by_key = {(row["instance_id"], row["hop_idx"]): row for row in mapping["rows"]}
    n_questions = len(mapping["rows"])
    n_episodes = n_questions * len(CONDITIONS)
    hop_by_key = {
        (hop.instance_id, hop.hop_idx): hop
        for instance in load_toolhop(str(args.toolhop))
        for hop in iter_hops(instance)
    }
    assert len(row_by_key) == n_questions

    arms = []
    report = {"mapping_sha256": sha256(args.mapping), "arms": {}}
    for pv in (0, 1, 2):
        path = args.data_dir / f"episodes_question_v2_near_pv{pv}.json"
        data = json.loads(path.read_text())
        assert data["summary"]["mapping_sha256"] == sha256(args.mapping)
        assert data["summary"]["toolhop_sha256"] == sha256(args.toolhop)
        assert data["summary"]["prompt_variant"] == pv
        assert data["summary"]["n_questions"] == n_questions
        assert data["summary"]["n_episodes"] == n_episodes
        episodes = data["episodes"]
        assert len(episodes) == n_episodes
        assert len({episode["episode_id"] for episode in episodes}) == n_episodes

        counts = Counter(episode["condition"] for episode in episodes)
        assert counts == Counter({condition: n_questions for condition in CONDITIONS})
        for episode in episodes:
            key = (episode["instance_id"], episode["hop_idx"])
            row, hop = row_by_key[key], hop_by_key[key]
            condition = episode["condition"]
            assert episode["episode_id"] == f"{key[0]}-{key[1]}-{condition}-v{pv}"
            assert episode["question"] == row["question"] == hop.question
            assert episode["gold_answer"] == row["gold_answer"] == hop.gold_answer
            assert episode["tool_schema"] == hop.tool_schema
            assert episode["tool_output"] == expected_output(condition, row["gold_answer"], row["final_near"])
            assert episode["tool_correct"] == (
                None if condition == "no_tool" else condition in ("tool_right", "no_conflict")
            )
            assert episode["prompts"]["closed_book"] == PROMPT_VERSIONS[2]["closed_book"][pv].format(
                question=row["question"]
            )
            if condition == "no_tool":
                assert episode["prompts"]["with_tool"] is None
            else:
                expected_prompt = PROMPT_VERSIONS[2]["with_tool"][pv].format(
                    schema=json.dumps(hop.tool_schema, ensure_ascii=False),
                    tool_output=json.dumps(episode["tool_output"], ensure_ascii=False),
                    question=row["question"],
                )
                assert episode["prompts"]["with_tool"] == expected_prompt
            if condition == "tool_wrong":
                assert episode["distractor"] == row["final_near"]
                assert episode["divergence_bucket"] == "near"
            else:
                assert "distractor" not in episode
                assert episode["divergence_bucket"] is None

        arms.append(episodes)
        report["arms"][f"pv{pv}"] = {"sha256": sha256(path), "conditions": dict(counts)}

    # Across prompt variants, only prompt wording and pv identifiers may differ.
    for index in range(n_episodes):
        projected = []
        for episodes in arms:
            episode = dict(episodes[index])
            episode.pop("episode_id")
            episode.pop("prompt_variant")
            episode.pop("prompts")
            projected.append(episode)
        assert projected[0] == projected[1] == projected[2]

    report["status"] = "PASS"
    report["n_asserted_episodes"] = n_episodes * 3
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
