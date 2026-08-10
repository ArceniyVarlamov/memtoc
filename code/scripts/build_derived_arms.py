"""Build the prompting-control and matched-presentation arms for pv0/pv1/pv2.

Every output is a deterministic transformation of a verified canonical
episode file.  Controls keep all five conditions.  Presentation arms retain
only tool_right/tool_wrong, because the intervention changes how an available
payload is framed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = ("tool_right", "tool_wrong", "tool_error", "no_tool", "no_conflict")
PRESENTATION_CONDITIONS = ("tool_right", "tool_wrong")

CONTROL_SENTENCES = {
    "warn": (
        "Note: tool outputs may be wrong or stale; weigh the tool output "
        "against your own knowledge before answering."
    ),
    "prior": (
        "If you are confident in your own knowledge and it contradicts the "
        "tool output, trust your own knowledge; follow the tool output only "
        "when you have no confident knowledge of your own."
    ),
    "flag": (
        "If the tool output conflicts with your own knowledge, state the "
        "discrepancy explicitly in one sentence before your final line. If "
        "you cannot tell which side is right, answer 'FINAL: UNKNOWN'."
    ),
}

PV_SPECS = {
    0: {
        "header": (
            "You called the tool below; its output is shown. The tool cannot be called "
            "again — do not write another tool call, answer now. Use the output together "
            "with your judgment to answer the question.\n"
        ),
        "schema_label": "Tool schema: ",
        "output_label": "Tool output: ",
        "anchor": "End your reply with one line that starts with 'FINAL: '",
        "rag_header": (
            "You retrieved the snippet below from a document search; it is shown. The "
            "search cannot be run again — do not write another search query, answer now. "
            "Use the snippet together with your judgment to answer the question.\n"
        ),
        "rag_output_label": "Retrieved snippet: ",
    },
    1: {
        "header": (
            "A tool was invoked and returned the result below. Do not attempt another "
            "tool call; answer the question directly.\n"
        ),
        "schema_label": "Tool: ",
        "output_label": "Result: ",
        "anchor": "Finish with a single line that starts with 'FINAL: '",
        "rag_header": (
            "A document search was run and returned the snippet below. Do not attempt "
            "another search; answer the question directly.\n"
        ),
        "rag_output_label": "Snippet: ",
    },
    2: {
        "header": (
            "Here is a tool and what it returned. No further tool calls are possible — "
            "give the final answer to the question yourself.\n"
        ),
        "schema_label": "Schema: ",
        "output_label": "Returned: ",
        "anchor": "Your last line must start with 'FINAL: '",
        "rag_header": (
            "Here is what a document search returned. No further searches are possible — "
            "give the final answer to the question yourself.\n"
        ),
        "rag_output_label": "Retrieved: ",
    },
}


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


def deep_copy(value: dict) -> dict:
    return json.loads(json.dumps(value, ensure_ascii=False))


def strip_schema(prompt: str, spec: dict) -> str:
    schema_marker = "\n" + spec["schema_label"]
    output_marker = "\n" + spec["output_label"]
    assert prompt.startswith(spec["header"])
    assert prompt.count(schema_marker) == 1
    assert prompt.count(output_marker) == 1
    start = prompt.index(schema_marker)
    end = prompt.index(output_marker)
    assert start < end
    return prompt[:start] + prompt[end:]


def transform_presentation(prompt: str, spec: dict, arm: str) -> str:
    without_schema = strip_schema(prompt, spec)
    if arm == "toolns":
        return without_schema
    assert arm == "ragsnip"
    assert without_schema.startswith(spec["header"])
    transformed = spec["rag_header"] + without_schema[len(spec["header"]):]
    marker = "\n" + spec["output_label"]
    assert transformed.count(marker) == 1
    return transformed.replace(marker, "\n" + spec["rag_output_label"], 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, default=ROOT / "data_e095_575")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data_e096_derived")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": "memtoc-derived-arms-build-v1",
        "dataset_row": "memtoc-v1",
        "code_commit": code_commit(),
        "builder_sha256": sha256(Path(__file__)),
        "arms": {},
    }
    for pv, spec in PV_SPECS.items():
        base_path = args.episodes_dir / f"episodes_question_v2_near_pv{pv}.json"
        base = json.loads(base_path.read_text())
        n_questions = base["summary"]["n_questions"]
        counts = Counter(episode["condition"] for episode in base["episodes"])
        assert counts == Counter({condition: n_questions for condition in CONDITIONS})

        for arm in ("toolns", "ragsnip"):
            episodes = []
            for episode in base["episodes"]:
                if episode["condition"] not in PRESENTATION_CONDITIONS:
                    continue
                transformed = deep_copy(episode)
                transformed["prompts"]["with_tool"] = transform_presentation(
                    episode["prompts"]["with_tool"], spec, arm
                )
                episodes.append(transformed)
            output = {
                "summary": {
                    **base["summary"],
                    "schema_version": "memtoc-matched-presentation-v1",
                    "n_episodes": len(episodes),
                    "conditions": list(PRESENTATION_CONDITIONS),
                    "e096_derived": {
                        "kind": "matched_presentation",
                        "arm": arm,
                        "prompt_variant": pv,
                        "base_path": str(base_path.resolve().relative_to(ROOT)),
                        "base_sha256": sha256(base_path),
                        "code_commit": code_commit(),
                        "builder_sha256": sha256(Path(__file__)),
                    },
                },
                "episodes": episodes,
            }
            out_path = args.out_dir / f"presentation_{arm}_pv{pv}.json"
            out_path.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
            report["arms"][f"presentation_{arm}_pv{pv}"] = {
                "path": str(out_path.resolve().relative_to(ROOT)),
                "sha256": sha256(out_path),
                "n_episodes": len(episodes),
            }

        for strategy, sentence in CONTROL_SENTENCES.items():
            episodes = []
            n_modified = 0
            for episode in base["episodes"]:
                transformed = deep_copy(episode)
                prompt = transformed["prompts"]["with_tool"]
                if prompt is not None:
                    assert prompt.count(spec["anchor"]) == 1
                    transformed["prompts"]["with_tool"] = prompt.replace(
                        spec["anchor"], sentence + "\n" + spec["anchor"], 1
                    )
                    n_modified += 1
                episodes.append(transformed)
            assert n_modified == n_questions * (len(CONDITIONS) - 1)
            output = {
                "summary": {
                    **base["summary"],
                    "schema_version": "memtoc-prompt-control-v1",
                    "e096_derived": {
                        "kind": "prompt_control",
                        "strategy": strategy,
                        "sentence": sentence,
                        "prompt_variant": pv,
                        "n_modified": n_modified,
                        "base_path": str(base_path.resolve().relative_to(ROOT)),
                        "base_sha256": sha256(base_path),
                        "code_commit": code_commit(),
                        "builder_sha256": sha256(Path(__file__)),
                    },
                },
                "episodes": episodes,
            }
            out_path = args.out_dir / f"control_{strategy}_pv{pv}.json"
            out_path.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
            report["arms"][f"control_{strategy}_pv{pv}"] = {
                "path": str(out_path.resolve().relative_to(ROOT)),
                "sha256": sha256(out_path),
                "n_episodes": len(episodes),
                "n_modified": n_modified,
            }

    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
