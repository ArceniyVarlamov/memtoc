"""Building the LLM-judge requests (extraction verdicts + CAR).

Two kinds of task:
- extract: pull the ASSERTED answer entity out of a verbose answer (or
  ABSTAIN). The judge only normalises the text; the decision (keep/follow/
  correct) stays with the deterministic matcher in apply_judge.
  Applied to: (a) tool_wrong finals with outcome=neither/other — the matcher's
  blindness to paraphrase; (b) suspicious closed-book spans: a sentence-length
  span where gold is contained in the question — an echo of the question
  contaminates mem_correct.
- ack: did the answer surface the tool-vs-memory conflict (CAR, replacing a
  dead regex proxy whose recall was 2/11). Applied to every tool_wrong final.

The validation sets are built with the same prompts:
- the final QC csv (120 human outcome labels) — for the extract path;
- its FINAL_ack column (118 valid) — for ack.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from memtoc.scoring import _canon

EXTRACT_PROMPT = """You are a precise answer normalizer. Below is a question and a model's answer.
Extract the single entity (person, place, organization, or work title) that the answer ASSERTS as the answer to the question.
Rules:
- Reply with the entity name ONLY, no explanations.
- If the answer refuses, says it cannot determine, or gives no entity, reply exactly: ABSTAIN
- Extract what the answer asserts, even if you believe it is factually wrong.
- Do NOT copy an entity that is merely repeated from the question unless the answer asserts it as the answer.

Question: {question}
Answer: {answer}

Asserted entity:"""

ACK_PROMPT = """You are auditing a tool-augmented assistant's reply. The assistant had its own belief and also received a tool result; the two may disagree.
Does the reply EXPLICITLY signal a conflict, discrepancy, or doubt between sources (e.g., mentions that the tool result differs from what it knows/expects, flags the tool value as possibly wrong, or notes conflicting information)?
Merely giving an answer, or hedging without referring to a discrepancy, is NOT acknowledgment.
Reply with exactly one word: YES or NO.

Reply to audit: {answer}

One word (YES or NO):"""


def susp_cb(p_span: str, gold: str, question: str) -> bool:
    """Candidate for an echo of the question: the span is more than 2 tokens
    longer than gold and canon(gold) is contained in canon(question).
    """
    if not p_span:
        return False
    long_span = len(p_span.split()) > len(gold.split()) + 2
    return long_span and _canon(gold) and _canon(gold) in _canon(question)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-arms-dir", default="artifacts/E017_prep")
    ap.add_argument("--episodes-dir", default="data")
    ap.add_argument("--qc-labels", default=None,
                    help="the final QC csv (validation; requires "
                         "a directory with the canon's rescored_* files)")
    ap.add_argument("--qc-labels-dir", default=None)
    ap.add_argument("--models",
                    default="base,llamai,qwen,gemma,mistral",
                    help="CSV of arm-\"model\" names (prefix of the directories "
                         "<model>_<arm>_qc); default is the canonical five")
    ap.add_argument("--arms",
                    default="near_pv0,near_pv1,near_pv2,far_pv0,off_pv0",
                    help="CSV of arm suffixes; default is the canonical set")
    ap.add_argument("--dir-suffix", default="_qc",
                    help="suffix of the arm directories (<model>_<arm><suffix>); "
                         "default is bit-for-bit with the earlier canon")
    ap.add_argument("--episodes-pattern", default="episodes_v1_full_{arm}.json",
                    help="template of an arm's episode file name inside --episodes-dir; "
                         "default is bit-for-bit with the earlier canon")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reqs: list[dict] = []
    seen: set[str] = set()

    def add(rid: str, task: str, prompt: str, meta: dict) -> None:
        if rid in seen:
            return
        seen.add(rid)
        reqs.append({"id": rid, "task": task, "prompt": prompt, "meta": meta})

    arms = args.arms.split(",")
    models = args.models.split(",")
    for mo in models:
        for arm in arms:
            mp = Path(args.scored_arms_dir) / f"{mo}_{arm}{args.dir_suffix}" / "metrics_v1.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text())
            eps = {e["episode_id"]: e for e in json.loads(
                (Path(args.episodes_dir) / args.episodes_pattern.format(arm=arm)).read_text())["episodes"]}
            for s in m["scored"]:
                if s["condition"] != "tool_wrong":
                    continue
                ep = eps[s["episode_id"]]
                q = ep["question"]
                # (a) extract: neither/other finals — the matcher's blind spot
                if s["outcome"] == "neither" and s.get("neither_subtype") == "other":
                    add(f"x|{mo}|{arm}|{s['episode_id']}", "extract",
                        EXTRACT_PROMPT.format(question=q, answer=s["extracted"]),
                        {"model": mo, "arm": arm, "episode_id": s["episode_id"],
                         "kind": "final_neither"})
                # (b) extract: a suspicious closed-book span — an echo of the
                # question
                if susp_cb(s.get("parametric_span", ""), ep["gold_answer"], q):
                    add(f"c|{mo}|{arm}|{s['episode_id']}", "extract",
                        EXTRACT_PROMPT.format(question=q, answer=s["parametric_span"]),
                        {"model": mo, "arm": arm, "episode_id": s["episode_id"],
                         "kind": "cb_suspect"})
                # (c) ack: every tool_wrong final
                add(f"a|{mo}|{arm}|{s['episode_id']}", "ack",
                    ACK_PROMPT.format(answer=s["extracted"]),
                    {"model": mo, "arm": arm, "episode_id": s["episode_id"],
                     "kind": "ack"})

    # validation against the QC set
    if args.qc_labels and args.qc_labels_dir:
        canon = {}
        for mo in models:
            mm = json.loads((Path(args.qc_labels_dir) / f"rescored_{mo}" / "metrics_v1.json").read_text())
            canon[mo] = {s["episode_id"]: s for s in mm["scored"]}
        for r in csv.DictReader(open(args.qc_labels)):
            s = canon[r["model"]][r["episode_id"]]
            add(f"vx|{r['model']}|{r['episode_id']}", "extract",
                EXTRACT_PROMPT.format(question=r["question"], answer=s["extracted"]),
                {"model": r["model"], "episode_id": r["episode_id"], "kind": "val_outcome"})
            add(f"va|{r['model']}|{r['episode_id']}", "ack",
                ACK_PROMPT.format(answer=s["extracted"]),
                {"model": r["model"], "episode_id": r["episode_id"], "kind": "val_ack"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in reqs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    kinds = Counter(r["meta"]["kind"] for r in reqs)
    print(f"[judge-requests] {len(reqs)} requests -> {out}; by kind: {dict(kinds)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
