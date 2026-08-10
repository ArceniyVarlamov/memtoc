"""DPO pairs for the control stage (the Context-DPO recipe, adapted to
arbitration-by-correctness; §7 DRAFT). The training objective is "follow
CORRECTNESS, not a side": half the pairs teach the model to resist an
incorrect tool, half teach it to follow a correct one.

- tool_wrong episode: chosen = 'FINAL: {gold}' (hold), rejected =
  'FINAL: {distractor}' (go with the incorrect tool);
- tool_right episode: chosen = 'FINAL: {gold}' (= the tool return, follow),
  rejected = 'FINAL: {distractor of the same hop}' (leave a correct tool).

Contamination: training hops are ONLY instances whose id is absent from the
v1-full eval files (asserted, and the ids are recorded in the metadata).

Run: python -m scripts.build_dpo_pairs --out data/dpo_pairs_v1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from memtoc.episodes import build_episode_set_v1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolhop", default="data/ToolHop.json")
    ap.add_argument("--type-map", default="data/entity_types.json")
    ap.add_argument("--eval-episodes", default="data/episodes_v1_full_near_pv0.json")
    ap.add_argument("--n-instances", type=int, default=400,
                    help="how many eligible instances to generate BEFORE filtering")
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--max-pairs", type=int, default=1600)
    ap.add_argument("--sanitize-pool", action="store_true",
                    help="draw the candidate pool through the sanitiser (pool v2+); "
                         "default False — bit-for-bit with the original build")
    ap.add_argument("--hops-manifest", default=None,
                    help="also write out (hop, question, gold) for the "
                         "training hops — the input of the gold-judge audit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eval_ids = {e["instance_id"] for e in json.loads(
        Path(args.eval_episodes).read_text())["episodes"]}
    data = build_episode_set_v1(
        toolhop_path=args.toolhop, type_map_path=args.type_map,
        n_instances=args.n_instances,
        conditions=["tool_right", "tool_wrong"],
        seed=args.seed, divergence="near", prompt_variant=0, prompt_version=2,
        sanitize_pool=args.sanitize_pool)

    by_hop: dict[str, dict] = {}
    for e in data["episodes"]:
        if e["instance_id"] in eval_ids:
            continue
        hop = f"{e['instance_id']}-{e['hop_idx']}"
        by_hop.setdefault(hop, {})[e["condition"]] = e

    pairs = []
    for hop, conds in sorted(by_hop.items()):
        tw, tr = conds.get("tool_wrong"), conds.get("tool_right")
        if not tw or not tr:
            continue
        distractor = (tw["tool_output"] or {}).get("result")
        gold = tw["gold_answer"]
        if not distractor or str(distractor) == gold:
            continue
        pairs.append({"hop": hop, "kind": "resist_wrong_tool",
                      "prompt": tw["prompts"]["with_tool"],
                      "chosen": f"FINAL: {gold}",
                      "rejected": f"FINAL: {distractor}"})
        pairs.append({"hop": hop, "kind": "follow_right_tool",
                      "prompt": tr["prompts"]["with_tool"],
                      "chosen": f"FINAL: {gold}",
                      "rejected": f"FINAL: {distractor}"})

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    pairs = pairs[: args.max_pairs]

    train_ids = {int(p["hop"].split("-")[0]) for p in pairs}
    assert not (train_ids & eval_ids), "CONTAMINATION: overlap with eval!"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {"n_pairs": len(pairs), "seed": args.seed,
            "kinds": {k: sum(1 for p in pairs if p["kind"] == k)
                      for k in ("resist_wrong_tool", "follow_right_tool")},
            "n_train_instances": len(train_ids),
            "train_instance_id_range": [min(train_ids), max(train_ids)],
            "eval_instances_excluded": len(eval_ids),
            "contamination": "0 (assert)",
            "sanitize_pool": args.sanitize_pool,
            "type_map": args.type_map,
            "toolhop_sha256": data["summary"]["toolhop_sha256"],
            "pairs_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))

    if args.hops_manifest:
        pair_hops = {p["hop"] for p in pairs}
        rows = []
        for hop in sorted(pair_hops):
            e = by_hop[hop]["tool_wrong"]
            rows.append({"hop": hop, "question": e["question"],
                         "gold": e["gold_answer"]})
        mp = Path(args.hops_manifest)
        mp.write_text(json.dumps(
            {"n_hops": len(rows), "pairs_file": str(out),
             "pairs_sha256": meta["pairs_sha256"], "hops": rows},
            ensure_ascii=False, indent=1))
        print(f"[manifest] {len(rows)} hops -> {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
