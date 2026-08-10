"""Paired statistics for far divergence and the protocol ablation.

A reading layer over the deterministic rescore and the single judge pass. The
statistics are imported unchanged from the paired-treatment module (cluster
bootstrap n=1000, sign-flip n=10000, RNG default_rng(0), Holm inside the
family); the contrast runs on a common support, because the far arm covers 443
of the 542 questions and the cell can only be compared on the intersection.

Pre-registration was written before the answers were collected. The families
are declared here before any aggregate is read:

  A far_keep      — 5 models, keep on arb: far vs canonical pv0 (Holm over 5);
  B protocol_keep — the anchor (llamai), keep on arb: the protocol without the
                    suggestion vs canonical pv0 (a single contrast, so Holm is
                    trivial).

Secondary readings (not Holm-gated): CRA_arb, prior_bw for far; for the
protocol also tool_gold_follow, te_abstain, nc_CRA — that arm carries all four
conditions with a payload.

Both arms are pv0: far and the protocol are defined only on the zeroth
paraphrase, so there are no pv slices and no pooled figure here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import scripts.build_paired_treatment_summary as bpt

ROOT = Path(__file__).resolve().parent.parent
FAR_MODELS = ["base", "gemma", "llamai", "mistral", "qwen"]
PROTOCOL_MODEL = "llamai"
FAR_METRICS = ["keep", "CRA_arb", "prior_bw"]
PROTOCOL_METRICS = ["keep", "CRA_arb", "prior_bw", "tool_gold_follow",
                    "te_abstain", "nc_CRA"]
PRIMARY = "keep"


def block(base_vals: dict[str, bool], treat_vals: dict[str, bool]) -> dict:
    """Paired contrast on the intersection of supports; the drift is printed beside it."""
    clusters: dict[str, list[float]] = {}
    for q in set(base_vals) & set(treat_vals):
        clusters[q] = [float(treat_vals[q]) - float(base_vals[q])]
    if not clusters:
        return {"_absent": "condition/cell is absent from this arm"}
    out = bpt.paired_block(clusters)
    out["n_arm_only"] = len(set(treat_vals) - set(base_vals))
    out["n_canonical_only"] = len(set(base_vals) - set(treat_vals))
    return out


def point(vals: dict[str, bool]) -> dict:
    if not vals:
        return {"_absent": "condition/cell is absent from this arm"}
    out = bpt.paired_block({q: [float(x)] for q, x in vals.items()})
    out.pop("p_signflip")
    out["rate"] = out.pop("delta")
    return out


def build_layer(load_arm, load_canonical) -> dict:
    layer: dict = {"far": {}, "protocol": {}}

    family_a: dict[str, dict] = {}
    for model in FAR_MODELS:
        arm = load_arm(model, "far_pv0")
        if arm is None:
            layer["far"][model] = {"_absent": "no such arm"}
            continue
        canon = load_canonical(model)
        entry: dict = {}
        for metric in FAR_METRICS:
            base_vals = bpt.qvals(canon, metric)
            treat_vals = bpt.qvals(arm, metric)
            entry[metric] = {
                "contrast": block(base_vals, treat_vals),
                "canonical_point": point(base_vals),
                "far_point": point(treat_vals),
            }
        layer["far"][model] = entry
        family_a[model] = entry[PRIMARY]["contrast"]
    if family_a:
        layer["far_keep_holm"] = bpt.with_holm(
            {k: dict(v) for k, v in family_a.items()})

    arm = load_arm(PROTOCOL_MODEL, "protocol_pv0")
    if arm is None:
        layer["protocol"] = {"_absent": "no such arm"}
    else:
        canon = load_canonical(PROTOCOL_MODEL)
        entry = {}
        for metric in PROTOCOL_METRICS:
            base_vals = bpt.qvals(canon, metric)
            treat_vals = bpt.qvals(arm, metric)
            entry[metric] = {
                "contrast": block(base_vals, treat_vals),
                "canonical_point": point(base_vals),
                "nosentence_point": point(treat_vals),
            }
        layer["protocol"] = {PROTOCOL_MODEL: entry}
    return layer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-dir", default="results/distractor_distance_prep")
    ap.add_argument("--judged-dir", default="results/distractor_distance_judged")
    ap.add_argument("--untreated-det-dir", default="artifacts/E107_prep")
    ap.add_argument("--untreated-judged-dir", default="results/scored_episodes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def scored(path: Path, key: str) -> list[dict] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text())[key]

    def load_det(model: str, arm: str):
        return scored(Path(args.det_dir) / f"{model}_{arm}" / "metrics_v1.json",
                      "scored")

    def load_judged(model: str, arm: str):
        return scored(Path(args.judged_dir) / f"{model}_{arm}_judged" /
                      "metrics_v1_judged.json", "scored")

    def canon_det(model: str):
        rows = scored(Path(args.untreated_det_dir) / f"{model}_canonical_pv0" /
                      "metrics_v1.json", "scored")
        assert rows, f"no canonical det arm for {model}"
        return rows

    def canon_judged(model: str):
        rows = scored(Path(args.untreated_judged_dir) /
                      f"{model}_canonical_pv0_judged" /
                      "metrics_v1_judged.json", "scored")
        assert rows, f"no canonical judged arm for {model}"
        return rows

    out = {
        "meta": {
            "layer": "memtoc-divergence-protocol-summary-v1",
            "primary": "judged (a single judge pass)",
            "sensitivity": "det (scorer v1.3)",
            "families": {
                "A": "far_keep: 5 models, keep on arb, Holm inside the family",
                "B": "protocol_keep: anchor llamai, a single contrast",
            },
            "arms": "pv0 only (far and the protocol are defined on pv0 alone)",
            "stats": "imported from scripts.build_paired_treatment_summary "
                     "(bootstrap n=1000, sign-flip n=10000, seed 0)",
            "code_commit": os.environ.get("KCB_CODE_COMMIT"),
        },
        "judged": build_layer(load_judged, canon_judged),
        "det": build_layer(load_det, canon_det),
    }

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[distance-summary] -> {dest}")
    for model, entry in out["judged"]["far"].items():
        c = entry.get("keep", {}).get("contrast", {})
        if "delta" in c:
            holm = out["judged"]["far_keep_holm"][model]["p_holm"]
            print(f"[distance-summary] far {model:8} keep Δ {c['delta']:+.4f} "
                  f"{c['ci95']} holm {holm:.4f} n={c['n_questions']}")
    p = out["judged"]["protocol"].get(PROTOCOL_MODEL, {}).get("keep", {}).get("contrast", {})
    if "delta" in p:
        print(f"[distance-summary] protocol llamai keep Δ {p['delta']:+.4f} "
              f"{p['ci95']} p {p['p_signflip']:.4f} n={p['n_questions']}")


if __name__ == "__main__":
    main()
