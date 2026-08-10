"""Paired statistics for the new treated arms (R6/R7/R9/R12) on the canon.

A reading layer over the deterministic rescore plus a single judge pass.
Pre-registration was written BEFORE any aggregate was read.

What is new here compared with the paired-treatment module:

1. **Out-of-fold merge.** Fold arms are named
   `<model>_<exp>_<method>[_s<seed>]_<ab|ba>_pv<k>`, where `ab` means trained
   on fold A and evaluated on the held-out fold B, and `ba` is the mirror. The
   treated map (model, method, pv) is the union of the two maps. Asserted: the
   folds are disjoint by qkey, and their union over ALL qkey equals the support
   of the untreated arm (542 questions).

2. **Common support instead of identical cells.** Previously the arms of one
   model shared the forced closed-book pass, so the cells matched bit-for-bit
   and `contrast_clusters` asserted set equality. That is not possible here:
   the cell (arb / both_wrong / tool_gold) is defined by the arm's OWN
   parametric answer, and fine-tuning moves it (measured on llamai/pv0: arb 161
   for treated against 162 for untreated, intersection 159). The contrast is
   computed on the intersection, and the drift of the support is printed beside
   the effect (`n_treated_only`, `n_untreated_only`).

The statistics are imported unchanged (cluster bootstrap n=1000, sign-flip
n=10000, RNG default_rng(0), Holm inside the families), and the default path is
not touched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import scripts.build_judged_summary as bjs
import scripts.build_paired_treatment_summary as bpt
from scripts.build_common_support_summary import git_commit, sha256

ROOT = Path(__file__).resolve().parent.parent

PVS = bpt.PVS
METRICS = bpt.METRICS
PRIMARY = bpt.PRIMARY_METRIC
PRESERVE = "tool_gold_follow"

CROSSFIT_MODELS = ["llamai", "qwen", "gemma", "mistral"]
CROSSFIT_METHODS = ["sft", "dpo"]
SEEDS_MODELS = ["llamai", "gemma"]
SEEDS_SEEDS = ["42", "20260719"]
TRANSFER_MODELS = ["llamai", "gemma"]
SCALE70B_ARM = "llama70b_e112"
SCALE70B_REFERENCE = "llamai"  # cross-model base of the spectrum (not a treatment)


# --------------------------------------------------------------------------
# supports and contrasts


def sha_or_none(path: Path) -> str | None:
    """The treated prep directory has no manifest.json — do not fail on that."""
    return sha256(path) if path.exists() else None


def merge_oof(part_a: dict[str, bool], part_b: dict[str, bool]) -> dict[str, bool]:
    """Union of the two fold maps; the qkey intersection must be empty."""
    dup = set(part_a) & set(part_b)
    assert not dup, f"the folds overlap on {len(dup)} qkey (out-of-fold leak)"
    return {**part_a, **part_b}


def contrast_common(base_vals: list[dict[str, bool]],
                    treat_vals: list[dict[str, bool]]
                    ) -> tuple[dict[str, list[float]], int, int]:
    """Clusters of paired differences on the intersection of supports (cluster = qkey)."""
    clusters: dict[str, list[float]] = {}
    only_treated = only_untreated = 0
    for base, treat in zip(base_vals, treat_vals):
        only_treated += len(set(treat) - set(base))
        only_untreated += len(set(base) - set(treat))
        for q in set(base) & set(treat):
            clusters.setdefault(q, []).append(float(treat[q]) - float(base[q]))
    return clusters, only_treated, only_untreated


def block(base_vals, treat_vals) -> dict:
    clusters, only_t, only_u = contrast_common(base_vals, treat_vals)
    if not clusters:
        return {"_absent": "condition/cell is absent from this arm"}
    out = bpt.paired_block(clusters)
    out["n_treated_only"] = only_t
    out["n_untreated_only"] = only_u
    return out


def point_block(vals: list[dict[str, bool]]) -> dict:
    """Point rate plus a cluster bootstrap CI (no p: this is not a contrast)."""
    clusters: dict[str, list[float]] = {}
    for v in vals:
        for q, x in v.items():
            clusters.setdefault(q, []).append(float(x))
    if not clusters:
        return {"_absent": "condition/cell is absent from this arm"}
    out = bpt.paired_block(clusters)
    out.pop("p_signflip")
    out["rate"] = out.pop("delta")
    return out


def slices(base_by_pv: dict[str, dict], treat_by_pv: dict[str, dict]) -> dict:
    """pv0/pv1/pv2 plus pooled (unit = (qkey,pv), cluster = qkey)."""
    out = {f"pv{pv}": block([base_by_pv[pv]], [treat_by_pv[pv]]) for pv in PVS}
    out["pooled"] = block([base_by_pv[pv] for pv in PVS],
                          [treat_by_pv[pv] for pv in PVS])
    return out


# --------------------------------------------------------------------------
# assembling the layer


def build_layer(load_arm, load_untreated) -> dict:
    """load_arm(arm_name) and load_untreated(model, pv) -> a list of scored rows."""

    def vals(rows, metric):
        return bpt.qvals(rows, metric)

    # untreated: (model, pv, metric) -> map
    U: dict[tuple, dict[str, bool]] = {}
    untreated_models = sorted(set(CROSSFIT_MODELS + SEEDS_MODELS + TRANSFER_MODELS
                                  + [SCALE70B_REFERENCE]))
    for m in untreated_models:
        for pv in PVS:
            rows = load_untreated(m, pv)
            for metric in METRICS:
                U[(m, pv, metric)] = vals(rows, metric)

    support: dict[str, dict] = {}

    def treated_folds(prefix: str, model: str) -> dict[str, dict[str, dict]]:
        """{metric: {pv: map}} with the out-of-fold merge and a support check."""
        by_metric: dict[str, dict[str, dict]] = {m: {} for m in METRICS}
        for pv in PVS:
            rows_ab = load_arm(f"{prefix}_ab_pv{pv}")
            rows_ba = load_arm(f"{prefix}_ba_pv{pv}")
            qa = {r["qkey"] for r in rows_ab}
            qb = {r["qkey"] for r in rows_ba}
            qu = {r["qkey"] for r in load_untreated(model, pv)}
            assert not (qa & qb), f"{prefix} pv{pv}: the folds overlap"
            merged = qa | qb
            assert merged == qu, (
                f"{prefix} pv{pv}: out-of-fold union {len(merged)} != "
                f"untreated support {len(qu)}")
            support[f"{prefix}_pv{pv}"] = {
                "fold_ab": len(qa), "fold_ba": len(qb),
                "oof_union": len(merged), "untreated": len(qu)}
            for metric in METRICS:
                by_metric[metric][pv] = merge_oof(vals(rows_ab, metric),
                                                  vals(rows_ba, metric))
        return by_metric

    def treated_full(prefix: str) -> dict[str, dict[str, dict]]:
        by_metric: dict[str, dict[str, dict]] = {m: {} for m in METRICS}
        for pv in PVS:
            rows = load_arm(f"{prefix}_pv{pv}")
            for metric in METRICS:
                by_metric[metric][pv] = vals(rows, metric)
        return by_metric

    layer: dict = {"crossfit": {}, "seeds": {}, "reverse_transfer": {}, "scale_70b": {}}

    # R6: crossfit, 4 models x {sft,dpo}
    for m in CROSSFIT_MODELS:
        layer["crossfit"][m] = {}
        for meth in CROSSFIT_METHODS:
            tv = treated_folds(f"{m}_e100_{meth}", m)
            layer["crossfit"][m][meth] = {
                metric: slices({pv: U[(m, pv, metric)] for pv in PVS},
                               tv[metric])
                for metric in METRICS}

    # R7: seeds, 2 anchors x 2 seeds
    for m in SEEDS_MODELS:
        layer["seeds"][m] = {}
        for seed in SEEDS_SEEDS:
            tv = treated_folds(f"{m}_e110_dpo_s{seed}", m)
            layer["seeds"][m][f"s{seed}"] = {
                metric: slices({pv: U[(m, pv, metric)] for pv in PVS},
                               tv[metric])
                for metric in METRICS}

    # R12: reverse transfer, the full canon (no out-of-fold split)
    for m in TRANSFER_MODELS:
        tv = treated_full(f"{m}_e111_rev")
        layer["reverse_transfer"][m] = {
            metric: slices({pv: U[(m, pv, metric)] for pv in PVS}, tv[metric])
            for metric in METRICS}

    # R9: 70B — a point on the scale, NOT a treatment contrast
    tv70 = treated_full(SCALE70B_ARM)
    layer["scale_70b"] = {
        "point": {metric: {**{f"pv{pv}": point_block([tv70[metric][pv]])
                              for pv in PVS},
                           "pooled": point_block([tv70[metric][pv]
                                                  for pv in PVS])}
                  for metric in METRICS},
        f"vs_{SCALE70B_REFERENCE}_untreated": {
            metric: slices({pv: U[(SCALE70B_REFERENCE, pv, metric)] for pv in PVS},
                           tv70[metric])
            for metric in METRICS},
        "_note": ("there is no untreated 70B arm: this is a point on the scale plus "
                  "a cross-model contrast on a common support, not an effect "
                  "of fine-tuning"),
    }

    # Holm inside the pre-declared families, for each slice
    layer["holm_families"] = {}
    for sl in [f"pv{pv}" for pv in PVS] + ["pooled"]:
        fam_t1 = {f"{m}|{meth}": dict(layer["crossfit"][m][meth][PRIMARY][sl])
                  for m in CROSSFIT_MODELS for meth in CROSSFIT_METHODS}
        fam_t2 = {f"{m}|{meth}": dict(layer["crossfit"][m][meth][PRESERVE][sl])
                  for m in CROSSFIT_MODELS for meth in CROSSFIT_METHODS}
        fam_t3 = {f"{m}|s{seed}": dict(layer["seeds"][m][f"s{seed}"][PRIMARY][sl])
                  for m in SEEDS_MODELS for seed in SEEDS_SEEDS}
        fam_t4 = {m: dict(layer["reverse_transfer"][m][PRIMARY][sl]) for m in TRANSFER_MODELS}
        layer["holm_families"][sl] = {
            "T1_e100_keep": bpt.with_holm(fam_t1),
            "T2_e100_toolgold": bpt.with_holm(fam_t2),
            "T3_e110_seeds_keep": bpt.with_holm(fam_t3),
            "T4_e111_reverse_keep": bpt.with_holm(fam_t4),
        }

    # DPO criterion: keep rises (Holm<.05) with no significant loss of
    # tool_gold_follow (pre-registered)
    layer["dpo_criterion"] = {}
    for sl in [f"pv{pv}" for pv in PVS] + ["pooled"]:
        fams = layer["holm_families"][sl]
        crit = {}
        for m in CROSSFIT_MODELS:
            for meth in CROSSFIT_METHODS:
                k = f"{m}|{meth}"
                keep = fams["T1_e100_keep"][k]
                gold = fams["T2_e100_toolgold"][k]
                gain = keep.get("delta", 0) > 0 and keep.get("p_holm", 1) < 0.05
                loss = gold.get("delta", 0) < 0 and gold.get("p_holm", 1) < 0.05
                crit[k] = {"keep_gain": bool(gain),
                           "toolgold_loss": bool(loss),
                           "criterion_met": bool(gain and not loss),
                           "keep_delta": keep.get("delta"),
                           "keep_p_holm": keep.get("p_holm"),
                           "toolgold_delta": gold.get("delta"),
                           "toolgold_p_holm": gold.get("p_holm")}
        layer["dpo_criterion"][sl] = crit

    # R7: agreement of directions across seeds (no new test, as declared)
    layer["seed_agreement"] = {}
    for m in SEEDS_MODELS:
        row = {"primary_s20260705": layer["crossfit"][m]["dpo"][PRIMARY]["pooled"]}
        for seed in SEEDS_SEEDS:
            row[f"s{seed}"] = layer["seeds"][m][f"s{seed}"][PRIMARY]["pooled"]
        deltas = [v["delta"] for v in row.values() if "delta" in v]
        cis = [v["ci95"] for v in row.values() if "ci95" in v]
        lo, hi = max(c[0] for c in cis), min(c[1] for c in cis)
        row["_agreement"] = {
            "deltas": deltas,
            "all_same_sign": all(d > 0 for d in deltas) or all(d < 0 for d in deltas),
            "ci_overlap": lo <= hi,
            "ci_overlap_interval": [round(lo, 4), round(hi, 4)] if lo <= hi else None,
        }
        layer["seed_agreement"][m] = row

    layer["oof_support"] = support
    return layer


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-dir", default="artifacts/E114_prep")
    ap.add_argument("--judged-dir", default="artifacts/E114_judged")
    ap.add_argument("--untreated-det-dir", default="artifacts/E107_prep")
    ap.add_argument("--untreated-judged-dir", default="results/scored_episodes")
    ap.add_argument("--canon-summary", default=None,
                    help="summary_judged_qminus.json, for checking untreated keep")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    det, judged = ROOT / args.det_dir, ROOT / args.judged_dir
    udet, ujudged = ROOT / args.untreated_det_dir, ROOT / args.untreated_judged_dir

    def load_det(arm: str) -> list[dict]:
        return bjs.load_scored(det / arm / "metrics_v1.json")

    def load_judged(arm: str) -> list[dict]:
        return bjs.load_scored(judged / f"{arm}_judged" / "metrics_v1_judged.json")

    def load_udet(m: str, pv: str) -> list[dict]:
        return bjs.load_scored(udet / f"{m}_canonical_pv{pv}" / "metrics_v1.json")

    def load_ujudged(m: str, pv: str) -> list[dict]:
        return bjs.load_scored(
            ujudged / f"{m}_canonical_pv{pv}_judged" / "metrics_v1_judged.json")

    # the arms of both layers must agree
    det_arms = {p.name for p in det.iterdir() if p.is_dir()}
    judged_arms = {p.name[:-len("_judged")] for p in judged.iterdir()
                   if p.is_dir() and p.name.endswith("_judged")}
    assert det_arms == judged_arms, (
        f"the arm lists diverged: det only {sorted(det_arms - judged_arms)}, "
        f"judged only {sorted(judged_arms - det_arms)}")

    summary = {
        "judged": build_layer(load_judged, load_ujudged),
        "det": build_layer(load_det, load_udet),
        "_meta": {
            "built_by": "scripts/build_finetuning_summary.py",
            "prereg": "pre-registered 2026-07-25, two entries",
            "design": "out-of-fold merge of the fold arms; paired contrasts against "
                      "untreated on the INTERSECTION of supports (the cell is "
                      "defined by the arm's own parametric answer)",
            "stats": f"delta; cluster bootstrap n={bpt.N_BOOT}; sign-flip p "
                     f"n={bpt.N_PERM}; RNG numpy default_rng({bpt.SEED}); Holm "
                     "inside the families T1-T4, over the slices pv0/1/2/pooled",
            "delta_direction": "treated minus untreated (same model and pv)",
            "primary_layer": "judged",
            "n_arms": len(det_arms),
            "dirs": {"det": args.det_dir, "judged": args.judged_dir,
                     "untreated_det": args.untreated_det_dir,
                     "untreated_judged": args.untreated_judged_dir},
            "manifests": {"det": sha_or_none(det / "manifest.json"),
                          "judged": sha_or_none(judged / "manifest.json"),
                          "untreated_det": sha_or_none(udet / "manifest.json"),
                          "untreated_judged": sha_or_none(
                              ujudged / "manifest.json")},
            "code_commit": os.environ.get("KCB_CODE_COMMIT") or git_commit(ROOT),
        },
    }

    if args.canon_summary:
        canon = json.loads((ROOT / args.canon_summary).read_text())
        checked = 0
        for m in sorted(set(CROSSFIT_MODELS + SEEDS_MODELS + TRANSFER_MODELS
                            + [SCALE70B_REFERENCE])):
            for pv in PVS:
                v = bpt.qvals(load_ujudged(m, pv), PRIMARY)
                keep = sum(v.values()) / len(v)
                ref = canon[m][f"canonical_pv{pv}"]["keep"]
                assert abs(keep - ref) < 1e-9, f"{m}/pv{pv}: {keep} != {ref}"
                checked += 1
        summary["_meta"]["canon_summary_check"] = (
            f"PASS ({checked} arms): {args.canon_summary}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {out}")

    # digest to the log
    for sl in ("pooled", "pv0"):
        fams = summary["judged"]["holm_families"][sl]
        for fam, rows in fams.items():
            sig = {k: v["delta"] for k, v in rows.items()
                   if v.get("p_holm", 1) < 0.05}
            print(f"[judged {sl}] {fam}: Holm<.05 -> {sig}")
    crit = summary["judged"]["dpo_criterion"]["pooled"]
    met = [k for k, v in crit.items() if v["criterion_met"]]
    print(f"[judged pooled] DPO criterion met: {met}")


if __name__ == "__main__":
    main()
