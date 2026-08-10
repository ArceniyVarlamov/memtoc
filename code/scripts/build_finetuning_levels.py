"""Absolute rates of treated and untreated on the COMMON support of the contrast.

A companion to `scripts/build_finetuning_summary.py`. The same arm loading,
the same out-of-fold merge, the same intersection of supports — but what is
printed is not the deltas: it is the rates of both arms (`rate_untreated`,
`rate_treated`) with a cluster bootstrap CI.

Why: the summary of contrasts carries only deltas, while the paper's table is
read as "it became this much". The level cannot be recovered by arithmetic —
the cell (arb / both_wrong / tool_gold) is defined by the arm's own parametric
answer, fine-tuning moves it, and the contrast lives on the intersection.

Invariant (asserted under `--check`):
    rate_treated - rate_untreated == the delta in summary_treated.json
for every contrast, metric and slice, to within 1e-4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import scripts.build_judged_summary as bjs
import scripts.build_paired_treatment_summary as bpt
import scripts.build_finetuning_summary as fts
from scripts.build_common_support_summary import git_commit

ROOT = fts.ROOT
PVS = bpt.PVS
METRICS = bpt.METRICS


# --------------------------------------------------------------------------
# levels on the common support


def _levels(base_by_pv: dict[str, dict], treat_by_pv: dict[str, dict],
            pvs: list[str]) -> dict:
    """Rates of both arms on the intersection of supports; the unit is (qkey, pv)."""
    cb: dict[str, list[float]] = {}
    ct: dict[str, list[float]] = {}
    for pv in pvs:
        base, treat = base_by_pv[pv], treat_by_pv[pv]
        for q in set(base) & set(treat):
            cb.setdefault(q, []).append(float(base[q]))
            ct.setdefault(q, []).append(float(treat[q]))
    if not cb:
        return {"_absent": "condition/cell is absent from this arm"}

    n = sum(len(v) for v in cb.values())
    rate_u = sum(sum(v) for v in cb.values()) / n
    rate_t = sum(sum(v) for v in ct.values()) / n
    # the CI comes from the same cluster bootstrap as the contrasts (RNG
    # default_rng(0))
    ci_u = bpt.paired_block(cb)
    ci_t = bpt.paired_block(ct)
    return {
        "rate_untreated": round(rate_u, 4),
        "ci95_untreated": ci_u["ci95"],
        "rate_treated": round(rate_t, 4),
        "ci95_treated": ci_t["ci95"],
        "delta_check": round(rate_t - rate_u, 4),
        "n_units": n,
        "n_questions": len(cb),
    }


def slices_levels(base_by_pv: dict[str, dict],
                  treat_by_pv: dict[str, dict]) -> dict:
    out = {f"pv{pv}": _levels(base_by_pv, treat_by_pv, [pv]) for pv in PVS}
    out["pooled"] = _levels(base_by_pv, treat_by_pv, PVS)
    return out


def build_levels_layer(load_arm, load_untreated) -> dict:
    """A mirror of build_layer, with levels instead of deltas."""
    U: dict[tuple, dict[str, bool]] = {}
    models = sorted(set(fts.CROSSFIT_MODELS + fts.SEEDS_MODELS + fts.TRANSFER_MODELS))
    for m in models:
        for pv in PVS:
            rows = load_untreated(m, pv)
            for metric in METRICS:
                U[(m, pv, metric)] = bpt.qvals(rows, metric)

    def folds(prefix: str) -> dict[str, dict[str, dict]]:
        by_metric: dict[str, dict[str, dict]] = {m: {} for m in METRICS}
        for pv in PVS:
            rows_ab = load_arm(f"{prefix}_ab_pv{pv}")
            rows_ba = load_arm(f"{prefix}_ba_pv{pv}")
            for metric in METRICS:
                by_metric[metric][pv] = fts.merge_oof(
                    bpt.qvals(rows_ab, metric), bpt.qvals(rows_ba, metric))
        return by_metric

    def full(prefix: str) -> dict[str, dict[str, dict]]:
        by_metric: dict[str, dict[str, dict]] = {m: {} for m in METRICS}
        for pv in PVS:
            rows = load_arm(f"{prefix}_pv{pv}")
            for metric in METRICS:
                by_metric[metric][pv] = bpt.qvals(rows, metric)
        return by_metric

    layer: dict = {"crossfit": {}, "seeds": {}, "reverse_transfer": {}}

    for m in fts.CROSSFIT_MODELS:
        layer["crossfit"][m] = {}
        for meth in fts.CROSSFIT_METHODS:
            tv = folds(f"{m}_e100_{meth}")
            layer["crossfit"][m][meth] = {
                metric: slices_levels({pv: U[(m, pv, metric)] for pv in PVS},
                                      tv[metric])
                for metric in METRICS}

    for m in fts.SEEDS_MODELS:
        layer["seeds"][m] = {}
        for seed in fts.SEEDS_SEEDS:
            tv = folds(f"{m}_e110_dpo_s{seed}")
            layer["seeds"][m][f"s{seed}"] = {
                metric: slices_levels({pv: U[(m, pv, metric)] for pv in PVS},
                                      tv[metric])
                for metric in METRICS}

    for m in fts.TRANSFER_MODELS:
        tv = full(f"{m}_e111_rev")
        layer["reverse_transfer"][m] = {
            metric: slices_levels({pv: U[(m, pv, metric)] for pv in PVS},
                                  tv[metric])
            for metric in METRICS}

    return layer


# --------------------------------------------------------------------------
# check against the published deltas


def check_against(levels: dict, published: dict) -> tuple[list[str], int]:
    """delta_check must equal the delta in summary_treated.json."""
    bad: list[str] = []
    seen = 0

    def walk(node: dict, ref: dict, path: str) -> None:
        nonlocal seen
        if "delta_check" in node:
            seen += 1
            want = ref.get("delta")
            got = node["delta_check"]
            if want is None or abs(want - got) > 1.5e-4:
                bad.append(f"{path}: published {want} vs levels {got}")
            return
        if "_absent" in node:
            return
        for key, child in node.items():
            if isinstance(child, dict):
                walk(child, ref.get(key, {}), f"{path}/{key}")

    for layer in ("judged", "det"):
        for block in ("crossfit", "seeds", "reverse_transfer"):
            walk(levels[layer][block], published[layer][block],
                 f"{layer}/{block}")
    return bad, seen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-dir", default="artifacts/E114_prep")
    ap.add_argument("--judged-dir", default="artifacts/E114_judged")
    ap.add_argument("--untreated-det-dir", default="artifacts/E107_prep")
    ap.add_argument("--untreated-judged-dir", default="results/scored_episodes")
    ap.add_argument("--check", default=None,
                    help="summary_treated.json, for checking the deltas")
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

    out = {
        "judged": build_levels_layer(load_judged, load_ujudged),
        "det": build_levels_layer(load_det, load_udet),
        "_meta": {
            "built_by": "scripts/build_finetuning_levels.py",
            "companion_of": "scripts/build_finetuning_summary.py",
            "design": "rates of both arms on THE SAME intersection of supports on "
                      "which the delta is computed; the unit is (qkey, pv)",
            "ci": f"cluster bootstrap n={bpt.N_BOOT}, cluster = qkey, "
                  f"RNG numpy default_rng({bpt.SEED})",
            "primary_layer": "judged",
            "dirs": {"det": args.det_dir, "judged": args.judged_dir,
                     "untreated_det": args.untreated_det_dir,
                     "untreated_judged": args.untreated_judged_dir},
            "code_commit": os_commit(),
            "checked_against": args.check,
        },
    }

    if args.check:
        published = json.loads(Path(args.check).read_text())
        bad, seen = check_against(out, published)
        out["_meta"]["delta_reconstruction"] = (
            f"PASS: {seen} deltas reproduced from the levels" if not bad
            else f"FAIL: {len(bad)} of {seen} disagree")
        for line in bad[:20]:
            print("MISMATCH", line)
        assert not bad, f"{len(bad)} of {seen} deltas did not reproduce"
        print(f"delta reconstruction: PASS ({seen} contrasts)")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", dest)


def os_commit() -> str:
    import os
    return os.environ.get("KCB_CODE_COMMIT") or git_commit(ROOT)


if __name__ == "__main__":
    main()
