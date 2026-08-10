"""Paired wave-1 statistics on the canon (542 questions).

Why the pairs are valid: the forced closed-book pass (the map of cells) is
shared across (model, pv) — the canonical, presentation and control arms of one
model and one pv have IDENTICAL qkey sets in every cell (asserted at load). The
contrast "arm vs canonical" on a shared cell is therefore an exact paired
design on identical questions. Cross-model comparisons run on the intersection
of the arb cells of two models, paired by qkey.

Statistics (pre-registered BEFORE the aggregates were read):
- delta = mean(arm - canonical) over the paired units;
- CI95 — percentile bootstrap n_boot=1000 over qkey clusters (parameters as in
  memtoc.metrics.bootstrap_ci; RNG numpy default_rng(0));
- p — sign-flip permutation over qkey clusters, n_perm=10000, seed=0,
  two-sided, with the +1 correction;
- Holm INSIDE the pre-declared families, per pv and pooled (pooled: units
  (qkey,pv), cluster = qkey):
    A frames_keep    — 5 models x {toolns, ragsnip};
    B controls_keep  — 4 instructs x {warn, prior, flag};
    C base_vs_instr_keep — base vs each instruct (arb intersection per pair).
- Secondary readings (not Holm-gated): prior_bw, tool_gold_follow, te_abstain,
  nc_CRA, CRA_arb — delta plus CI, with p left ungated.

Layers: primary = judged (a single judge pass), sensitivity = det (scorer
v1.3). The point keep values of the canonical arms are checked with an assert
against the canonical summary (--canon-summary).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import scripts.build_judged_summary as bjs
from scripts.build_common_support_summary import git_commit, sha256

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["base", "llamai", "qwen", "gemma", "mistral"]
INSTRUCTS = ["llamai", "qwen", "gemma", "mistral"]
FRAMES = ["presentation_toolns", "presentation_ragsnip"]
CONTROLS = ["control_warn", "control_prior", "control_flag"]
PVS = ["0", "1", "2"]
N_BOOT, N_PERM, SEED = 1000, 10000, 0

# metric -> (condition, cell | None=all, predicate)
METRICS = {
    "keep": ("tool_wrong", "arb", lambda s: s["outcome"] == "kept_memory"),
    "CRA_arb": ("tool_wrong", "arb", lambda s: s["final_correct"]),
    "prior_bw": ("tool_wrong", "both_wrong", lambda s: s["followed_tool"]),
    "tool_gold_follow": ("tool_right", "tool_gold", lambda s: s["followed_tool"]),
    "nc_CRA": ("no_conflict", "tool_gold", lambda s: s["final_correct"]),
    "te_abstain": ("tool_error", None, lambda s: s["abstain"]),
}
PRIMARY_METRIC = "keep"


def qvals(scored: list[dict], metric: str) -> dict[str, bool]:
    """qkey -> the metric's value on its cell; qkey is unique within a cell."""
    cond, cell, pred = METRICS[metric]
    rows = [s for s in scored if s["condition"] == cond
            and (cell is None or s["cell"] == cell)]
    out = {s["qkey"]: bool(pred(s)) for s in rows}
    assert len(out) == len(rows), f"qkey is not unique in {cond}/{cell}"
    return out


def paired_block(clusters: dict[str, list[float]]) -> dict:
    """Delta + cluster bootstrap CI + sign-flip p (cluster = qkey)."""
    keys = sorted(clusters)
    csum = np.array([sum(clusters[k]) for k in keys])
    cnt = np.array([len(clusters[k]) for k in keys])
    n_units = int(cnt.sum())
    delta = float(csum.sum()) / n_units

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    boots = csum[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    boots.sort()
    ci = [round(float(boots[int(0.025 * N_BOOT)]), 4),
          round(float(boots[min(N_BOOT - 1, int(0.975 * N_BOOT))]), 4)]

    signs = rng.integers(0, 2, size=(N_PERM, len(keys))) * 2 - 1
    perm = (signs @ csum) / n_units
    p = (int((np.abs(perm) >= abs(delta) - 1e-12).sum()) + 1) / (N_PERM + 1)
    return {"delta": round(delta, 4), "ci95": ci,
            "p_signflip": round(p, 6), "n_units": n_units,
            "n_questions": len(keys)}


def contrast_clusters(base_vals: list[dict[str, bool]],
                      treat_vals: list[dict[str, bool]]) -> dict[str, list[float]]:
    """Clusters of paired differences by qkey; the lists hold one pv per input."""
    clusters: dict[str, list[float]] = {}
    for a, b in zip(base_vals, treat_vals):
        assert set(a) == set(b), "the qkey sets of the cell diverged (is the forced pass shared?)"
        for q in a:
            clusters.setdefault(q, []).append(float(b[q]) - float(a[q]))
    return clusters


def with_holm(family: dict[str, dict]) -> dict[str, dict]:
    adj = bjs.holm({k: v["p_signflip"] for k, v in family.items()})
    for k in family:
        family[k]["p_holm"] = round(adj[k], 6)
    return family


def build_layer(load_arm) -> dict:
    """The whole paired layer on top of the arm loader load_arm(model, arm)."""
    # qvals for each (model, arm, pv, metric)
    V: dict[tuple, dict[str, bool]] = {}
    for m in MODELS:
        arm_list = [f"canonical_pv{pv}" for pv in PVS] + \
            [f"{fr}_pv{pv}" for fr in FRAMES for pv in PVS] + \
            ([f"{c}_pv{pv}" for c in CONTROLS for pv in PVS]
             if m != "base" else [])
        for arm in arm_list:
            scored = load_arm(m, arm)
            for metric in METRICS:
                V[(m, arm, metric)] = qvals(scored, metric)

    def contrasts(m: str, treat_base: str, metric: str) -> dict[str, dict]:
        # the presentation arms have no tool_error/no_conflict conditions
        if any(not V[(m, f"{treat_base}_pv{pv}", metric)] for pv in PVS):
            return {"_absent": "condition/cell not present in this arm"}
        out = {}
        for pv in PVS:
            cl = contrast_clusters(
                [V[(m, f"canonical_pv{pv}", metric)]],
                [V[(m, f"{treat_base}_pv{pv}", metric)]])
            out[f"pv{pv}"] = paired_block(cl)
        out["pooled"] = paired_block(contrast_clusters(
            [V[(m, f"canonical_pv{pv}", metric)] for pv in PVS],
            [V[(m, f"{treat_base}_pv{pv}", metric)] for pv in PVS]))
        return out

    layer: dict = {"frames": {}, "controls": {}, "base_vs_instructs": {}}
    for m in MODELS:
        layer["frames"][m] = {fr: {metric: contrasts(m, fr, metric)
                                   for metric in METRICS} for fr in FRAMES}
    for m in INSTRUCTS:
        layer["controls"][m] = {c: {metric: contrasts(m, c, metric)
                                    for metric in METRICS} for c in CONTROLS}

    # C: base vs the instructs on the arb intersection (canonical), delta =
    # base - instr
    for m in INSTRUCTS:
        out = {}
        per_pv_clusters = []
        for pv in PVS:
            vb = V[("base", f"canonical_pv{pv}", PRIMARY_METRIC)]
            vi = V[(m, f"canonical_pv{pv}", PRIMARY_METRIC)]
            common = sorted(set(vb) & set(vi))
            cl = {q: [float(vb[q]) - float(vi[q])] for q in common}
            per_pv_clusters.append(cl)
            out[f"pv{pv}"] = paired_block(cl)
        pooled: dict[str, list[float]] = {}
        for cl in per_pv_clusters:
            for q, d in cl.items():
                pooled.setdefault(q, []).extend(d)
        out["pooled"] = paired_block(pooled)
        layer["base_vs_instructs"][f"base_vs_{m}"] = out

    # Holm inside the pre-declared families, per slice
    layer["holm_families"] = {}
    for sl in [f"pv{pv}" for pv in PVS] + ["pooled"]:
        fam_a = {f"{m}|{fr}": dict(layer["frames"][m][fr][PRIMARY_METRIC][sl])
                 for m in MODELS for fr in FRAMES}
        fam_b = {f"{m}|{c}": dict(layer["controls"][m][c][PRIMARY_METRIC][sl])
                 for m in INSTRUCTS for c in CONTROLS}
        fam_c = {k: dict(v[sl])
                 for k, v in layer["base_vs_instructs"].items()}
        layer["holm_families"][sl] = {
            "A_frames_keep": with_holm(fam_a),
            "B_controls_keep": with_holm(fam_b),
            "C_base_vs_instr_keep": with_holm(fam_c),
        }
    return layer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-dir", default="artifacts/E107_prep")
    ap.add_argument("--judged-dir", default="results/scored_episodes")
    ap.add_argument("--canon-summary", default=None,
                    help="summary_judged_qminus.json, for checking the point keep values")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    det_dir, judged_dir = ROOT / args.det_dir, ROOT / args.judged_dir

    def load_det(m: str, arm: str) -> list[dict]:
        return bjs.load_scored(det_dir / f"{m}_{arm}/metrics_v1.json")

    def load_judged(m: str, arm: str) -> list[dict]:
        return bjs.load_scored(judged_dir / f"{m}_{arm}_judged/metrics_v1_judged.json")

    summary = {
        "judged": build_layer(load_judged),
        "det": build_layer(load_det),
        "_meta": {
            "built_by": "scripts/build_paired_treatment_summary.py",
            "design": "paired contrasts on identical qkey; the forced closed-book pass is "
                      "shared across (model,pv), so the cells are identical between arms",
            "stats": f"delta; cluster percentile bootstrap n={N_BOOT}; "
                     f"sign-flip p n={N_PERM}; RNG numpy default_rng({SEED}); "
                     "Holm inside the families A/B/C, over the slices pv0/1/2/pooled",
            "delta_direction": {
                "frames/controls": "arm minus canonical (same model and pv)",
                "base_vs_instructs": "base minus instruct (keep on the arb intersection)"},
            "primary_layer": "judged",
            "dirs": {"det": args.det_dir, "judged": args.judged_dir},
            "manifests": {
                "det": sha256(det_dir / "manifest.json"),
                "judged": sha256(judged_dir / "manifest.json")},
            # the server receives the code as files rather than as a commit
            "code_commit": os.environ.get("KCB_CODE_COMMIT") or git_commit(ROOT),
        },
    }

    # check the point keep of the canonical arms against the canonical summary
    if args.canon_summary:
        canon = json.loads((ROOT / args.canon_summary).read_text())
        for m in MODELS:
            for pv in PVS:
                arm = f"canonical_pv{pv}"
                scored = load_judged(m, arm)
                v = qvals(scored, PRIMARY_METRIC)
                keep = sum(v.values()) / len(v)
                ref = canon[m][arm]["keep"]
                assert abs(keep - ref) < 1e-9, f"{m}/{arm}: {keep} != {ref}"
        summary["_meta"]["canon_summary_check"] = "PASS: " + args.canon_summary

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {out}")

    # digest, for checking
    for sl in ("pooled", "pv0"):
        fams = summary["judged"]["holm_families"][sl]
        for fam, rows in fams.items():
            sig = {k: v["delta"] for k, v in rows.items() if v["p_holm"] < 0.05}
            print(f"[judged {sl}] {fam}: Holm<.05 -> {sig}")


if __name__ == "__main__":
    main()
