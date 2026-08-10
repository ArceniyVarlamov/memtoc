"""Cross-model comparisons on a common support of episodes (answer to W2).

W2 of the external review: cells are built per model from each model's own
memory, so the prior spectrum and the keep rates compare DIFFERENT subsamples
of hops. Here the same metrics are recomputed on the intersection of the cells
of all five models ("the same tickets"):
  arb-cap — episodes where the closed-book answer is right for all five (cell
            arb for every model);
  bw-cap  — episodes where it is wrong for all five (cell both_wrong for every
            model).

Cell membership is read from the canonical files (elicitation is NOT
recomputed). The primary set uses the cells of the judged canon (the judge's
corrections are part of the protocol); the prep set (det, closed-book forced)
is reported as a sensitivity check, together with the composition of the
disagreement.

Metrics and statistics reuse the standard aggregator
(scripts.build_judged_summary: load_scored / trivial_map / two_prop_z /
pairwise_family / apply_ctrl_judge) and memtoc.metrics.bootstrap_ci (percentile
episode bootstrap, n_boot=1000, seed=0 — as in the paper). The full per-model
values are recomputed by the same functions and checked with an assert against
the canonical summary (--canon-summary).

Every directory is an argument, so the same command serves the shipped tree
and the earlier canon revisions; only the paths differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

from memtoc.metrics import bootstrap_ci
import scripts.build_judged_summary as bjs

MODELS = bjs.MODELS
CANON_ARM = "near_pv0"
# Planned family for the Finding 2 claim (base is the best Q- defender)
PLANNED_PAIRS = [("base", m) for m in MODELS if m != "base"]
# Canon-significant pairs of the prior spectrum (Finding 1: p_holm=0.038 both)
PRIOR_CLAIM_PAIRS = [("llamai", "mistral"), ("base", "mistral")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(repo: Path) -> str:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    return head + ("+dirty" if dirty else "")


def cell_map(scored: list[dict]) -> dict[str, str]:
    return {s["episode_id"]: s["cell"] for s in scored
            if s["condition"] == "tool_wrong"}


def prop_ci(records: list[dict], pred) -> tuple[float, list, int]:
    """Share of pred(s) over records, plus an episode bootstrap CI (as in the paper)."""
    items = [{"v": bool(pred(s))} for s in records]
    p = sum(i["v"] for i in items) / len(items)
    return round(p, 4), bootstrap_ci(items, "v"), len(items)


def paired_delta_ci(pairs: list[tuple[bool, bool]], n_boot: int = 1000,
                    seed: int = 0) -> list:
    """Episode bootstrap of the paired delta mean(b) - mean(a); the same
    parameters (percentile, 1000, seed 0) as memtoc.metrics.bootstrap_ci.
    """
    n = len(pairs)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        sa = sb = 0
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            sa += a
            sb += b
        deltas.append((sb - sa) / n)
    deltas.sort()
    return [round(deltas[int(0.025 * n_boot)], 4),
            round(deltas[min(n_boot - 1, int(0.975 * n_boot))], 4)]


def model_metrics(judged: dict[str, list[dict]], arb_ids: set[str],
                  bw_ids: set[str], trivial: dict[str, bool]) -> dict:
    """Each model's metrics on the SHARED sets of episodes."""
    out = {}
    for m in MODELS:
        tw = [s for s in judged[m] if s["condition"] == "tool_wrong"]
        arb = [s for s in tw if s["episode_id"] in arb_ids]
        arbq = [s for s in arb if not trivial[s["episode_id"]]]
        bw = [s for s in tw if s["episode_id"] in bw_ids]
        assert all(s["cell"] == "arb" for s in arb), m
        assert all(s["cell"] == "both_wrong" for s in bw), m
        keep, keep_ci, n_arb = prop_ci(arb, lambda s: s["outcome"] == "kept_memory")
        keepq, keepq_ci, n_arbq = prop_ci(arbq, lambda s: s["outcome"] == "kept_memory")
        cra, cra_ci, _ = prop_ci(arb, lambda s: s["final_correct"])
        prior, prior_ci, n_bw = prop_ci(bw, lambda s: s["followed_tool"])
        out[m] = {
            "keep": keep, "keep_ci95": keep_ci, "n_arb": n_arb,
            "keepQ": keepq, "keepQ_ci95": keepq_ci, "n_arbQ": n_arbq,
            "CRA_arb": cra, "CRA_arb_ci95": cra_ci,
            "prior": prior, "prior_ci95": prior_ci, "n_bw": n_bw,
        }
    return out


def pair_intersection_tests(judged: dict[str, list[dict]],
                            cellsets: dict[str, dict[str, set[str]]],
                            pairs: list[tuple[str, str]], cell: str,
                            pred, metric: str,
                            trivial: dict[str, bool] | None = None) -> dict:
    """Every pair on its OWN two-model intersection of the cell (plus the Q-
    filter when trivial). Holm is applied inside the family passed in.
    """
    rows, raw = {}, {}
    for a, b in pairs:
        ids = cellsets[a][cell] & cellsets[b][cell]
        if trivial is not None:
            ids = {e for e in ids if not trivial[e]}
        counts = {}
        for m in (a, b):
            recs = [s for s in judged[m] if s["condition"] == "tool_wrong"
                    and s["episode_id"] in ids]
            counts[m] = (sum(bool(pred(s)) for s in recs), len(recs))
        key = f"{a}_vs_{b}"
        raw[key] = bjs.two_prop_z(*counts[a], *counts[b])
        rows[key] = {
            "n_common": len(ids),
            a: round(counts[a][0] / counts[a][1], 4),
            b: round(counts[b][0] / counts[b][1], 4),
        }
    adj = bjs.holm(raw)
    for key in rows:
        rows[key]["p_raw"] = round(raw[key], 6)
        rows[key]["p_holm"] = round(adj[key], 6)
    return {"metric": metric, "family_size": len(pairs), "pairs": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judged-dir", required=True,
                    help="directory of <model>_<arm>_judged/metrics_v1_judged.json")
    ap.add_argument("--det-dir", required=True,
                    help="directory of det arms <model>_<arm>_qc (prep cells)")
    ap.add_argument("--ctrl-dir", default=None,
                    help="directory of ctrl_dpo (optional DPO block)")
    ap.add_argument("--judge-responses", default=None,
                    help="jsonl of the single judge pass — recipe overlay on ctrl_dpo")
    ap.add_argument("--episodes-suffix", default="",
                    help="suffix of the episode files")
    ap.add_argument("--arm", default=CANON_ARM)
    ap.add_argument("--exclusions", default=None,
                    help="frozen_mapping_v2.json (canon v2)")
    ap.add_argument("--canon-summary", default=None,
                    help="summary_judged_qminus*.json, for the full-sample assert")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.exclusions:
        mp = json.loads(Path(args.exclusions).read_text())
        bjs.EXCLUDED_IH = frozenset(
            bjs._ih(e["episode_id"]) for e in mp["excluded_episodes"])

    judged_dir, det_dir = Path(args.judged_dir), Path(args.det_dir)
    inputs: dict[str, str] = {}

    judged, prep_cells, judged_cells = {}, {}, {}
    for m in MODELS:
        jp = judged_dir / f"{m}_{args.arm}_judged/metrics_v1_judged.json"
        pp = det_dir / f"{m}_{args.arm}_qc/metrics_v1.json"
        judged[m] = bjs.load_scored(jp)
        judged_cells[m] = cell_map(judged[m])
        prep_cells[m] = cell_map(bjs.load_scored(pp))
        inputs[str(jp)] = sha256(jp)
        inputs[str(pp)] = sha256(pp)

    ep_path = bjs.ROOT / bjs.arm_episodes(args.arm, args.episodes_suffix)
    trivial = bjs.trivial_map(ep_path)
    inputs[str(ep_path)] = sha256(ep_path)

    def isect(cells: dict[str, dict[str, str]], cell: str) -> set[str]:
        return set.intersection(
            *({e for e, c in cells[m].items() if c == cell} for m in MODELS))

    arb_j, bw_j = isect(judged_cells, "arb"), isect(judged_cells, "both_wrong")
    arb_p, bw_p = isect(prep_cells, "arb"), isect(prep_cells, "both_wrong")
    # prep arb-cap episodes displaced by the judge's corrections, and where
    # they went
    dropped = {e: {m: judged_cells[m][e] for m in MODELS
                   if judged_cells[m][e] != "arb"}
               for e in sorted(arb_p - arb_j)}
    arb_q = {e for e in arb_j if not trivial[e]}

    cellsets = {m: {c: {e for e, cc in judged_cells[m].items() if cc == c}
                    for c in ("arb", "both_wrong")} for m in MODELS}

    summary: dict = {
        "intersections": {
            "arb_judged": {"n": len(arb_j), "episode_ids": sorted(arb_j)},
            "arb_qminus": {"n": len(arb_q), "episode_ids": sorted(arb_q)},
            "bw": {"n": len(bw_j), "prep_equals_judged": bw_p == bw_j,
                   "episode_ids": sorted(bw_j)},
            "arb_prep_sensitivity": {
                "n": len(arb_p),
                "note": "prep cells (closed-book forced, det); primary = "
                        "judged (the judge's corrections are part of the canonical protocol)",
                "dropped_by_judge": dropped,
            },
            "per_model_n": {m: {c: len(cellsets[m][c])
                                for c in ("arb", "both_wrong")}
                            for m in MODELS},
        },
        "common_support": model_metrics(judged, arb_j, bw_j, trivial),
    }

    # Full per-model subsamples — the same functions as the canonical summary
    full = {m: bjs.arm_stats(judged[m], trivial) for m in MODELS}
    for m in MODELS:
        tw = [s for s in judged[m] if s["condition"] == "tool_wrong"]
        arb = [s for s in tw if s["cell"] == "arb"]
        bw = [s for s in tw if s["cell"] == "both_wrong"]
        full[m]["keep_ci95"] = bootstrap_ci(
            [{"v": s["outcome"] == "kept_memory"} for s in arb], "v")
        full[m]["keepQ_ci95"] = bootstrap_ci(
            [{"v": s["outcome"] == "kept_memory"} for s in arb
             if not trivial[s["episode_id"]]], "v")
        full[m]["CRA_arb"] = round(sum(s["final_correct"] for s in arb) / len(arb), 4)
        full[m]["prior_ci95"] = bootstrap_ci(
            [{"v": s["followed_tool"]} for s in bw], "v")
    if args.canon_summary:
        canon = json.loads(Path(args.canon_summary).read_text())
        inputs[args.canon_summary] = sha256(Path(args.canon_summary))
        for m in MODELS:
            for k, v in canon[m][args.arm].items():
                assert abs(full[m][k] - v) < 1e-9, f"canon check {m}/{k}: {full[m][k]} != {v}"
    summary["full_sample"] = full

    # Pairwise tests on the common support (the same families as the canon)
    cs = summary["common_support"]
    summary["pairwise_tests_common"] = {
        "arm": args.arm,
        "prior_bw": bjs.pairwise_family(
            {m: (round(cs[m]["prior"] * cs[m]["n_bw"]), cs[m]["n_bw"])
             for m in MODELS},
            "source prior on bw∩ (shared episodes)"),
        "keepQ": bjs.pairwise_family(
            {m: (round(cs[m]["keepQ"] * cs[m]["n_arbQ"]), cs[m]["n_arbQ"])
             for m in MODELS},
            "keep strict on Q−(arb∩) (shared episodes)"),
        "keepQ_base_vs_instructs": bjs.pairwise_family(
            {m: (round(cs[m]["keepQ"] * cs[m]["n_arbQ"]), cs[m]["n_arbQ"])
             for m in MODELS},
            "keep strict on Q−(arb∩): base vs each instruct (planned family)",
            pairs=PLANNED_PAIRS),
    }

    # Pairwise two-model intersections (pre-registered branch: n_Q-(arb-cap) <
    # 30)
    summary["pair_intersections"] = {
        "keepQ_planned": pair_intersection_tests(
            judged, cellsets, PLANNED_PAIRS, "arb",
            lambda s: s["outcome"] == "kept_memory",
            "keep strict on pairwise Q−(arb_a∩arb_b)", trivial),
        "prior_claim_pairs": pair_intersection_tests(
            judged, cellsets, PRIOR_CLAIM_PAIRS, "both_wrong",
            lambda s: s["followed_tool"],
            "source prior on pairwise bw_a∩bw_b"),
    }

    # Optional DPO block: the paired effect inside llamai on arb-cap
    if args.ctrl_dir and args.judge_responses:
        from scripts.apply_judge import load_judge
        dpo_path = Path(args.ctrl_dir) / "ctrl_dpo/metrics_v1.json"
        inputs[str(dpo_path)] = sha256(dpo_path)
        inputs[args.judge_responses] = sha256(Path(args.judge_responses))
        judge = load_judge(args.judge_responses)
        dpo = bjs.apply_ctrl_judge(bjs.load_scored(dpo_path), "ctrl_dpo", judge)
        dpo_by_id = {s["episode_id"]: s for s in dpo
                     if s["condition"] == "tool_wrong"}
        unt_by_id = {s["episode_id"]: s for s in judged["llamai"]
                     if s["condition"] == "tool_wrong"}
        # an episode absent from the DPO file cannot enter arb(dpo): the legacy
        # ctrl_dpo was scored on a different episode base than the later canons
        ids = sorted(e for e in arb_j
                     if e in dpo_by_id and dpo_by_id[e]["cell"] == "arb")
        missing = sorted(e for e in arb_j if e not in dpo_by_id)
        pairs = [(unt_by_id[e]["outcome"] == "kept_memory",
                  dpo_by_id[e]["outcome"] == "kept_memory") for e in ids]
        keep_u = sum(a for a, _ in pairs) / len(pairs)
        keep_d = sum(b for _, b in pairs) / len(pairs)
        summary["dpo_common_support"] = {
            "n": len(pairs),
            "note": "arb∩ ∩ arb(ctrl_dpo); untreated = llamai judged; dpo = "
                    "single-judge recipe overlay",
            "keep_untreated": round(keep_u, 4),
            "keep_dpo": round(keep_d, 4),
            "delta": round(keep_d - keep_u, 4),
            "delta_ci95_paired_bootstrap": paired_delta_ci(pairs),
        }
        if missing:
            summary["dpo_common_support"]["arbj_missing_in_dpo"] = {
                "n": len(missing), "episode_ids": missing}

    # Automatic verdicts against the pre-registered criteria
    keepq_pts = {m: cs[m]["keepQ"] for m in MODELS}
    prior_pts = {m: cs[m]["prior"] for m in MODELS}
    instr_order = sorted(("llamai", "qwen", "gemma", "mistral"),
                         key=lambda m: prior_pts[m])
    bottom_two = sorted(prior_pts, key=prior_pts.get)[:2]
    claim_a = max(keepq_pts, key=keepq_pts.get) == "base" and \
        all(keepq_pts["base"] > keepq_pts[m] for m in MODELS if m != "base")
    claim_b = (max(prior_pts, key=prior_pts.get) == "mistral"
               and set(bottom_two) == {"base", "llamai"}
               and instr_order == ["llamai", "qwen", "gemma", "mistral"])
    summary["claims"] = {
        "A_base_strongest_qminus": {
            "criterion": "base is the strict pointwise maximum of keepQ on Q-(arb-cap)",
            "keepQ_points": keepq_pts, "verdict": "survived" if claim_a else "failed",
        },
        "B_prior_spectrum": {
            "criterion": "on bw-cap: mistral is the maximum; {base,llamai} are the lower "
                         "two (their internal order is not fixed); the instructs "
                         "llamai<qwen<gemma<mistral",
            "prior_points": prior_pts, "instruct_order": instr_order,
            "verdict": "survived" if claim_b else "failed",
        },
    }
    if "dpo_common_support" in summary:
        d = summary["dpo_common_support"]
        lo, hi = d["delta_ci95_paired_bootstrap"]
        summary["claims"]["C_dpo_paired"] = {
            "criterion": "delta keep (dpo - untreated) > 0 and the paired CI95 does not "
                         "cover 0",
            "verdict": "survived" if d["delta"] > 0 and lo > 0 else "failed",
        }

    summary["_meta"] = {
        "built_by": "scripts/build_common_support_summary.py",
        "code_commit": git_commit(bjs.ROOT),
        "args": vars(args),
        "bootstrap": "memtoc.metrics.bootstrap_ci: percentile, n_boot=1000, seed=0"
                     " (the paired DPO delta shows the same pattern)",
        "input_sha256": inputs,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {out}")
    print(f"arb∩={len(arb_j)} (Q− {len(arb_q)}), bw∩={len(bw_j)}, "
          f"arb_prep∩={len(arb_p)} (judge dropped {len(dropped)})")
    for m in MODELS:
        c = cs[m]
        print(f"{m:8s} keepQ∩ {c['keepQ']:.3f} {c['keepQ_ci95']} | "
              f"keep∩ {c['keep']:.3f} | CRA∩ {c['CRA_arb']:.3f} | "
              f"prior∩ {c['prior']:.3f} {c['prior_ci95']} "
              f"(full: keepQ {full[m]['keepQ']:.3f} prior {full[m]['prior']:.3f})")
    for name, cl in summary["claims"].items():
        print(f"{name}: {cl['verdict']}")


if __name__ == "__main__":
    main()
