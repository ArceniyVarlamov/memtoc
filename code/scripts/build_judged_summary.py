"""Builder of the v1-full canonical summary (judged) + statistical tests +
the strict slice of §7.

Provenance of the paper's numbers: everything reported from the
judged/control canon must be derivable by this command from the artefacts in
the repository.

Definition of keep — STRICT: P(outcome == "kept_memory") on the arb cell; the
both outcome (the answer matches memory and the tool at once) is separate and
does NOT count towards keep. Q- = arb minus the hops where normalize(gold) is
a substring of normalize(question) (trivial hops, see Finding 2).

Writes:
- <judged-dir>/summary_judged_qminus.json — per-arm prior / keep / keepQ /
  te_abstain plus a pairwise_tests block (two-proportion z, Holm); before
  overwriting it checks its own numbers against the summary already on disk
  (assert).
- <ctrl-dir>/summary_control_strict.json — the §7 table under strict (+flag
  for tracing), the judged canonical anchor, and the strict seed-1 vs seed-2
  deltas.

Every directory is an argument. On the shipped tree:
  python -m scripts.build_judged_summary \
    --judged-dir results/scored_episodes \
    --det-arm-suffix "" --canon-arm canonical_pv0 \
    --ctrl-mode permodel --seed2-dir NONE --base-all-pv
In permodel mode the control table is built per model (canonical +
control_{warn,prior,flag} x pv) from the det AND judged layers directly: a
single judge pass scored every arm, so no recipe overlay is needed.

With --judge-responses instead, the judged §7 rows (DPO included) are computed
as a recipe overlay (tool_wrong neither/other only; the matcher makes the
decision) on the control arms of a single judge run, and the judged anchor
comes from <judged-dir>. --exclusions additionally drops the sibling
episodes of an excluded (instance,hop) in ALL conditions, which is what makes
the denominators of every cell comparable. Without the new flags the behaviour
and the output are byte-for-byte the legacy ones.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

from memtoc.metrics import normalize
from scripts.apply_judge import corrected_outcome, load_judge, parse_extract

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["base", "llamai", "qwen", "gemma", "mistral"]
ARMS = ["near_pv0", "near_pv1", "near_pv2", "far_pv0", "off_pv0"]
SEED2_MODELS = ["llamai", "qwen", "gemma"]


EPISODES_PATTERN_DEFAULT = "data/episodes_v1_full_{arm}{suffix}.json"


def arm_episodes(arm: str, suffix: str,
                 pattern: str = EPISODES_PATTERN_DEFAULT) -> str:
    return pattern.format(arm=arm, suffix=suffix)


# (instance,hop) exclusions; empty = the default mode (no filter)
EXCLUDED_IH: frozenset[tuple[int, int]] = frozenset()


def _ih(episode_id: str) -> tuple[int, int]:
    i, h = episode_id.split("-")[:2]
    return int(i), int(h)


def load_scored(path: Path) -> list[dict]:
    scored = json.loads(path.read_text())["scored"]
    if not EXCLUDED_IH:
        return scored
    return [s for s in scored if _ih(s["episode_id"]) not in EXCLUDED_IH]


def trivial_map(episodes_path: Path) -> dict[str, bool]:
    """episode_id -> "gold appears literally in the question text" (a trivial hop)."""
    eps = json.loads(episodes_path.read_text())["episodes"]
    return {
        e["episode_id"]: normalize(e["gold_answer"]) in normalize(e["question"])
        for e in eps
    }


def arm_stats(scored: list[dict], trivial: dict[str, bool]) -> dict:
    bw = [s for s in scored if s["condition"] == "tool_wrong" and s["cell"] == "both_wrong"]
    arb = [s for s in scored if s["condition"] == "tool_wrong" and s["cell"] == "arb"]
    arbq = [s for s in arb if not trivial[s["episode_id"]]]
    te = [s for s in scored if s["condition"] == "tool_error"]
    return {
        "prior": sum(s["followed_tool"] for s in bw) / len(bw),
        # share of both_wrong answers that matched gold (under-elicitation,
        # §3.5)
        "bw_gold": sum(s["final_correct"] for s in bw) / len(bw),
        "n_bw": len(bw),
        "keep": sum(s["outcome"] == "kept_memory" for s in arb) / len(arb),
        "n_arb": len(arb),
        "keepQ": sum(s["outcome"] == "kept_memory" for s in arbq) / len(arbq),
        "n_arbQ": len(arbq),
        # presentation arms carry only tool_right/tool_wrong
        "te_abstain": (sum(s["abstain"] for s in te) / len(te)) if te else None,
    }


def two_prop_z(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-proportion z test (pooled), two-sided p."""
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def holm(raw: dict[str, float]) -> dict[str, float]:
    """Holm correction: p_adj = the running maximum of (m-i)*p, monotone."""
    items = sorted(raw.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def pairwise_family(
    counts: dict[str, tuple[int, int]],
    metric: str,
    pairs: list[tuple[str, str]] | None = None,
) -> dict:
    """Pairwise tests over counts {model: (successes, n)} plus Holm.
    
    The correction family is the list of pairs passed in (by default all 10);
    Holm is applied WITHIN the family — planned narrow families (for example
    base vs each instruct) must be given explicitly.
    """
    if pairs is None:
        pairs = list(combinations(MODELS, 2))
    raw = {f"{a}_vs_{b}": two_prop_z(*counts[a], *counts[b]) for a, b in pairs}
    adj = holm(raw)
    models_used = sorted({m for p in pairs for m in p}, key=MODELS.index)
    return {
        "metric": metric,
        "method": "two-proportion z (pooled), two-sided, Holm correction",
        "family_size": len(pairs),
        "counts": {m: {"x": counts[m][0], "n": counts[m][1]} for m in models_used},
        "pairs": {
            k: {"p_raw": round(raw[k], 6), "p_holm": round(adj[k], 6)}
            for k in sorted(raw, key=raw.get)
        },
    }


def apply_ctrl_judge(scored: list[dict], tag: str, judge: dict) -> list[dict]:
    """Judge overlay on a control arm: tool_wrong neither/other only; the tool
    value comes from the judge request's metadata, and the decision
    (keep/follow/both) is made by the matcher over the extraction.
    """
    out = []
    for s in scored:
        if (s["condition"] == "tool_wrong" and s["outcome"] == "neither"
                and s.get("neither_subtype") == "other"):
            jr = judge.get(f"x|{tag}|near_pv0|{s['episode_id']}")
            if jr:
                s = corrected_outcome(
                    s, jr["meta"].get("tool"), parse_extract(jr.get("judge_raw")))
        out.append(s)
    return out


def control_row(scored: list[dict]) -> dict:
    """One row of the §7 table from scored[] of a single arm (strict + flag)."""
    arb = [s for s in scored if s["condition"] == "tool_wrong" and s["cell"] == "arb"]
    bw = [s for s in scored if s["condition"] == "tool_wrong" and s["cell"] == "both_wrong"]
    tg = [s for s in scored if s["condition"] == "tool_right" and s["cell"] == "tool_gold"]
    nc = [s for s in scored if s["condition"] == "no_conflict" and s["cell"] == "tool_gold"]
    te = [s for s in scored if s["condition"] == "tool_error"]
    r = lambda v: round(v, 4)
    return {
        "keep_strict": r(sum(s["outcome"] == "kept_memory" for s in arb) / len(arb)),
        "keep_flag": r(sum(s["kept_memory"] for s in arb) / len(arb)),
        "n_both_in_arb": sum(s["outcome"] == "both" for s in arb),
        "CRA_arb": r(sum(s["final_correct"] for s in arb) / len(arb)),
        "tool_gold_follow": r(sum(s["followed_tool"] for s in tg) / len(tg)),
        "prior_bw": r(sum(s["followed_tool"] for s in bw) / len(bw)),
        "nc_CRA": r(sum(s["final_correct"] for s in nc) / len(nc)),
        "te_abstain": r(sum(s["abstain"] for s in te) / len(te)),
        "n_arb": len(arb),
        "n_bw": len(bw),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judged-dir", default="artifacts/E019_prep",
                    help="directory of <model>_<arm>_judged/metrics_v1_judged.json")
    ap.add_argument("--det-dir", default="artifacts/E017_prep",
                    help="directory of det arms <model>_<arm>_qc and seed2_*")
    ap.add_argument("--ctrl-dir", default="artifacts/E021_prep",
                    help="directory of ctrl_{warn,prior,flag,dpo}")
    ap.add_argument("--episodes-suffix", default="",
                    help="suffix of the episode files")
    ap.add_argument("--judge-responses", default=None,
                    help="jsonl of a single judge run: builds the judged §7 rows "
                         "as a recipe overlay on the control arms (DPO included)")
    ap.add_argument("--exclusions", default=None,
                    help="frozen_mapping_v2.json: drops the sibling episodes "
                         "of an excluded (instance,hop) in every condition (canon v2)")
    ap.add_argument("--seed2-dir", default=None,
                    help="directory of seed2_*_qc (default: --det-dir); NONE — seed2 was not "
                         "re-run on this canon, and a note is written into the summary "
                         "instead of a cross-canon delta")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="models of the judged summary matrix")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="arms of the judged summary matrix (shared by all models)")
    ap.add_argument("--canon-arm", default="near_pv0",
                    help="canonical arm of the pairwise tests and the judged anchor")
    ap.add_argument("--canon-model", default="llamai",
                    help="model of the det anchor of the control table")
    ap.add_argument("--episodes-pattern", default=EPISODES_PATTERN_DEFAULT,
                    help="path template of the episodes, fields {arm} and {suffix}")
    ap.add_argument("--det-arm-suffix", default="_qc",
                    help="suffix of an arm's det directory: <model>_<arm><suffix>")
    ap.add_argument("--ctrl-mode", choices=["legacy", "permodel"],
                    default="legacy",
                    help="legacy: ctrl_dir/ctrl_{warn,prior,flag,dpo}; "
                         "permodel: canonical+control_{strategies} x pv for "
                         "each instruct, from the det and judged layers directly")
    ap.add_argument("--ctrl-models", default="llamai,qwen,gemma,mistral",
                    help="models of the permodel control table")
    ap.add_argument("--ctrl-strategies", default="warn,prior,flag",
                    help="strategies of the permodel control table")
    ap.add_argument("--base-all-pv", action="store_true",
                    help="print base on every pv (closed-book parametrics "
                         "make the base cells valid on pv1/pv2)")
    args = ap.parse_args()
    models = args.models.split(",")
    arms = args.arms.split(",")
    canon = args.canon_arm
    canon_base = canon.rsplit("_pv", 1)[0]
    pv_arms = [a for a in arms if a.rsplit("_pv", 1)[0] == canon_base]
    if args.ctrl_mode == "permodel" and args.judge_responses:
        ap.error("--judge-responses is incompatible with --ctrl-mode permodel: "
                 "the judged control rows are read from the judged directory directly")
    if args.exclusions:
        mp = json.loads(Path(args.exclusions).read_text())
        global EXCLUDED_IH
        EXCLUDED_IH = frozenset(
            _ih(e["episode_id"]) for e in mp["excluded_episodes"])
    judged_dir = ROOT / args.judged_dir
    det_dir = ROOT / args.det_dir
    ctrl_dir = ROOT / args.ctrl_dir

    trivial = {arm: trivial_map(ROOT / arm_episodes(
        arm, args.episodes_suffix, args.episodes_pattern)) for arm in arms}

    # --- 1. Canonical judged summary ---
    summary: dict = {}
    for m in models:
        summary[m] = {}
        for arm in arms:
            scored = load_scored(judged_dir / f"{m}_{arm}_judged/metrics_v1_judged.json")
            summary[m][arm] = arm_stats(scored, trivial[arm])

    out_path = judged_dir / "summary_judged_qminus.json"
    if out_path.exists():  # check against the already published summary before
                           # overwriting
        old = json.loads(out_path.read_text())
        for m in models:
            for arm in arms:
                for k, v in summary[m][arm].items():
                    ov = old[m][arm].get(k)  # the old summary has none of the new
                                             # fields
                    assert ov is None or v is None or abs(ov - v) < 1e-9, \
                        f"disagreement {m}/{arm}/{k}: {ov} != {v}"

    summary["pairwise_tests"] = {
        "arm": canon,
        "prior_bw": pairwise_family(
            {
                m: (
                    round(summary[m][canon]["prior"] * summary[m][canon]["n_bw"]),
                    summary[m][canon]["n_bw"],
                )
                for m in models
            },
            "source prior (followed_tool | both_wrong)",
        ),
        "keepQ": pairwise_family(
            {
                m: (
                    round(summary[m][canon]["keepQ"] * summary[m][canon]["n_arbQ"]),
                    summary[m][canon]["n_arbQ"],
                )
                for m in models
            },
            "keep strict on Q- (arb without the trivial hops)",
        ),
        # Planned narrow family for the Finding 2 claim "base is the best
        # Q- defender": 4 comparisons, base vs each instruct.
        "keepQ_base_vs_instructs": pairwise_family(
            {
                m: (
                    round(summary[m][canon]["keepQ"] * summary[m][canon]["n_arbQ"]),
                    summary[m][canon]["n_arbQ"],
                )
                for m in models
            },
            "keep strict on Q-: base vs each instruct (planned family)",
            pairs=[("base", m) for m in models if m != "base"],
        ),
    }
    summary["_meta"] = {
        "built_by": "scripts/build_judged_summary.py",
        "keep_definition": "strict: P(outcome=='kept_memory') on arb; 'both' excluded",
        "qminus_rule": "normalize(gold) not substring of normalize(question)",
        "source": f"{args.judged_dir}/<model>_<arm>_judged/metrics_v1_judged.json",
        "episodes_suffix": args.episodes_suffix,
    }
    if args.exclusions:
        summary["_meta"]["qc8_exclusions"] = {
            "path": args.exclusions, "n_instance_hops": len(EXCLUDED_IH)}
    out_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {out_path}")

    # --- 2. Strict slice of §7 + the judged canonical anchor ---
    if args.ctrl_mode == "legacy":
        ctrl = {
            "canon_v3": control_row(load_scored(
                det_dir / f"{args.canon_model}_{canon}{args.det_arm_suffix}/metrics_v1.json"
            )),
            "warn": control_row(load_scored(ctrl_dir / "ctrl_warn/metrics_v1.json")),
            "source_priority": control_row(
                load_scored(ctrl_dir / "ctrl_prior/metrics_v1.json")
            ),
            "abstain_and_flag": control_row(
                load_scored(ctrl_dir / "ctrl_flag/metrics_v1.json")
            ),
            "dpo": control_row(load_scored(ctrl_dir / "ctrl_dpo/metrics_v1.json")),
        }
    else:  # permodel: canonical + control_{strategies} x pv, det AND judged
           # layers
        pvs = [a.rsplit("_pv", 1)[1] for a in pv_arms]
        ctrl = {"_layout": "permodel", "det": {}, "judged": {}}
        for m in args.ctrl_models.split(","):
            ctrl["det"][m], ctrl["judged"][m] = {}, {}
            for pv in pvs:
                for arm in [f"{canon_base}_pv{pv}"] + [
                        f"control_{st}_pv{pv}"
                        for st in args.ctrl_strategies.split(",")]:
                    ctrl["det"][m][arm] = control_row(load_scored(
                        det_dir / f"{m}_{arm}{args.det_arm_suffix}/metrics_v1.json"))
                    ctrl["judged"][m][arm] = control_row(load_scored(
                        judged_dir / f"{m}_{arm}_judged/metrics_v1_judged.json"))
    canon_judged_scored = load_scored(
        judged_dir / f"{args.canon_model}_{canon}_judged/metrics_v1_judged.json"
    )
    canon_judged = arm_stats(canon_judged_scored, trivial[canon])
    ctrl["_meta"] = {
        "built_by": "scripts/build_judged_summary.py",
        "keep_definition": "strict (see summary_judged_qminus.json); keep_flag is the old flag field, both included",
        "anchor_canon_judged_keep_strict": round(canon_judged["keep"], 4),
        "note": "+the judge's delta for the anchor = anchor_canon_judged_keep_strict - canon_v3.keep_strict",
        "dirs": {"judged": args.judged_dir, "det": args.det_dir, "ctrl": args.ctrl_dir},
    }
    if args.ctrl_mode != "legacy":
        ctrl["_meta"]["ctrl_mode"] = args.ctrl_mode
        ctrl["_meta"]["note"] = (
            "permodel: det/judged rows per model and pv; canonical = "
            "the model's untreated anchor, with the judge applied to every arm directly")
    if args.exclusions:
        ctrl["_meta"]["qc8_exclusions"] = {
            "path": args.exclusions, "n_instance_hops": len(EXCLUDED_IH)}

    # --- 2b. Judged §7 rows from a single judge run ---
    if args.judge_responses:
        judge = load_judge(str(ROOT / args.judge_responses))
        judged_rows = {"untreated": control_row(canon_judged_scored)}
        for tag, name in [("ctrl_warn", "warn"), ("ctrl_prior", "source_priority"),
                          ("ctrl_flag", "abstain_and_flag"), ("ctrl_dpo", "dpo")]:
            judged_rows[name] = control_row(apply_ctrl_judge(
                load_scored(ctrl_dir / f"{tag}/metrics_v1.json"), tag, judge))
        ctrl["judged_rows"] = judged_rows
        ctrl["_meta"]["judged_rows_recipe"] = (
            f"untreated = the judged anchor from {args.judged_dir}; control arms (DPO included) = "
            f"a recipe overlay ({args.judge_responses}) on det-scored, "
            "tool_wrong neither/other only"
        )

    # --- 3. Strict seed-1 vs seed-2 deltas ---
    if args.seed2_dir == "NONE":
        ctrl["seed2_strict_deltas"] = {
            "_note": "seed2 was not re-run on this canon; a cross-canon "
                     "delta is not computed (see the legacy canon)"
        }
    else:
        seed2_dir = ROOT / args.seed2_dir if args.seed2_dir else det_dir
        seed2 = {}
        for m in SEED2_MODELS:
            s1 = control_row(load_scored(
                det_dir / f"{m}_{canon}{args.det_arm_suffix}/metrics_v1.json"))
            s2 = control_row(load_scored(
                seed2_dir / f"seed2_{m}_{canon}{args.det_arm_suffix}/metrics_v1.json")
            )
            seed2[m] = {
                "d_prior": round(abs(s1["prior_bw"] - s2["prior_bw"]), 4),
                "d_keep_strict": round(abs(s1["keep_strict"] - s2["keep_strict"]), 4),
                "d_keep_flag": round(abs(s1["keep_flag"] - s2["keep_flag"]), 4),
            }
        ctrl["seed2_strict_deltas"] = seed2

    ctrl_path = ctrl_dir / "summary_control_strict.json"
    ctrl_path.write_text(json.dumps(ctrl, indent=1, ensure_ascii=False))
    print(f"wrote {ctrl_path}")

    # --- Console digest, for checking against the paper ---
    pt = summary["pairwise_tests"]
    for fam in ("prior_bw", "keepQ", "keepQ_base_vs_instructs"):
        sig = {k: v for k, v in pt[fam]["pairs"].items() if v["p_holm"] < 0.05}
        print(f"{fam}: significant (Holm<.05): {sig}")
    print("keepQ gemma_vs_llamai:", pt["keepQ"]["pairs"].get("llamai_vs_gemma"))
    print("prior base_vs_llamai:", pt["prior_bw"]["pairs"].get("base_vs_llamai"))
    nonzero_pv = [a for a in pv_arms if a != canon]
    reported = [
        (m, a)
        for m in models
        for a in arms
        if args.base_all_pv or not (m == "base" and a in nonzero_pv)
    ]
    n_cells = [summary[m][a][k] for m, a in reported for k in ("n_arb", "n_bw")]
    print("conflict-cell n over REPORTED arms (base=pv0-only):", min(n_cells), max(n_cells))
    pv0_cells = [
        summary[m][a][k]
        for m, a in reported
        if a.endswith("pv0")
        for k in ("n_arb", "n_bw")
    ]
    print("conflict-cell n over pv0 arms:", min(pv0_cells), max(pv0_cells))
    for arm_name, row in ctrl.items():
        if isinstance(row, dict) and "keep_strict" in row:
            print(f"§7 {arm_name}: strict {row['keep_strict']} (flag {row['keep_flag']})")
    for arm_name, row in (ctrl.get("judged_rows") or {}).items():
        print(f"§7 {arm_name} (judged): keep {row['keep_strict']} CRA {row['CRA_arb']} "
              f"tg {row['tool_gold_follow']} prior {row['prior_bw']}")
    print("seed2 strict deltas:", ctrl["seed2_strict_deltas"])
    inst_range = {
        m: round(
            max(summary[m][a]["keep"] for a in pv_arms)
            - min(summary[m][a]["keep"] for a in pv_arms),
            3,
        )
        for m in models
        if m != "base"
    }
    print("keep strict paraphrase range (instruct):", inst_range)
    n_arbs = [summary[m][a]["n_arb"] for m in models for a in arms if a in pv_arms]
    print("n_arb min/max across near arms:", min(n_arbs), max(n_arbs))


if __name__ == "__main__":
    main()
