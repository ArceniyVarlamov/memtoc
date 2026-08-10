"""Paraphrase-pooled primary for the cross-model metrics (answer to W4).

W4 of the external review: the headline presentation leads with point
estimates on a single "canonical" paraphrase (pv0), even though the paraphrase
shift is larger than the gaps between models. What follows is the reviewer's
minimal well-posed alternative: per-model metrics averaged over the three
paraphrases (pooled mean, equal weight per paraphrase; each paraphrase has its
own cells, because closed-book elicitation is paraphrase-dependent), with the
min-max spread as the primary number and a bootstrap CI.

"Stable core" (the pre-registered rule): a statement about ordering enters the
core if and only if it holds on the pooled means AND on each of the three
paraphrases separately.

The base model: closed-book degrades on pv1/pv2 (§5.4), so its block is pv0
only and carries a disclosure; pooled comparisons against base are undefined.

Metrics and loading reuse the standard aggregator
(scripts.build_judged_summary: load_scored / trivial_map / arm_stats); the
per-arm numbers are checked with an assert against the canonical summary
(--canon-summary). The pooled CI is an episode bootstrap inside each arm
independently, then the mean over arms; percentile, n_boot=1000, seed=0.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import scripts.build_judged_summary as bjs
from scripts.build_common_support_summary import git_commit, sha256

INSTRUCTS = ["llamai", "qwen", "gemma", "mistral"]
NEAR_ARMS = ["near_pv0", "near_pv1", "near_pv2"]


def pooled_ci(arm_items: list[list[dict]], key: str, n_boot: int = 1000,
              seed: int = 0) -> list:
    """CI of the mean over arms: resample episodes inside each arm
    independently, then take the mean of the arm rates; percentile (the same
    parameters as in the paper).
    """
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        arm_means = []
        for items in arm_items:
            n = len(items)
            arm_means.append(
                sum(bool(items[rng.randrange(n)][key]) for _ in range(n)) / n)
        means.append(sum(arm_means) / len(arm_means))
    means.sort()
    return [round(means[int(0.025 * n_boot)], 4),
            round(means[min(n_boot - 1, int(0.975 * n_boot))], 4)]


def arb_records(scored: list[dict], qminus: bool,
                trivial: dict[str, bool]) -> list[dict]:
    recs = [s for s in scored if s["condition"] == "tool_wrong"
            and s["cell"] == "arb"]
    if qminus:
        recs = [s for s in recs if not trivial[s["episode_id"]]]
    # flat key for the bootstrap
    return [{"v": s["outcome"] == "kept_memory"} for s in recs]


def bw_records(scored: list[dict]) -> list[dict]:
    return [{"v": s["followed_tool"]} for s in scored
            if s["condition"] == "tool_wrong" and s["cell"] == "both_wrong"]


def ordering(points: dict[str, float]) -> list[str]:
    return sorted(points, key=points.get)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judged-dir", required=True)
    ap.add_argument("--episodes-suffix", default="")
    ap.add_argument("--arms", default=",".join(NEAR_ARMS),
                    help="paraphrase arms of the pool (canonical_pv0..2)")
    ap.add_argument("--base-arm", default="near_pv0",
                    help="the only reported base arm (closed-book collapse)")
    ap.add_argument("--episodes-pattern", default=bjs.EPISODES_PATTERN_DEFAULT,
                    help="path template of the episodes, fields {arm} and {suffix}")
    ap.add_argument("--exclusions", default=None)
    ap.add_argument("--canon-summary", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.exclusions:
        mp = json.loads(Path(args.exclusions).read_text())
        bjs.EXCLUDED_IH = frozenset(
            bjs._ih(e["episode_id"]) for e in mp["excluded_episodes"])

    judged_dir = Path(args.judged_dir)
    arms_list = args.arms.split(",")
    inputs: dict[str, str] = {}
    trivial = {}
    for arm in dict.fromkeys(arms_list + [args.base_arm]):
        ep = bjs.ROOT / bjs.arm_episodes(arm, args.episodes_suffix,
                                         args.episodes_pattern)
        trivial[arm] = bjs.trivial_map(ep)
        inputs[str(ep)] = sha256(ep)

    canon = None
    if args.canon_summary:
        canon = json.loads(Path(args.canon_summary).read_text())
        inputs[args.canon_summary] = sha256(Path(args.canon_summary))

    per_arm: dict[str, dict] = {}
    scored: dict[tuple[str, str], list[dict]] = {}
    for m in INSTRUCTS + ["base"]:
        arms = arms_list if m != "base" else [args.base_arm]
        per_arm[m] = {}
        for arm in arms:
            p = judged_dir / f"{m}_{arm}_judged/metrics_v1_judged.json"
            scored[(m, arm)] = bjs.load_scored(p)
            inputs[str(p)] = sha256(p)
            st = bjs.arm_stats(scored[(m, arm)], trivial[arm])
            if canon:  # per-arm check against the canonical summary
                for k, v in canon[m][arm].items():
                    assert abs(st[k] - v) < 1e-9, f"canon check {m}/{arm}/{k}"
            per_arm[m][arm] = st

    pooled: dict[str, dict] = {}
    for m in INSTRUCTS:
        vals = {k: [per_arm[m][a][k] for a in arms_list]
                for k in ("prior", "keep", "keepQ")}
        prior_items = [bw_records(scored[(m, a)]) for a in arms_list]
        keep_items = [arb_records(scored[(m, a)], False, trivial[a])
                      for a in arms_list]
        keepq_items = [arb_records(scored[(m, a)], True, trivial[a])
                       for a in arms_list]
        pooled[m] = {}
        for k, items in [("prior", prior_items), ("keep", keep_items),
                         ("keepQ", keepq_items)]:
            v = vals[k]
            pooled[m][k] = {
                "pooled_mean": round(sum(v) / len(v), 4),
                "range": [round(min(v), 4), round(max(v), 4)],
                "spread": round(max(v) - min(v), 4),
                "per_pv": {a: round(per_arm[m][a][k], 4) for a in arms_list},
                "pooled_ci95": pooled_ci(items, "v"),
                "n_per_pv": {a: len(it) for a, it in zip(arms_list, items)},
            }

    # Orderings: pooled and each paraphrase; the stable core by the
    # pre-registered rule
    orders = {
        "pooled": {k: ordering({m: pooled[m][k]["pooled_mean"]
                                for m in INSTRUCTS})
                   for k in ("prior", "keepQ")},
    }
    for a in arms_list:
        orders[a] = {k: ordering({m: per_arm[m][a][k] for m in INSTRUCTS})
                     for k in ("prior", "keepQ")}

    def scope_points(metric: str, scope: str) -> dict[str, float]:
        if scope == "pooled":
            return {m: pooled[m][metric]["pooled_mean"] for m in INSTRUCTS}
        return {m: per_arm[m][scope][metric] for m in INSTRUCTS}

    def core_check(metric: str, check) -> dict:
        res = {scope: bool(check(scope_points(metric, scope)))
               for scope in ["pooled"] + arms_list}
        res["stable_core"] = all(res.values())
        return res

    stable_core = {
        "prior_canonical_order_llamai_qwen_gemma_mistral": core_check(
            "prior", lambda p: ordering(p) == ["llamai", "qwen", "gemma", "mistral"]),
        "prior_mistral_max": core_check("prior", lambda p: max(p, key=p.get) == "mistral"),
        "prior_llamai_min": core_check("prior", lambda p: min(p, key=p.get) == "llamai"),
        "keepQ_qwen_max": core_check("keepQ", lambda p: max(p, key=p.get) == "qwen"),
        "keepQ_gemma_min": core_check("keepQ", lambda p: min(p, key=p.get) == "gemma"),
    }

    base_pv0 = {k: round(per_arm["base"][args.base_arm][k], 4)
                for k in ("prior", "keep", "keepQ")}

    summary = {
        "pooled_instructs": pooled,
        "orders_by_scope": orders,
        "stable_core": stable_core,
        "base_pv0_only": {
            **base_pv0,
            "note": "closed-book degrades for base on pv1/pv2 (§5.4) — pooled is "
                    "undefined; the claim 'base is the strongest Q- defender' is "
                    "scoped to pv0 (confirmed there on the common support)",
        },
        "_meta": {
            "built_by": "scripts/build_paraphrase_pooled_summary.py",
            "code_commit": git_commit(bjs.ROOT),
            "args": vars(args),
            "pooling": "equal weight per paraphrase; cells are per paraphrase "
                       "(closed-book elicitation is paraphrase-dependent); the CI is an episode "
                       "bootstrap inside an arm, then the mean over arms, percentile "
                       "1000/seed0",
            "stable_core_rule": "holds on pooled AND on each of the 3 "
                                "paraphrases (pre-registered)",
            "input_sha256": inputs,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {out}")
    for m in INSTRUCTS:
        p, q = pooled[m]["prior"], pooled[m]["keepQ"]
        print(f"{m:8s} prior pooled {p['pooled_mean']:.3f} {p['pooled_ci95']} "
              f"spread {p['spread']:.2f} | keepQ pooled {q['pooled_mean']:.3f} "
              f"{q['pooled_ci95']} spread {q['spread']:.2f}")
    print("base pv0-only:", base_pv0)
    print("pooled orders:", orders["pooled"])
    for name, r in stable_core.items():
        print(f"stable_core {name}: {r['stable_core']} ({r})")


if __name__ == "__main__":
    main()
