"""Validation and application of the LLM judge (extraction + acknowledgment).

Production rule (mirrored in validation): the judge's extraction is applied
ONLY where the deterministic scorer said neither/other (the matcher's blind
spot) or where the closed-book span is flagged as suspicious (an echo of the
question). The decision (keep/follow/correct) is always made by the matcher
over the extraction — the judge does not decide, the judge normalises.

validate: the extract path against the final QC set (outcome, 120 labels) and
          the ack path against FINAL_ack (recall/precision on 11 positives).
apply:    judge-corrected metrics_v1_judged.json per arm.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from memtoc.metrics import aggregate_v1
from memtoc.scoring import is_abstain, match


def load_judge(path: str) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        out[r["id"]] = r
    return out


def parse_extract(raw: str) -> str | None:
    """The extractor judge's answer -> entity | 'ABSTAIN' | None (junk)."""
    if raw is None or raw.startswith("[JUDGE-ERROR]"):
        return None
    ans = raw.strip().splitlines()[0].strip().strip('"*').strip()
    if not ans or len(ans.split()) > 12:
        return None
    if ans.upper().startswith("ABSTAIN"):
        return "ABSTAIN"
    return ans


def parse_ack(raw: str) -> bool | None:
    if raw is None or raw.startswith("[JUDGE-ERROR]"):
        return None
    w = raw.strip().upper()
    if w.startswith("YES"):
        return True
    if w.startswith("NO"):
        return False
    return None


def corrected_outcome(s: dict, tool_value, ext: str | None) -> dict:
    """Rebuild the outcome for neither/other from the judge's extraction."""
    upd = dict(s)
    if ext is None:
        return upd  # junk answer from the judge — keep the scorer's label
    if ext == "ABSTAIN":
        upd["neither_subtype"] = "abstain"
        upd["abstain"] = True
        return upd
    km = bool(s.get("parametric_span")) and match(ext, s["parametric_span"])
    ft = tool_value is not None and match(ext, str(tool_value))
    if ft and km:
        upd["outcome"] = "both"
    elif ft:
        upd["outcome"] = "followed_tool"
    elif km:
        upd["outcome"] = "kept_memory"
    else:
        upd["outcome"] = "neither"
    if upd["outcome"] != "neither":
        upd["neither_subtype"] = None
    upd["kept_memory"], upd["followed_tool"] = km, ft
    upd["judge_extract"] = ext
    return upd


_CELL = {(True, True): "agree", (True, False): "arb",
         (False, True): "tool_gold", (False, False): "both_wrong"}


def cmd_validate(args) -> int:
    judge = load_judge(args.judge)
    canon = {}
    for mo in ["base", "llamai", "qwen", "gemma", "mistral"]:
        mm = json.loads((Path(args.qc_labels_dir) / f"rescored_{mo}" / "metrics_v1.json").read_text())
        canon[mo] = {s["episode_id"]: s for s in mm["scored"]}
    eps = {e["episode_id"]: e for e in json.loads(
        Path(args.episodes).read_text())["episodes"]}

    # --- outcome path ---
    y_true, y_scorer, y_pred, fixed, broken = [], [], [], [], []
    for r in csv.DictReader(open(args.qc_labels)):
        s = canon[r["model"]][r["episode_id"]]
        human = r["FINAL_outcome"]
        pred = s["outcome"]
        if s["outcome"] == "neither" and s.get("neither_subtype") == "other":
            jr = judge.get(f"vx|{r['model']}|{r['episode_id']}")
            if jr:
                ext = parse_extract(jr["judge_raw"])
                tool_value = (eps[r["episode_id"]].get("tool_output") or {}).get("result")
                pred = corrected_outcome(s, tool_value, ext)["outcome"]
        y_true.append(human)
        y_scorer.append(s["outcome"])
        y_pred.append(pred)
        if pred == human and s["outcome"] != human:
            fixed.append(r["episode_id"])
        if pred != human and s["outcome"] == human:
            broken.append((r["episode_id"], s["outcome"], pred))

    def _agree_kappa(yt, yp):
        agree = sum(t == p for t, p in zip(yt, yp)) / len(yt)
        cats = sorted(set(yt) | set(yp))
        pt, pp = Counter(yt), Counter(yp)
        pe = sum(pt[c] * pp[c] for c in cats) / len(yt) ** 2
        return agree, (agree - pe) / (1 - pe) if pe < 1 else 0.0

    a0, k0 = _agree_kappa(y_true, y_scorer)
    a1, k1 = _agree_kappa(y_true, y_pred)
    print(f"[val-outcome] scorer: agreement {a0:.3f} (kappa {k0:.3f}) -> "
          f"with the judge: {a1:.3f} (kappa {k1:.3f}) on {len(y_true)}; "
          f"fixed {len(fixed)}, broken {len(broken)} {broken[:5]}")

    # --- ack path ---
    tp = fp = fn = tn = miss = 0
    for r in csv.DictReader(open(args.qc_labels)):
        if r["FINAL_ack"] not in ("yes", "no"):
            continue
        jr = judge.get(f"va|{r['model']}|{r['episode_id']}")
        pred = parse_ack(jr["judge_raw"]) if jr else None
        if pred is None:
            miss += 1
            continue
        human = r["FINAL_ack"] == "yes"
        tp += pred and human
        fp += pred and not human
        fn += (not pred) and human
        tn += (not pred) and (not human)
    rec = tp / (tp + fn) if tp + fn else None
    prec = tp / (tp + fp) if tp + fp else None
    print(f"[val-ack] recall {rec} ({tp}/{tp+fn}), precision {prec} "
          f"({tp}/{tp+fp}), tn {tn}, no answer {miss}")
    return 0


def cmd_apply(args) -> int:
    judge = load_judge(args.judge)
    outroot = Path(args.out)
    for mo in args.models.split(","):
        for arm in args.arms.split(","):
            mp = Path(args.scored_arms_dir) / f"{mo}_{arm}{args.dir_suffix}" / "metrics_v1.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text())
            eps = {e["episode_id"]: e for e in json.loads(
                (Path(args.episodes_dir) / args.episodes_pattern.format(arm=arm)).read_text())["episodes"]}
            # hop-level closed-book fixes (echo of the question) from
            # cb_suspect extractions
            cb_fix: dict[str, str] = {}
            for s in m["scored"]:
                jr = judge.get(f"c|{mo}|{arm}|{s['episode_id']}")
                if not jr:
                    continue
                ext = parse_extract(jr["judge_raw"])
                if ext:
                    hop = "-".join(s["episode_id"].split("-")[:2])
                    cb_fix[hop] = ext
            corrected, n_cb, n_out, n_ack = [], 0, 0, 0
            for s in m["scored"]:
                s = dict(s)
                ep = eps[s["episode_id"]]
                hop = "-".join(s["episode_id"].split("-")[:2])
                # mem_correct from the closed-book extraction
                if hop in cb_fix:
                    ext = cb_fix[hop]
                    if ext == "ABSTAIN" or is_abstain(ext):
                        s["mem_absent"], s["memory_correct"] = True, None
                        s["parametric_span"] = ""
                    else:
                        s["parametric_span"] = ext
                        s["memory_correct"] = match(ext, ep["gold_answer"])
                        s["mem_absent"] = False
                    if ep["tool_correct"] is None:
                        s["cell"] = None
                    elif s["mem_absent"]:
                        s["cell"] = "mem_absent"
                    else:
                        s["cell"] = _CELL[(bool(s["memory_correct"]), bool(ep["tool_correct"]))]
                    n_cb += 1
                # neither/other through the extraction of the final answer
                jr = judge.get(f"x|{mo}|{arm}|{s['episode_id']}")
                if jr and s["outcome"] == "neither" and s.get("neither_subtype") == "other":
                    tool_value = (ep.get("tool_output") or {}).get("result")
                    s2 = corrected_outcome(s, tool_value, parse_extract(jr["judge_raw"]))
                    n_out += s2["outcome"] != s["outcome"] or s2.get("neither_subtype") != s.get("neither_subtype")
                    s = s2
                # CAR: the judge instead of the proxy
                jr = judge.get(f"a|{mo}|{arm}|{s['episode_id']}")
                if jr:
                    a = parse_ack(jr["judge_raw"])
                    if a is not None:
                        s["ack_judge"] = a
                        n_ack += 1
                corrected.append(s)
            agg = aggregate_v1(corrected)
            car = {}
            tw = [s for s in corrected if s["condition"] == "tool_wrong"
                  and s.get("cell") in ("arb", "both_wrong") and "ack_judge" in s]
            if tw:
                car["CAR_judge_tool_wrong_conflict"] = round(
                    sum(s["ack_judge"] for s in tw) / len(tw), 4)
                car["n"] = len(tw)
            od = outroot / f"{mo}_{arm}_judged"
            od.mkdir(parents=True, exist_ok=True)
            (od / "metrics_v1_judged.json").write_text(json.dumps({
                "meta": {"base": str(mp), "judge": args.judge,
                         "n_cb_fixed_hops": len(cb_fix), "n_outcome_corrected": n_out,
                         "n_ack_judged": n_ack, "car_judge": car},
                "aggregate_v1": agg, "scored": corrected}, ensure_ascii=False, indent=2))
            arb = agg["tool_wrong"]["cells"].get("arb", {})
            print(f"[apply] {mo}/{arm}: closed-book fixes {len(cb_fix)}, outcomes corrected {n_out}, "
                  f"CAR_judge {car.get('CAR_judge_tool_wrong_conflict')} | "
                  f"arb keep {arb.get('kept_memory')} (n={arb.get('n')})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--judge", required=True)
    v.add_argument("--qc-labels", required=True)
    v.add_argument("--qc-labels-dir", required=True)
    v.add_argument("--episodes", required=True,
                   help="episode file p1 (tool_output for the validation rows)")
    a = sub.add_parser("apply")
    a.add_argument("--judge", required=True)
    a.add_argument("--scored-arms-dir", default="artifacts/E017_prep")
    a.add_argument("--episodes-dir", default="data")
    a.add_argument("--models",
                   default="base,llamai,qwen,gemma,mistral",
                   help="CSV of arm-\"model\" names (prefix of the directories "
                        "<model>_<arm>_qc); default is the canonical five")
    a.add_argument("--arms",
                   default="near_pv0,near_pv1,near_pv2,far_pv0,off_pv0",
                   help="CSV of arm suffixes; default is the canonical set")
    a.add_argument("--dir-suffix", default="_qc",
                   help="suffix of the arm directories (<model>_<arm><suffix>); "
                        "default is bit-for-bit with the earlier canon")
    a.add_argument("--episodes-pattern", default="episodes_v1_full_{arm}.json",
                   help="template of an arm's episode file name inside --episodes-dir; "
                        "default is bit-for-bit with the earlier canon")
    a.add_argument("--out", default="artifacts/E019_prep")
    args = ap.parse_args()
    return cmd_validate(args) if args.cmd == "validate" else cmd_apply(args)


if __name__ == "__main__":
    sys.exit(main())
