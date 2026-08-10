#!/usr/bin/env python3
"""Topic histogram of the 575-question construction pool and the 542 analysis
canon (2026-07-30).

Why this exists: the appendix's "Topics" paragraph carried a tail that did not
close --- ten named topics (439 questions) plus "the other 29 topics" (67)
accounted for 39 of the stated 47 topics and 506 of the 575 questions. It was
logged as defect D1 in paper/CUTLIST-2026-07-28-postmerge.md and marked
"regenerate from the artifact before submission". This is that regeneration.

Method. ToolHop labels a *chain* with a domain; a question inherits the label of
the chain it was attributed to. So the histogram is the freeze's per-question
`instance_id` (a ToolHop list index) joined to `ToolHop.json[...]["domain"]`.
Labels are case-folded: ToolHop ships 88 raw domain strings that collapse to 62
distinct labels (e.g. "Film"/"film", "Genealogy"/"genealogy"), and the paper's
published figures are the case-folded ones.

Result (575 pool): 47 topics; top ten 439 = 76.3%; tail 37 topics / 136
questions; largest tail topic 12. All ten published top-ten counts, the 76.3%
share, the 62-domain figure, the 309 contributing chains and the Step 7 split
(PASS 503 / REPAIR 39 / EXCLUDE 33) reproduce exactly, which is what validates
the join. The published tail ("29 topics", "67 questions", "none with more than
five") does not reproduce and was corrected on Overleaf on 2026-07-30.

Run from the repo root:
    python tools/recount_topics.py
"""
import collections
import json
import pathlib
import sys

FREEZE = pathlib.Path("results/quality_control/semantic_review_575.json")
# ToolHop is not vendored (Apache-2.0, ByteDance). In this repo it sits in the
# code checkout next door; in the released artifact, pass --toolhop or drop the
# downloaded file in the tree root. Expected sha256:
# 0a51f71a44b7025645e452123af3caf2e348301922af91778e268db0188a7fab
TOOLHOP_CANDIDATES = [
    pathlib.Path("ToolHop.json"),
    pathlib.Path("data/ToolHop.json"),
    pathlib.Path("../smiles-2026-memtoc/data/ToolHop.json"),
]


def find_toolhop():
    for p in TOOLHOP_CANDIDATES:
        if p.exists():
            return p
    return None


def domain_of(chain):
    return (chain.get("domain") or "").strip().lower()


def histogram(rows, toolhop):
    return collections.Counter(domain_of(toolhop[r["instance_id"]]) for r in rows)


def report(name, rows, toolhop):
    counts = histogram(rows, toolhop)
    total = sum(counts.values())
    ranked = counts.most_common()
    top10 = sum(n for _, n in ranked[:10])
    tail = ranked[10:]
    print(f"\n=== {name}: {total} questions, {len(counts)} distinct topics ===")
    for i, (topic, n) in enumerate(ranked[:10], 1):
        print(f"  {i:>2}. {topic:<18} {n:>4}  {100 * n / total:.1f}%")
    print(f"  top ten            {top10:>4}  {100 * top10 / total:.1f}%")
    print(f"  tail               {sum(n for _, n in tail):>4}"
          f"  over {len(tail)} topics, largest {tail[0][1] if tail else 0}")
    hfg = counts["history"] + counts["film"] + counts["genealogy"]
    print(f"  history+film+genealogy {hfg}  {100 * hfg / total:.1f}%")
    return counts


def main():
    argv = sys.argv[1:]
    override = None
    if "--toolhop" in argv:
        override = pathlib.Path(argv[argv.index("--toolhop") + 1])
    toolhop_path = override or find_toolhop()
    if not FREEZE.exists():
        sys.exit(f"missing {FREEZE} (run from the tree root)")
    if toolhop_path is None or not toolhop_path.exists():
        sys.exit(
            "ToolHop.json not found. It is not vendored (Apache-2.0, ByteDance).\n"
            "  curl -L -o ToolHop.json https://huggingface.co/datasets/"
            "bytedance-research/ToolHop/resolve/main/data/ToolHop.json\n"
            "  sha256 must be 0a51f71a44b7025645e452123af3caf2e348301922af9"
            "1778e268db0188a7fab\n"
            "Then re-run, or pass --toolhop <path>."
        )

    toolhop = json.loads(toolhop_path.read_text(encoding="utf-8"))
    rows = json.loads(FREEZE.read_text(encoding="utf-8"))["rows"]

    raw = [(c.get("domain") or "").strip() for c in toolhop]
    print(f"ToolHop: {len(toolhop)} chains, {sum(1 for d in raw if d)} with a domain")
    print(f"  distinct raw {len({d for d in raw if d})}"
          f" -> case-folded {len({d.lower() for d in raw if d})}  (paper says 62)")

    print(f"\nfreeze: {len(rows)} rows, {len({r['instance_id'] for r in rows})}"
          f" distinct chains  (paper says 575 from 309)")
    print(f"  Step 7 verdicts: {dict(collections.Counter(r['final_status'] for r in rows))}")

    report("575 construction pool", rows, toolhop)
    report("542 analysis canon",
           [r for r in rows if r["final_status"] != "EXCLUDE"], toolhop)


if __name__ == "__main__":
    main()
