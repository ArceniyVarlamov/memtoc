# MemToC — benchmark, results and reproduction harness

Everything needed to inspect the benchmark and to re-derive the paper's numbers
without a GPU.

> Paper: Arseniy Varlamov, Rishat Zinnatullin, Elisei Rykov, Alexander Panchenko,
> Ilseyar Alimova. *MemToC: Benchmarking Memory–Tool Conflict Resolution in Large
> Language Models.* arXiv:2608.26295, 2026. <https://arxiv.org/abs/2608.26295>

```
benchmark/    the benchmark itself: 575 questions x 5 tool conditions,
              in three prompt formulations
results/      the scoring and aggregation layer every table is read from
code/         the pipeline that built the benchmark and scored the runs
tools/        two scripts that re-derive the paper's numbers and print PASS/FAIL
CLAIM-TO-FILE.md   which file carries which reported number
LICENSE, NOTICE    Apache-2.0, and the ToolHop / Wikidata attributions
```

## Reproducing the experiments

```bash
python tools/verify_paper_numbers.py -v     # 180 checks, all from results/
python tools/recount_topics.py              # topic histogram of the question pool
```

`verify_paper_numbers.py` recomputes 180 published values and diffs each against
the number printed in the manuscript: the three headline ranges of the abstract,
every cell of the main table's original-checkpoint rows, the whole prompting
ladder for four models, the retention and incorrect-tool columns for five models
under three formulations, the paired fine-tuning changes, the acknowledgment
count, the Section 3.2 construction branch counts and the verdicts of the
blinded semantic review. It exits 0 on success and reports, separately from the checks, the one
value that does not reproduce and the two families that are out of reach from
this tree.

Requirements are `numpy` and `openpyxl` (`code/requirements.txt`); the
two reproduction scripts above need neither.

**The one disagreement.** The main table prints `99.5` for gemma's tool-error
abstention; this archive gives `99.4467` (542/542, 537/542, 538/542 across the
three formulations), which rounds to `99.4`. The other three models' cells in
that column are exact. The harness reports this rather than hiding it.

## The benchmark

`benchmark/reference.json`, `paraphrase_a.json`, `paraphrase_b.json` — one file
per prompt formulation, matching the names used in the paper. Each holds
**2,875 episodes = 575 questions x 5 conditions**, with 575 distinct question
texts.

**What varies across the three formulations is the instruction, not the
question.** All 2,875 `question` strings are byte-identical across the three
files, as are the verified answers, tool calls and tool returns; diffing two
files shows this directly. What is paraphrased is the wording that wraps them —
the closed-book instruction differs in all 2,875 episodes, and the with-tool
instruction in all 2,300 that have one (`with_tool` is `null` for the 575
`no_tool` episodes):

| file | the with-tool instruction opens |
|---|---|
| `reference.json` | *You called the tool below; its output is shown…* |
| `paraphrase_a.json` | *A tool was invoked and returned the result below…* |
| `paraphrase_b.json` | *Here is a tool and what it returned…* |

Read the robustness result as what it is: no cross-model ordering survives a
change of instruction wording, with the question held fixed.

The five stored conditions are `tool_right`, `tool_wrong`, `tool_error`,
`no_tool`, `no_conflict`. The paper's headline count of **6,504 evaluation
episodes** is `542 x 4 x 3`: it uses the 542-question analysis set (the 33
records the blinded semantic review excluded, see `results/quality_control/`) and counts four
conditions — `no_conflict` is a byte-identical duplicate of `tool_right` and is
not counted twice. The `far` condition is not a separate file: far distractors
live in the mapping (present for 463 of 575 questions) and materialise through
the same builder.

These files are the arms that were actually evaluated: the base build plus the
repair overlay produced by the blinded semantic review. The overlay changes 39 episodes — one per `REPAIR`
verdict — and only their substituted incorrect value; question wording and
verified answers are frozen byte-for-byte. `benchmark/repair_overlay/` carries
the overlay so the step is checkable, and each file records `v2_sha256` and
`patch_sha256` alongside `mapping_sha256`.

Rebuilding from the mapping (below) reproduces the **pre-repair** build; apply
the overlay to reach the evaluated one. Reporting numbers against the pre-repair
build will disagree with the paper on those 39 questions.

## What is in `results/`

| Directory | Contents |
|---|---|
| `construction/` | the frozen distractor mapping: for each of the 575 questions, the verified answer, its type, and the near and far substituted values, with the branch that produced each |
| `quality_control/` | the blinded semantic review of all 575 records, the 39-question repair list, the analysis-set membership lists, the cross-fitting folds, and the build verification reports |
| `scored_episodes/` | per-episode scoring records for the fifteen evaluated arms (five models × three formulations) — one file per arm |
| `summaries/` | the aggregation layer every table is read from: per-arm metrics, pooling across formulations, and the paired comparisons with their intervals and Holm families |
| `finetuning/` | paired changes and levels for the SFT and DPO arms, plus the 70B scale point; see the README in that directory |
| `distractor_distance/` | the near-versus-far contrast and the protocol-sentence ablation |
| `acknowledgment/` | the two-annotator round behind the 0-of-120 acknowledgment result |

`scored_episodes/` is the scoring layer, not raw generations: for every episode
it holds the extracted answer span and the decisions taken on it
(`followed_tool`, `kept_memory`, `final_correct`, `outcome`, `abstain`,
`ack_proxy`, …) together with the arm's aggregate block. Full model output text
is not retained. `CLAIM-TO-FILE.md` maps each reported number to the file that
carries it, and names the two families that cannot be recomputed from this tree.

## Rebuilding the benchmark from source

ToolHop is **not vendored** — it is Apache-2.0 from ByteDance. Download it and
check the hash before anything else:

```bash
curl -L -o ToolHop.json \
  https://huggingface.co/datasets/bytedance-research/ToolHop/resolve/main/data/ToolHop.json
sha256sum ToolHop.json
# must be 0a51f71a44b7025645e452123af3caf2e348301922af91778e268db0188a7fab
```

From the extracted archive:

```bash
cp -r results code/results && cp ToolHop.json code/
cd code && git init -q . && git add -A && git -c user.email=a@b -c user.name=c commit -qm extract
PYTHONUTF8=1 PYTHONPATH=. python scripts/build_episodes.py \
  --mapping "$PWD/results/construction/distractor_mapping_575.json" \
  --toolhop "$PWD/ToolHop.json" --out-dir "$PWD/out" --stage-dir "$PWD/stage"
```

which prints three formulations of 2,875 episodes each.

### The shipped mapping is redacted, so its hash differs from the one recorded

Read this before checking hashes, because the obvious check fails by design.

Each benchmark file records `mapping_sha256 =
560dea277c736caecf5a6f4c6396a0ab00fda962f1891ef7bfa2ca066569e3d7`. That is the
hash of the mapping **as the builder read it**. The copy in this archive hashes
`b4da7db580f9984e994779d726470b2924ab5235796948fab1e1590ae0e14a06`, because six
fields were rewritten on the way out:

| JSON pointer | shipped value | why |
|---|---|---|
| `/sources/annotator_a/path` | `<path-redacted>` | local filesystem path |
| `/sources/annotator_b/path` | `<path-redacted>` | local filesystem path |
| `/schema_version` | `memtoc-frozen-distractors-v2` | internal run label |
| `/correction/row` | `memtoc-v1` | internal run label |
| `/correction/supersedes` | `memtoc-v0` | internal run label |
| `/correction/note` | leading run label dropped | internal run label |

The first two were absolute local filesystem paths to the two annotation
workbooks. The last four named rows of our internal experiment journal, which
carry no meaning outside it; they are renamed here exactly as the directory
names are, and the same rename is applied to the assert in
`code/scripts/build_crossfit_folds.py` that checks `/correction/row`, so the two
still agree. Nothing else in the file differs, and none of the six fields is
read during construction, so the benchmark is unaffected.

Verified, not asserted: rebuilding with the redacted mapping and the ToolHop
above yields 2,875 episodes per formulation whose `episode_id` set is identical
to the shipped one, with **exactly 39 episodes differing in content** — precisely
the 39 entries in `benchmark/repair_overlay/reference.json`. The other 2,836
match field for field.

`toolhop_sha256` inside the benchmark files is the ToolHop hash above and is
unaffected by any of this.

## Important notes on recomputing the metrics

Four things about the layout will otherwise cost time.

1. **Read `keep_strict`, not `keep_flag`, and pick the right summary file.**
   Each retention cell exists in more than one form, and the near-miss is the
   expensive part.

   | Table | File | Field |
   |---|---|---|
   | prompting ladder — original, warning, source-priority, abstain-and-flag | `results/summaries/arm_metrics.json` | `judged.<model>.control_<prompt>_pv<k>.keep_strict` |
   | canonical and presentation arms | `results/summaries/arm_metrics_qminus.json` | `keepQ`, which additionally requires that the verified answer is not a substring of the question |

   Worked example, the published warning-prompt row of the prompting ladder —
   `judged.llamai.control_warn_pv0` gives `keep_strict 0.2099`,
   `tool_gold_follow 0.7513`, `prior_bw 0.6534`, `nc_CRA 0.7513`,
   `te_abstain 0.1642`, i.e. the printed `21.0 | 75.1 | 65.3 | 16.4`. In the
   same cell `keep_flag` is `0.2346`; it is an older field that also counts
   `both`, and it is not what the paper reports. Note that
   `arm_metrics_qminus.json` does **not** contain the `control_*` arms at all,
   so it cannot be the source for that ladder.

2. **Field names in the result files are the internal metric names.**
   `keep_strict` is correct-answer retention (Ret.), `tool_gold_follow` is
   correct-tool following (Tool), `prior_bw` is incorrect-tool following where
   neither source is correct (Wrong), and `te_abstain` is tool-error abstention
   (Err.). Model keys are `base`, `llamai`, `gemma`, `qwen`, `mistral`;
   formulation keys are `canonical_pv0`, `canonical_pv1`, `canonical_pv2`, in
   the order Reference, Paraphrase A, Paraphrase B.

3. **Pooling across formulations is the unweighted mean of the three**, not an
   n-weighted mean.

4. **Do not let git rewrite line endings on `ToolHop.json`.** A CRLF-converted
   copy fails the pinned hash. (It will not, however, produce
   `gold mismatch: (32, 0)` — that signature comes from a non-UTF-8 locale,
   which is why the rebuild recipe above sets `PYTHONUTF8=1`; the offending row
   is `S. P. L. Sørensen`. It reproduces with a byte-correct ToolHop.) Keep
   `*.json -text` in `.gitattributes`.

Tool-error abstention and no-conflict accuracy are deterministic-scorer values
by design; retention, accuracy, correct-tool following and incorrect-tool
following are judge-consistent.

## Notes on the code

`code/memtoc/` is the package and `code/scripts/` the pipeline that builds the
benchmark and aggregates the runs. The pipeline scripts are the compute node's,
unmodified except that the directory names used in this archive have been
substituted for the internal ones, so that every path they mention resolves in
this tree. They were written to run **in place** on that node, which costs two
further things on any other machine — both environmental, neither needing a
source change:

1. **Give absolute paths, and put the inputs under `code/`.** The scripts
   compute `ROOT = <script>/../..`, i.e. `code/`, and call
   `Path.relative_to(ROOT)` when recording provenance. Inputs outside `code/`,
   or relative paths, raise `ValueError: ... is not in the subpath of ...`.
2. **`git init` the extract.** Provenance blocks record `git rev-parse --short
   HEAD`; outside a repository the call exits 128 and takes the run with it.

Adapters, model weights, server paths and credentials are deliberately absent.

Source comments and a few provenance notes are in Russian; they are the authors'
working notes and reference an internal workspace that is not part of this
archive. Nothing in the pipeline depends on them.

## Provenance

The released benchmark was produced by
`code/scripts/build_episodes.py` from the frozen distractor mapping
in `results/construction/`, under prompt template version 2 and random seed
20260721, and then had the repair overlay applied. All evaluation runs
used greedy decoding (temperature 0), one process per arm.

## Citation

```bibtex
@misc{varlamov2026memtoc,
  title         = {MemToC: Benchmarking Memory-Tool Conflict Resolution in Large Language Models},
  author        = {Varlamov, Arseniy and Zinnatullin, Rishat and Rykov, Elisei and Panchenko, Alexander and Alimova, Ilseyar},
  year          = {2026},
  eprint        = {2608.26295},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2608.26295}
}
```
