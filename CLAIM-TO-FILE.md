# Where each reported number comes from

Every quantitative claim in the paper and in the supplementary document resolves
to one file in this archive. The table below is that mapping. Files under
`results/` are the scoring and aggregation layer; `tools/verify_paper_numbers.py`
re-derives 180 of these values and diffs each against the printed number.

Metric names follow the paper: **Ret.** correct-answer retention, **Tool**
correct-tool following, **Wrong** incorrect-tool following, **Err.** tool-error
abstention.

## Main paper

| Claim | File | Where in the file |
|---|---|---|
| Table 1, original-checkpoint rows (Ret./Tool/Wrong/Err., four models) | `results/summaries/pooled_across_formulations.json`, `results/summaries/arm_metrics.json` | `pooled_instructs.<model>`; `det.<model>.canonical_pv<k>` |
| Table 1, SFT and DPO rows, as paired changes | `results/finetuning/paired_deltas.json` | `judged.crossfit.<model>.<sft\|dpo>` |
| Table 1, SFT and DPO rows, as absolute levels | not re-derivable here — see *Not in this archive* | — |
| Retention 6.5–17.1, correct-tool 86.0–93.1, incorrect-tool 78.4–86.0 | `results/summaries/pooled_across_formulations.json` | `pooled_instructs` |
| 6,504 evaluation episodes = 542 × 4 × 3 | `results/quality_control/semantic_review_575.json` | 575 rows minus 33 `EXCLUDE` |
| Fine-tuning success criterion, per model and objective | `results/finetuning/paired_deltas.json` | `judged.dpo_criterion.pooled`, `judged.holm_families.pooled` |
| Formulation sensitivity and the stability rule | `results/summaries/pooled_across_formulations.json` | `orders_by_scope`, `stable_core` |
| One protocol sentence: Ret. −1.9, Tool −9.3, no-conflict −5.6 | `results/distractor_distance/summary.json` | `judged.protocol` |
| Prompting across four models and three strategies | `results/summaries/arm_metrics.json`, `results/summaries/paired_comparisons.json` | `judged.<model>.control_<strategy>_pv<k>`; paired family B |
| Distractor distance: Ret. +19.4 on the pretrained checkpoint, null on the four instruction-tuned | `results/distractor_distance/summary.json` | `judged.far` |
| Presentation format: passage frame vs executed-tool frame | `results/summaries/paired_comparisons.json` | paired family A |
| Conflict acknowledgment 0 of 120, interval [0.000, 0.030] | `results/acknowledgment/annotation_round.json` | `acknowledgment`, `legacy_anchor` |
| Base-vs-instruction-tuned paired comparison | `results/summaries/paired_comparisons.json` | paired family C |

## Supplementary document

| Appendix | Claim | File |
|---|---|---|
| B | Construction counts of Section 3.2 (605 → 575, branch sizes, answer types) | `results/construction/distractor_mapping_575.json` |
| B | Blinded semantic review: 503 pass, 39 repair, 33 exclude | `results/quality_control/semantic_review_575.json` |
| B | The 39-question repair overlay from that review | `benchmark/repair_overlay/`, `results/quality_control/repair_list_39.json` |
| B | Cross-fitting folds: 287 questions from 154 chains, 288 from 155 | `results/quality_control/crossfit_folds.json` |
| C | Prompting ladder, four models × four prompts | `results/summaries/arm_metrics.json` |
| C | Criterion evaluated per formulation | `results/finetuning/paired_deltas.json` |
| C | Seed replication of the fine-tuning effect | `results/finetuning/paired_deltas.json` (`judged.seed_agreement`) |
| C | Distractor distance, full estimates | `results/distractor_distance/summary.json` |
| C | Answers from neither source (third-entity share) | `results/summaries/arm_metrics_qminus.json` (`bw_gold`) |
| C | Case sizes per model and formulation | `results/summaries/arm_metrics_qminus.json` (`n_arb`, `n_bw`) |
| D | Acknowledgment round, agreement 118/120 | `results/acknowledgment/annotation_round.json` |
| E | Fine-tuning contrasts as levels on the paired support | `results/finetuning/absolute_levels.json` |
| E | The 70B scale point | `results/finetuning/paired_deltas.json` (`judged.scale_70b`) |
| E | Reverse transfer from the external set | `results/finetuning/paired_deltas.json` (`judged.reverse_transfer`) |
| B | Topic and answer-type distribution | `tools/recount_topics.py` over `results/quality_control/semantic_review_575.json` |

## Not in this archive

Three things are named here rather than shipped, so that a reader looks for them
once and stops.

1. **Raw model output text.** What ships is the scoring layer over it:
   `results/scored_episodes/` holds, for every episode of the fifteen evaluated
   arms, the extracted answer span and the scoring decisions taken on it. The
   generations themselves are not retained.
2. **The absolute SFT and DPO levels of Table 1.** They are computed by
   `code/scripts/build_finetuning_summary.py` over the 48 cross-fitted arms,
   on a support convention that neither pooling the two folds nor the paired
   intersection reproduces. `results/finetuning/absolute_levels.json` gives the
   levels on the paired-intersection convention instead, with the caveat stated
   in `results/finetuning/README.md`; the paired *changes* those rows summarise
   are checked by the reproduction harness.
3. **Adapter weights, model checkpoints and serving configuration.**

## Other evidence in the paper

Three measurement layers were produced on the previous, pre-curation version of
the benchmark and are labelled as such wherever they appear: the SAE-steering
probe, the acknowledgment-detector validation with its two-stage prevalence
estimator, and the switch-composition analysis. They are described in full in
the supplementary document. Their inputs are not part of this archive, because
they describe a benchmark version that this archive does not contain.
