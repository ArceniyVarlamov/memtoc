# Fine-tuning results

Two files, both read on the same support.

| File | What it holds |
|---|---|
| `paired_deltas.json` | every paired change of a fine-tuned arm against its original checkpoint: change, 95% interval, Holm family, success-criterion verdict, seed replication, and the 70B scale point |
| `absolute_levels.json` | the same contrasts written as **levels** — the rate of each arm and of its comparison arm, on the identical paired support |

`absolute_levels.json` exists because the paper's tables read as "it became this
much", and a level cannot be recovered from a change by arithmetic: a case is
defined by the arm's own parametric answer, so fine-tuning moves which questions
belong to it, and each contrast lives on the intersection of the two supports.
Its internal gate reconstructs all 672 changes from the levels; no new number is
introduced.

## Three things to know before quoting a number from here

**1. The level of the original checkpoint here is bound to the arm it is paired
with.** It is computed on the intersection with one specific fine-tuned arm, so
for the same model it varies slightly from row to row — Llama-Instruct
retention, pooled across formulations, ranges 0.164 to 0.173 — and it does not
equal the full-benchmark level of 0.166 in
`../summaries/pooled_across_formulations.json`. That is a difference of support,
not a disagreement. Quote the level that stands next to its own change.

**2. These levels are not the ones printed in Table 1 of the paper.** On the
paired-intersection convention used here, Llama-Instruct SFT retention comes out
at 32.55 where the table prints 31.6; pooling the two cross-fitting folds gives
a third value, 31.08. The printed value comes from the support convention inside
`../../code/scripts/build_finetuning_summary.py` applied to the 48 cross-fitted
arms, which are not part of this archive. The *changes* all three conventions
summarise do agree, and those are what the reproduction harness checks.

**3. Support drift is reported, and it is large in two places.** Because
fine-tuning moves case membership, every contrast records how many questions
entered and left (`n_treated_only`, `n_untreated_only`). The largest drift is on
Mistral's SFT arm for correct-tool following (+283 / −97) and gemma's SFT arm
(+168 / −105); those two changes have to be read together with that fact.

## Checks this layer passed

- **Change reconstruction.** For every contrast,
  `rate_treated − rate_untreated == delta` from `paired_deltas.json`:
  672 contrasts, all pass.
- **Out-of-fold coverage.** For each cross-fitted arm the two folds have empty
  intersection by question, and their union is the full 542-question analysis
  set.
- **Repair overlay applied.** All 81 evaluated arms carry the repair overlay;
  26 of the 39 repaired questions fall in one evaluation fold and 13 in the
  other, and the nine full-set arms carry all 39.
- **Scoring-layer sensitivity.** Judge-normalised against deterministic
  scoring: 5 of 88 tests differ in significance, all on single-formulation
  slices and all borderline; none on the pooled slice. The largest difference
  in a change is 0.0166.

## Two conventions worth stating

- **Correct-tool following and no-conflict accuracy share their support by
  construction** — each question contributes exactly one episode per condition,
  and the case is a property of the question–model pair. Their sample sizes
  therefore match in every arm while the changes differ in six of eight. They
  are not two independent pieces of evidence.
- **The 70B row is a scale point, not a fine-tuning effect.** No original-
  checkpoint 70B arm exists, so the cross-model contrast is taken against the
  Llama-Instruct original checkpoint on common support and is labelled as such.
