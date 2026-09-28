# M2 claim ledger

SIMPLIFY: fixed-vocabulary score sharpening and complementary evidence with representation sensitivity; no blanket necessity or online/retrieval claim

| claim | status | matched_control | decision |
| --- | --- | --- | --- |
| N0 score source is necessary | MIXED | M2_CAL versus A3_QS2 | SIMPLIFY_NECESSITY_CLAIM |
| Query observations add useful evidence | SUPPORTED_IN_THIS_STUDY | M2_CAL versus CAL-selected A2_NS2 | RETAIN_BOUNDED_SOURCE_ABLATION |
| Static S2 contributes beyond N0+Q | MIXED | M2_CAL versus A1_NQ | RETAIN_WITH_COST_AND_VARIABILITY |
| Soft distributions outperform hard voting | SUPPORTED_IN_THIS_STUDY | M2_CAL versus A4_HARD | RETAIN_FIXED_VOCABULARY_COMPARISON |
| Source-specific fitting is required beyond sharpening | NOT_SUPPORTED | M2_CAL versus A5_T001 and A6_SHARED_T | SIMPLIFY_TO_SHARPENING_EXPLANATION |
| Native score representation affects hidden weighting | SUPPORTED_IN_THIS_STUDY | A7_COS_FIXED/REFIT versus unchanged M2_CAL | RETAIN_REPRESENTATION_CAVEAT |
| Learned query selection outperforms matched alternatives | MIXED | B200 COMBINE and random17/23/41 | NO_BROAD_LEARNED_SELECTION_SUPERIORITY |
| Gains survive official reranking and pooling | MIXED | M2_CAL versus N0; official-current-class released dataset pool | REPORT_METRIC_TRADEOFF |
| Temperature calibration improves transfer probabilities | SUPPORTED_IN_THIS_STUDY | M2_CAL versus M2_RAW on the same paired population | RETAIN_CONDITIONAL_PROBABILITY_RESULT |
| Current formula supports arbitrary single-text retrieval | NOT_SUPPORTED | Singleton softmax degeneracy fixture | RESTRICT_TO_FIXED_VOCABULARY_RECOGNITION |
| Independent fresh scenes justify broad generalization | NOT_TESTED_PREREQUISITE | Four genuinely unexposed local families | WITHHOLD_GENERALIZATION_CLAIM |

## N0 score source is necessary

Status: `MIXED`. Control: M2_CAL versus A3_QS2.

Small pooled difference; native mapping/owner/rank cost remains mandatory even when N0 scores are deleted. Eight exposed scenes do not establish necessity.

Decision: `SIMPLIFY_NECESSITY_CLAIM`. Numeric evidence is in novelty_decision.json.

## Query observations add useful evidence

Status: `SUPPORTED_IN_THIS_STUDY`. Control: M2_CAL versus CAL-selected A2_NS2.

Supports this source inclusion comparison, not superiority of learned selection; query observations add compute.

Decision: `RETAIN_BOUNDED_SOURCE_ABLATION`. Numeric evidence is in novelty_decision.json.

## Static S2 contributes beyond N0+Q

Status: `MIXED`. Control: M2_CAL versus A1_NQ.

Measured gains are conditional on the same fixed maps and extra frozen S2 model; no universal three-source necessity.

Decision: `RETAIN_WITH_COST_AND_VARIABILITY`. Numeric evidence is in novelty_decision.json.

## Soft distributions outperform hard voting

Status: `SUPPORTED_IN_THIS_STUDY`. Control: M2_CAL versus A4_HARD.

Hard votes use declared normalized counts; this does not make either distribution universally calibrated.

Decision: `RETAIN_FIXED_VOCABULARY_COMPARISON`. Numeric evidence is in novelty_decision.json.

## Source-specific fitting is required beyond sharpening

Status: `NOT_SUPPORTED`. Control: M2_CAL versus A5_T001 and A6_SHARED_T.

Small label/metric differences; common sharpening closely matches performance and has slightly better paired transfer NLL/Brier. Additional CAL supervision is disclosed.

Decision: `SIMPLIFY_TO_SHARPENING_EXPLANATION`. Numeric evidence is in novelty_decision.json.

## Native score representation affects hidden weighting

Status: `SUPPORTED_IN_THIS_STUDY`. Control: A7_COS_FIXED/REFIT versus unchanged M2_CAL.

Same native aggregate and source temperatures for fixed probe. This is a representation mechanism probe, not a newly selected deployment method.

Decision: `RETAIN_REPRESENTATION_CAVEAT`. Numeric evidence is in novelty_decision.json.

## Learned query selection outperforms matched alternatives

Status: `MIXED`. Control: B200 COMBINE and random17/23/41.

CAL gate did not show dominance; Replica was historically exposed. Seeds are separate replicates, never an ensemble or post-hoc seed choice.

Decision: `NO_BROAD_LEARNED_SELECTION_SUPERIORITY`. Numeric evidence is in novelty_decision.json.

## Gains survive official reranking and pooling

Status: `MIXED`. Control: M2_CAL versus N0; official-current-class released dataset pool.

APall improves modestly while AP50 and mIoU decrease. Macro-bootstrap intervals are not pooled intervals.

Decision: `REPORT_METRIC_TRADEOFF`. Numeric evidence is in novelty_decision.json.

## Temperature calibration improves transfer probabilities

Status: `SUPPORTED_IN_THIS_STUDY`. Control: M2_CAL versus M2_RAW on the same paired population.

Conditional on identifiable objects with all three actual sources; not calibration of every false positive. Common sharpening explains much of the improvement.

Decision: `RETAIN_CONDITIONAL_PROBABILITY_RESULT`. Numeric evidence is in novelty_decision.json.

## Current formula supports arbitrary single-text retrieval

Status: `NOT_SUPPORTED`. Control: Singleton softmax degeneracy fixture.

Prompt and fixed-distractor tests are vocabulary stress tests. They do not validate arbitrary-text retrieval or certify distractors absent in every region.

Decision: `RESTRICT_TO_FIXED_VOCABULARY_RECOGNITION`. Numeric evidence is in novelty_decision.json.

## Independent fresh scenes justify broad generalization

Status: `NOT_TESTED_PREREQUISITE`. Control: Four genuinely unexposed local families.

No eligible local family; old confirmation and Replica cannot be relabelled fresh.

Decision: `WITHHOLD_GENERALIZATION_CLAIM`. Numeric evidence is in novelty_decision.json.
