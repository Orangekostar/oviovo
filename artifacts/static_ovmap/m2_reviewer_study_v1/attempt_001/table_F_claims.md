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
