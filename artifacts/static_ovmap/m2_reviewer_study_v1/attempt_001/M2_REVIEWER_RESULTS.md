# M2 reviewer evidence results

Measured study complete; scientific support is mixed. Report identity: `4357cf36788db730187506848266122131aad3b51dd643e43dc2dbc9b96e55ea`. Deployment remains N0.

## Scope and interpretation

Ten fixed scenes: two ScanNet CAL scenes and eight historically exposed Replica
transfer scenes. CAL was already exposed to the learned query checkpoint.
CAL selected A2 (N0+static S2) and static S2 as the single-source comparator before
new Replica evaluations. No Replica fitting or comparator/prompt/seed selection.

The core matrix contains 260/260 measured scene/rank rows
and 26/26 Replica pools.
APall equals uAP over the evaluator's recorded0.50–0.90 overlaps; AP25 is separate.
Both frozen N0 ranks and official current-class area ranks are reported. Dataset
pools call the released evaluator over the original ordered files; scene means
are separate. No macro bootstrap is described as a pooled interval.

## Main outcome

Official-current-class Replica pooled M2_CAL minus N0: **uap +0.225 pp, ap25 -0.555 pp, ap50 -1.302 pp, miou -0.470 pp, macc +1.192 pp**.
This is a metric tradeoff, not uniform superiority. Equal-weight probability
fusion does not give equal effective score influence: replacing only native
canonical-relative scores by cosine changes the result. Common sharpening/shared
temperature largely reproduces source-specific calibration; its necessity is
not established. See full matched controls and paired scene deltas.

On157 fixed all-three-available identifiable Replica objects, M2_RAW→M2_CAL
NLL improves3.32554→1.74688 and Brier0.94642→0.66166; accuracy48.408%→50.318%.
The constant0.01 control has NLL1.74291 and Brier0.66077. Probability improvement
is conditional on this measured population, not all false positives.

## Object-level failure evidence

office1 frozen-rank AP50 falls0.25→0.19117647. The released trace adds blanket
owner80 but loses blanket owner79 and desk owner8. Both Q/S2 predict cloth for
owner79 and tv-screen for owner8, replacing correct native labels. Desk class AP
falls1→0; blanket class AP stays unchanged. All scenes retain duplicate/ignore
events at every recorded overlap; semantic diagnostics use a separate strict
unique geometry IoU>0.5 correspondence. First two corrected/harmed owners are
selected deterministically, including failures rather than flattering examples.

## Query, compute and robustness

CAL did not trigger B100/B400: GAIN fusion uAP0.0366501/mIoU0.234450 versus
COMBINE0.0324184/0.238655 and random seed mean0.0390670/0.241298.
Three random seeds remain separate; no lucky seed selection or ensemble.
Completed B200 acquisitions: 50/50.

Replica B200 CAL fusion, official-current-class released dataset pools (percent):

| policy | uap | ap25 | ap50 | miou | macc |
| --- | --- | --- | --- | --- | --- |
| Q_GAIN | 8.910 | 34.030 | 20.175 | 26.792 | 33.887 |
| Q_COMBINE | 8.378 | 33.120 | 18.425 | 24.924 | 33.170 |
| RV_Q_RANDOM_s17 | 8.344 | 34.047 | 18.918 | 23.536 | 32.063 |
| RV_Q_RANDOM_s23 | 8.426 | 33.212 | 19.374 | 24.827 | 32.380 |
| RV_Q_RANDOM_s41 | 8.972 | 35.124 | 20.537 | 24.829 | 33.279 |

New recorded visual forwards: 5565;
new crops: 33390; measured incremental
inference seconds: 1141.965.
Recorded scalar-fit seconds: 0.073547; scalar timing
complete: False. Recorded unique evaluation-cache and
pool seconds: 280.250. These are sums of measured
work, not elapsed end-to-end time or a parallel speedup estimate.
Shared mapping/frontend, source-attributed logical requests, exact operation
unions, and physical cached execution are separate in the cost ledger.
Historical peak memory and initial ad-hoc fold timing are missing, not zero.
Failed-attempt overhead is incompletely measured. These are offline cached
experiments, not an online end-to-end latency benchmark.

Vocabulary testing reused frozen image aggregates and query trajectories, with
zero new image forwards. On the same157 paired objects M2_CAL accuracy is50.318%
(original),55.414%(photo),46.497%(close-up),50.318%(16 added distractors); the last
condition has0% distractor selection. This tests controlled word sets, not global
distractor absence. Singleton softmax is identically1, so arbitrary single-text
retrieval is not supported by this formula. Prompts remain diagnostics, not a
newly selected method.

Fresh status: `FRESH_BLOCKED_INSUFFICIENT_LOCAL_UNEXPOSED_FAMILIES`. All14 authorized local physical families were
already exposed; no fresh scene was recycled or downloaded. The implemented
success path exists but cannot be claimed as real fresh validation.

## Tables and reproducible evidence

- [Table A — core](../../../artifacts/static_ovmap/m2_reviewer_study_v1/attempt_001/table_A_core.md)
- [Table B — probability](../../../artifacts/static_ovmap/m2_reviewer_study_v1/attempt_001/table_B_probability.md)
- [Table C — objects](../../../artifacts/static_ovmap/m2_reviewer_study_v1/attempt_001/table_C_objects.md)
- [Table D — query](../../../artifacts/static_ovmap/m2_reviewer_study_v1/attempt_001/table_D_query.md)
- [Table E — vocabulary](../../../artifacts/static_ovmap/m2_reviewer_study_v1/attempt_001/table_E_vocabulary.md)
- [Table F — claims](../../../artifacts/static_ovmap/m2_reviewer_study_v1/attempt_001/table_F_claims.md)

All metric CSVs store fractions, not percentages; Markdown AP/IoU tables display
percent. Raw scene rows, paired macro deltas, probability populations, source
operation identities, text receipts and external file hashes accompany the
tables. Missing prerequisites remain explicit. See the claim ledger for the
supported/mixed/not-supported/not-tested distinctions.
