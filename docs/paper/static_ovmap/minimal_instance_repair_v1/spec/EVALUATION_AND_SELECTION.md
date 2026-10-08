# Evaluation, diagnostics, selection and resource tables

## 1. Fixed benchmark and reference semantics

Use exactly the eight Replica scenes and eighteen CF18 captures in PROTOCOL_SPEC.
The same original planned/completed frames, RGB-D/poses, category arrays, target
coordinates and fixed nearest/matched map remain bound. CF18 is 18 captures of
seven physical families, not the full ScanNet200 validation set. All current
cohorts have prior exposure. New results are retrospective exploratory evidence.

IR00/IR01 correspond to actual EV00_D2/EV01_G1_V2, not hand-entered rounded numbers.
Read parent per-scene and pooled receipts. All new structural arms hold the 3D
surface fixed, but change instance partitions. Tables must say **fixed surface**,
not 'fixed masks'. Semantic-only arms preserve instance masks.

Do not add random scene replacements, a positive-only subset, per-dataset models,
or a new standard evaluator to rescue results. The default active branch remains
N0_UNCHANGED regardless of the research recommendation.

## 2. Diagnostic opportunity: separate full R, residual K and whole-object unions

Run one cached/CPU evaluation-only diagnostic after the GT-free proposal bank is
locked. Prediction modules cannot import its labels or its decisions. For each
old omitted candidate retain:
- source rows/physical area of R and K and clipping ratio;
- evaluation-point mask obtained with the unchanged nearest/matched projection;
- best class-agnostic IoU with actual evaluator-eligible GT for R, K, and any
  pre-generated attachment/group hypothesis containing it;
- whether its best GT is already matched by an incumbent prediction;
- baseline assigned class and error type, actual ignore events and score ties.

R masks may overlap incumbent supports and so are diagnostic only. A high IoU for
R does not authorize stealing P_j in the predictor. Record candidate combinations
which would require forbidden edits rather than claiming them as reachable gains.

For original incumbents, measure best eligible-GT IoU and old D2 class correctness
for diagnostics only. Separately report how many selected low-margin incumbents
are geometrically matchable and actually wrong; do not choose the subset by this
answer.

At .25, .50, .75, .90 compute maximum bipartite matching from the fixed candidate
library to eligible GT, ignoring semantic classes. Evaluate both the current
partition and the diagnostic candidate library. Name the latter **relaxed
candidate-set matching ceiling**: its overlapping/incompatible hypotheses may not
be simultaneously realizable. It is not an AP bound or a deployable oracle method.
No ranking or threshold is tuned with this value. Empty opportunities are reported,
not used for data-specific prediction branches.

The actual changed outputs are also analyzed at every loaded APall threshold.
Report additions/losses of unique GT matches and changes of existing-object IoUs.
Use named GT-ID sets with scene identity to avoid double counting captures. Report
class-agnostic matching separately from released class-aware matching.

## 3. Authoritative scientific evaluation

Use SceneEvaluator / released assign_instances_for_scan and trace_released_matches
against each actual partition-specific registry. Rank mode is OFFICIAL_CURRENT_CLASS.
APall averages the loaded nine overlaps .50,...,.90; validate mathematical values
with tolerance, do not replace the loaded NumPy vector. AP25 is separate. Preserve
minimum region-size and ignore rules. Small/unknown objects must not disappear
from semantic evaluation simply because their instance mask is omitted by the
released size test.

Final source owner maps are mutually exclusive. Build both instance and semantic
outputs from the same map and owner-to-class decisions. Do not report multi-label
or overlapping proposals as if they were the same protocol. Any labels on added
surface must count normally, including errors.

Run ONE released dataset evaluation call over all ordered scene manifests for each
of nine methods and each cohort. Do not use the mean of scene AP as pool AP. Keep
apall/ap50/ap25/miou/macc as fractions in JSON and convert once to percentages in
printed tables; deltas are percentage points. Run full per-class/scene records,
including negative and tied outcomes. Records whose full scoring input is exactly
identical may reference an existing receipt with a proven alias.

Do not combine the former oracle/counterfactual pools with official metric tables.
A structural support edit cannot inherit a previous method's AP even when its
classification labels look unchanged.

## 4. What counts as mechanism evidence

Prespecified comparisons are IR03−IR02, IR05−IR04, IR07−IR06, IR08−IR05,
IR08−IR07; all methods also compare to IR01 and IR00.

For every structural arm retain:
- complete input graph/library sizes, observation-bank sizes and edge abstentions;
- selected/proposed/verified/applied operations, conflicts and canceled unions;
- number and physical area of moved residual rows; original-incumbent rows moved
  must be zero; host support growth and disappeared constituent output IDs;
- best-IoU deltas, unique new/lost GT matches at .50 and .75, overmerge indicators
  derived AFTER prediction, definite/ambiguous scoring entries and per-class AP.

For semantic arms retain:
- original selection identity and unchanged output masks;
- original class, proposed class, accepted change, full/core/intersection success;
- same-frame duplicate masks removed, number of pose-distinct views;
- which stability test prevents a change;
- correct-to-wrong and wrong-to-correct results after prediction, with geometrically
  matchable and insufficient cases separated.

Do not equate a lower fragment count with improvement, a larger owner count with
recall, or disappearance of FP entries with proof of a new TP. Support growth and
label changes can alter official within-class area normalization and cross-scene
ranking; retain old/new rank changes. Score ties remain ambiguous where the released
matcher does not identify a unique source. No edits targeted by true TP identities.

## 5. Exact research recommendation

Use unrounded fractions and epsilon=1e-10. Candidate passes only if:
1. all five Replica metrics >= IR01_G1−epsilon;
2. all five CF18 metrics >= IR01_G1−epsilon;
3. CF18 APall > IR00_D2+epsilon and CF18 AP50 >= IR00_D2−epsilon.
A separate MATERIAL_TARGET_MET flag requires CF18 APall−D2>=.001−epsilon
(0.10 percentage points). This is an engineering magnitude flag, not significance.

Among passing new methods, lexicographically compare CF18 APall, CF18 AP50,
Replica APall and CF18 mIoU, using epsilon ties. If still tied use the fixed
simplicity_order from the JSON. No passing method means recommend IR01_G1.
The fixed combination is never a retrospective per-object/per-cohort mixture.
Do not silently relax the five-metric target. A nonpassing but useful tradeoff may
be documented in a separate Pareto discussion, not relabeled as a passed target.

A method can meet the performance target without proving its proposed mechanism
novel or necessary. The relevant paired simple-control differences must be stated
separately. Conversely an implementation-complete negative result is valid.
The selection is retrospective on exposed cohorts; do not call it untouched test
confirmation or infer significance from many frames in the same scans.

## 6. New paired latency and memory series

Timing arm list: IR01 first; append the selected nonbaseline method if any; append
unique methods in JSON timing priority until four. Freeze this list before timing.
Run all eight Replica scenes twice with second method and scene order reversed.
This is at most 64 actual calls, not 64 new maps. A cold call has a fresh call-local
context. Common XYZ/faces/Native partition, NQF and required models are resident;
old G1 views, recovered vectors/outputs and new observer/hypothesis/AnyUp caches are
NOT resident inputs. Recompute necessary G1 v2 from its original full capture
schedule, then execute the selected repair/reread path. Within-call reuse is valid.

Use only models required by that arm. Report model load separately, absolute and
incremental allocated/reserved GPU peaks, CPU environment, thread counts, GPU ID,
original precision/TF32 flags, page-cache condition and unrelated system load.
One shared process may unload irrelevant models before reset; it must not grant
an arm another arm's features or ignore a still-allocated model in the peak.

Begin the synchronized wall timer before G1 view/BVH construction and end after
final PredictionPayload construction and official rank preparation. Hashing,
GT scoring, prediction comparisons, final scientific receipt writes are after the
timer. Any view IO unavoidably performed inside production build_views is inside
both control and candidate boundaries. Refactor only symmetrically or report the
extra operation; never hide a candidate's observation construction outside time.

Exclusive stages: base G1 recovery, additional observer/proposal work, repair
verification, additional FC/AnyUp, final payload. GPU events annotate but do not
add to wall stages. Report unaccounted wall residual rather than manufacturing
perfect sums. Both rounds and all valid calls count; do not pick minimum times.
A timed output must match its scientific result content after the timer.

Since this work changes masks/classes, accuracy is NOT inherited by 'prediction
parity with G1'. Timing parity compares each timed arm with its own new scientific
output. The IR01 control alone must reproduce parent G1. Compare only this
same-boundary new series; old 18.85, 19.89, 22.71 seconds remain historical records.

## 7. Tables and deliverables

Table 1: nine rows, Replica and CF18 APall/AP50/mIoU; all five metrics in the same
store and supplemental output. Include explicit IR01 and D2 paired differences.
Table 2: five pairwise comparisons with AP/mIoU, structural edits or relabels,
unique GT50/75 added/lost, and wrong-to-right/right-to-wrong semantic changes.
Table 3: measured arms, AP/mIoU, seconds/scene, allocated/reserved GiB, actual FC
frame inputs, AnyUp QK/region calls and observer frames. Unmeasured times are null.
Source identities, supports and all metrics live in one canonical result store.

Preserve the prior OVI reported/reference tables without using their values as new
same-protocol remeasurements. This task does not reproduce new external baselines.
No table row may combine AP from one partition with mIoU from another.

Store compact operation ledgers, category arrays, geometry/partition digests,
classification-source identities, old/new score provenance, branch/config/producer
pins, resource events and exact reproduction commands. The large observation maps,
licensed scans and model weights remain outside Git with reconstructible manifests.
