# Revised V2 acceptance ledger — completed evidence review with limitations

This ledger records the completed backend screening and static confirmation
evidence review. It does not claim perfect protocol adherence, complete launch
provenance, or external dynamic confirmation. Final remote delivery is bound
separately from the experimental code provenance.

| Requirement | Inspected evidence | Current finding |
|---|---|---|
| §0.4 four-family DEV full maps | `all_method_results.json`, 85 scored rows; complete local-background matrix | Complete numeric matrix: room0 27, B3 29, H2 29; 83 frozen rows plus two late supplements |
| §0.4 static confirmation | `room1_confirmation_status.json`, 11 method JSONs, invariant report | Four families plus direct controls evaluated after frozen selection |
| External dynamic confirmation | Existing Office asset-status record and handoff limitation | Not run; raw assets missing, no generalization claim |
| Source/owner separation | Geometry audit across 58 dynamic readouts; room1 source and metric invariants | Fixed source geometry; M3 ownership edits separately recorded |
| Ghost and recovery attribution | Ghost numerator/denominator checks; 29-method recovery report | Zero-denominator Ghost disclosed; 2,939 rows/666 physical samples retained |
| M1 matched controls/local background | Six complete local-background scores, cached/reencoded controls | Negative results retained, coverage mismatch counted |
| M2 patch versus propagation | Point/patch/geometry/boundary rows; frozen registries | Missing top-k boundary variant completed after confirmation with existing parameters; nine headline metrics match geometry-only, both states fail eligibility |
| M3 consensus and resem | Owner-only and same-partition resem rows, independent-mask registries | Both tasks preserved; dynamic AP remains N/A |
| M4 trained weights and matched pooling | Model manifest, paired DEV/room1 scores | Official trained checkpoint used; zero new training updates |
| §10.2 selection | `selected_configs.json`, immutable source snapshot and SHA binding | Frozen before room1; metric and eligible winners distinguished |
| §10.3 final table fields | Inspected 85-row JSON; provenance and coverage tests | Model/source, checkpoint, observed case domain and per-method confirmation states integrated; unrecorded cost/seed fields remain explicit nulls |
| §10.4 geometry and local display | Full four-view export audit; recovery source CSV and figure | All final H2 rows/connectivity verified; successes and failures displayed without outcome-selected ROI |
| §15.1 documents | Plan, results, family discussion and handoff files | Present; stale early status statements corrected; tables embedded/linked rather than duplicated |
| §15.2 upload boundaries | Model/map manifests and artifact index | External weights referenced; full PLY local-only; compact results and figures pushed |
| §15.3 code/delivery versions | `code_provenance.json`, Git history comparisons | Exact launch HEAD absent from old score records; narrower file-level evidence and limitation recorded |
| §15.3 remote payload verification | `f1f1c129aa8941877588bd3eb3e4b79984f6b2b8` local/remote match; GitHub API readback of result and handoff | Both decoded file hashes match local bytes; subsequent receipt-only tip checked separately in final handoff response |

All task sections were checked against the linked code and evidence. The
199-entry compact index verifies the current tracked artifacts, excluding itself.
Seven focused summary tests pass, including rejection of modified frozen scores
and separation of late supplements. This supplements the 25 core tests below;
no full-repository test sweep was used. Original selection and all room1 files
remain unchanged. Delivery verification is recorded separately; these checks
do not erase the two substantive limitations below.

Audit correction: the strongest new dynamic M1 top-k combination had point,
patch-only and geometry-graph results at freeze, but lacked the boundary variant
required by §6.4. It was completed after room1 confirmation as an explicitly
late supplemental DEV control. Existing selection and confirmation files remain
immutable; the new result must not be represented as evidence available at freeze
or used to retune the confirmed configuration. This timing deviation is retained
in the final protocol discussion rather than hidden by regenerating the snapshot.

## Focused implementation checks

The following existing suites were executed during acceptance: multiview
semantics, surface readout graph, surface mask consensus, mask adapter readout
and surface readout metrics (25 tests passed). Source review and the room0
consensus registry confirm the shared two-view/0.60-score/0.15-margin readout,
original-owner residuals and explicit local adaptation attribution.

| Clause | Inspected implementation/test evidence | Limit of evidence |
|---|---|---|
| §5.2–5.3 view aggregation | `surface_multiview_semantics.py` tests; cached/reencoded method registries | Tests exercise aggregation; full-run scores and feature bindings provide separate execution evidence |
| §6.2 source topology | Different-visit welding rejection, disconnected-triangle and full-current-row tests | Does not by itself prove every dataset input; complete geometry audits remain required |
| §6.3 pseudo/real unary | S2 confidence/physical-mass test; real-posterior and tie tests | Pseudo unary remains explicitly distinct from an encoder posterior |
| §6.3–6.4 unsupported/zero λ | Zero-support messages blocked; λ=0 patch-only tests | Boundary completeness has the late top-k correction noted above |
| §7.1 observer counting | Undersegmented-observer removal and per-frame support-union tests | CROVE adaptation, not an unmodified MaskClustering reproduction |
| §7.3 ambiguity/residual | Tie/single-view rejection and original residual preservation tests | Real whole-map M3 metrics are needed in addition to these unit checks |
| §8 paired Adapter | Raw-feature pooling/projection order, empty masks, activation resize tests | Official checkpoint loading and full-map paired results are separate evidence |

Frozen registries retain their original creation-time descriptive status fields
(for example an early pending resem label). They are not rewritten after
confirmation to make historical snapshots look current. Current completion is
determined from scored result records and the confirmation status, while the
registries remain the exact parameter/input contracts used by the frozen binding.

Additional source-bound checks cover the remaining detailed clauses:

| Clauses | Evidence inspected | Finding |
|---|---|---|
| §0–2 fixed inputs and confirmation | `input_selection.json`, input/confirmation registries and completed room1 records | room0/room1 use stride-10 frames within 0–1999; Apartment visits retain the 766–1021 and 1217–1472 mappings; Office lacks raw inputs |
| §3 bridge | `bridge_B3.json`, `bridge_H2.json`, focused role/input tests | Legacy and pointwise nine-metric deltas are zero; all current rows covered; owner changes do not determine roles |
| §4 shared observations | Cached/reencoded banks and independent CropFormer registries described in results | Cached six-crop averages are not represented as independent scale features; reencoding and real 2D masks have separate input provenance |
| §8 official paired head | `mask_adapter_readout.py`, model manifest, `adapter_reference_check.json` | Strict core loading; raw 1536/head 768; per-map projection before averaging; three real masks match reference, chunked logit maximum error 0.000398 |
| §9 state interaction | Separate B3/H2 scores and source-index/recovery audits | Same graph parameters, separately constructed state graphs; 2,939 recovered rows evaluated after prediction without changing validity thresholds |
| §11–13 execution and resources | Actual runner commands and stage-runtime records in results/handoff | Full predictions precede scoring; shared caches used; missing historical runtime/seed values are null, not fabricated; no new training |
| §14 scientific scope | Four-family discussion and confirmation table | Negative results retained; no SOTA or external dynamic-generalization claim; role-conditioned metric gains are not geometry repair |

The initial input-selection record deliberately preserves its pre-run PARTIAL
and PENDING fields; current completion is reported by the final aggregate and
room1 confirmation status, not by rewriting that initial record.
