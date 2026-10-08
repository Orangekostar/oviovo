# Implementation contracts

## 1. What is inherited, and what is new

Inherited evidence: the exact final G1 v2 partition, D2 N/Q/F distributions and temperatures, the original 16-or-fewer incumbent selection per scene, the original FULL masks and their AnyUp scores, original IR06/IR07 behavior and its output cancellation.

New computation: at most one ordinary coarse FC region readout on each already selected FULL mask, then five new probability-update rules and two matched historical decision controls. No new geometry, view, AnyUp network call, learned parameter, or semantic vocabulary.

This is a controlled **fixed-evidence replay** experiment. The common domain is not an automatically available deployment oracle: C must actually be calculated to establish C availability. No standalone acquisition, generalization or overall speed claim follows from replay timing.

## 2. Canonical paired evidence

For scene s and selected original incumbent i, retain:

```
(scene, owner, original_support_digest, old_class,
 ordered_valid_ids, class_text_identity,
 original_scores_N/Q/F or unavailable,
 positive_temperatures_N/Q/F,
 original_D2_probabilities,
 parent_selected_rank,
 protected_raw_zero_count,
 FULL_views[])
```

Every FULL view record binds `frame_id`, `bank`, `RGB_sha256`, mask digest/file, canonical bbox, original/resized/padded shapes, parent feature/content identity, FC physical model/text identity, AnyUp operator identity and full cosine array A. C is added to the same record with its own pooling operator identity. Store the underlying precise arrays, not just top classes or rounded probabilities.

The parent saved fields were inspected: per-region `observations` contain `frame_id`, `bank`, `type`, `mask_digest`, `scores`, and `feature_content_key`; parent plans supply RGB/mask/files/sizes. Parent frame receipts supply dense arrays/receipts. Do not guess that plan keys alone prove scores were successfully computed.

Deduplicate only identical `(frame_id, mask_digest)` FULL observations; assert one FULL per bank as in the locked plan. FULL gets precedence only when verifying the historical duplicate rule. Do not deduplicate different frames or invent diversity weights.

V_i consists of every successful parent AnyUp FULL observation, one or two. Exclude CORE and FRONTEND_INTERSECTION from new aggregate A. Parent IR07 replay continues to consume all original qualified regional evidence exactly as before.

Compute coarse C on all members of V_i. For the common-domain comparison do not truncate V_i to a successful C subset. An unavailable C in any paired FULL view yields a common-domain keep for every SU02–SU08 arm. Never replace the frame using new class confidence.

## 3. Source probabilities and aggregation

All class arrays use the frozen `valid_ids` order. The integer class ID is not the array offset. Class order must match across sources by an explicit equality check.

For available historical source X in {N,Q,F}:

\[
p_X(c)=\operatorname{softmax}(s_X/T_X)_c.
\]

Let \(q\) be the arithmetic mean of the available N/Q probabilities, normalized exactly as the existing `pool`. If at least one of N/Q is present, the original dense-group mass is \(w_F=1/2\); if F alone is available, \(w_F=1\). When F is unavailable there is no common-domain update; keep the actual G1 class. Do not create a missing F group, fill absent scores with zeros, or count N and Q twice at a new top-level grouping.

For full source availability:

\[
p_0=\tfrac14p_N+\tfrac14p_Q+\tfrac12p_F.
\]

Generally \(p_0=(1-w_F)q+w_Fp_F\); with F only the q term is absent. The production reference p0 is the existing D2 function, including normalization; reconstruct it first with exact labels and probability error <=1e-12.

For the identical paired view set V_i:

\[
a=|V_i|^{-1}\sum_{v\in V_i}s^A_v,\qquad
c=|V_i|^{-1}\sum_{v\in V_i}s^C_v,
\]
\[
p_A=\operatorname{softmax}(a/T_F),\qquad
p_C=\operatorname{softmax}(c/T_F).
\]

Use **mean per-view cosines, then softmax**, matching the parent's FULL-score aggregation. Do not replace this by averaging per-view probabilities or by normalizing an averaged visual vector. Those operations are not equivalent. Reconstruct parent `aggregate_scores` to verify the arithmetic.

A and C use the same frozen FC text/head and inherited T_F. This controls the scale choice; it does not establish that T_F is optimally calibrated for A. Do not fit T_A or a new scale in this task. Use stable softmax/log_softmax in float64; do not multiply by an additional OpenCLIP logit scale.

## 4. Common valid domain and matched controls

An object is update-eligible only if all are true:

1. It belongs to the exact parent-selected original incumbent list.
2. The original D2 probability exists and reconstructs.
3. Historical F is available with valid score/temperature/order.
4. V_i contains one or two successful parent FULL A observations.
5. Every member of V_i has completed successful same-mask C.
6. The original G1 incumbent support contains zero `raw==0` rows.

Compute this domain from fixed inputs/statuses and freeze it before any new decision evaluation. Each exclusion is an explicit reason, not an omitted data row. Resource failures are not exclusions: until resolved they are task blocks.

SU02_HARD_MATCHED copies the **parent proposed/applied IR06 class logic** for eligible incumbents and keeps old G1 elsewhere. SU03 similarly replays IR07. Reapply the original whole-incumbent raw-zero cancellation at the output layer as a defensive consistency check.

For lineage, separately reproduce the original unrestricted IR06/IR07 payloads exactly from their original selection and recorded scores, before applying this common domain. Archive comparison identities. Main matched-control rows may differ from the old reports and require actual evaluation. If common domain happens to include every effective parent edit, an exact alias is allowed after proof.

No label-based gating such as “only change an old wrong class,” no class-specific rules, no dataset switch. The same domain rule applies to all seven comparative arms.

## 5. Five new update operators

Let \(\alpha=1/2\), \(\eta=1/2\), and denote dense-group reconstruction by
\(D(x)=(1-w_F)q+w_Fx\), normalized using the same sum convention.

| ID | Exact probability rule | Fixed purpose |
|---|---|---|
| SU04_F_REPLACE | \(p_4=D(p_A)\) | Keep old N/Q, replace F |
| SU05_F_BLEND | \(p_5=D((1-\alpha)p_F+\alpha p_A)\) | Retain old dense evidence |
| SU06_F_COARSE | \(p_6=D((1-\alpha)p_F+\alpha p_C)\) | Same new views with ordinary FC |
| SU07_GLOBAL_BLEND | \(p_7=(1-\beta)p_0+\beta p_A,\ \beta=\alpha w_F\) | Same A coefficient as SU05 but dilute the whole old system |
| SU08_PAIRED_DELTA | \(p_{F,\Delta}=\operatorname{softmax}(\log p_F+\eta(a-c)/T_F),\ p_8=D(p_{F,\Delta})\) | Inject a same-observation representation difference into F only |

For full N/Q/F availability the SU05 coefficients are N=.25, Q=.25, F=.25, A=.25. SU07 coefficients are N=.1875, Q=.1875, F=.375, A=.25. The A mass is deliberately equal; differences come from which old group is preserved. When F is the only old source, SU05 and SU07 coincide. That is an expected degenerate case, not a bug or an extra method success.

The paired residual subtracts the newly computed same-view C, **not historical F**. Using historical F would confound changing the view set with changing the representation. The production repository already has `paired_residual` and `ratio_update`; they may be reused on pF with the full class set. Use `log_softmax(sF/T_F)` directly for a stable implementation if probabilities underflow. No arbitrary clipping, top-5 restriction or search for eta is allowed. An identically zero/constant residual is the identity, including unchanged class.

All five new methods take the full-vocabulary argmax in original class order, then map the index back to valid ID. There is no IR06 .01 guard or IR07 vote guard on them. Log old/new margins and third-class outcomes for analysis. None of these operators guarantees nondecreasing accuracy or AP; their role is to test alternative injection mechanisms.

## 6. Coarse representation contract

A coarse record is a **same-observation control**, not the historical F source.

Pipeline:

```
original uint8 RGB -> parent's BGR-to-RGB rule
  -> image_tensor -> exact same dense tensor/model
  -> signed_mask(original FULL mask, exact sizes)
  -> area_fallback.region_vector(model, operators, dense, signed)
  -> cosine_record with same FC text matrix and valid_ids
```

Keep original signed +/-1 mask handling, bilinear hard-support detection, v2 area fallback and frozen visual head. Both feature layers and head matter. Never pool text-space scores before the visual head and call that the same control.

New region key includes scene/owner/support, exact RGB tensor identity, mask, bbox and sizes, FC model/head/text identity, ordered classes, coarse pooling protocol/source digest, and dtype. Historical A retains its own fine-map/output grid identity. Different representations cannot share a vector-cache key.

Only missing dense images may be newly encoded. Reuse successful parent dense tensors by exact content/model identity. Do not cache by filename or owner ID alone. Do not recreate entire fine maps or invoke AnyUp to obtain a C control. A used from the parent is still evidence whose original production cost must be disclosed.

## 7. Output and scoring invariants

Except SU00_D2, all conditions have the identical G1 owner array. Original vertex coordinates, faces, TSDF and source-to-target projection remain unchanged. All G1 recovered-owner classes, unselected/ineligible original-incumbent classes, and every unknown row stay identical.

Original selected incumbent relabeling is uniform over the whole G1 owner support. When `raw==0` occurs in it, preserve the entire original class just as the parent did. This includes native painted boundary rows. Do not relabel only the nonzero raw portion, split an instance, remove the guard, or call the guard a newly learned reliability measure.

Recompute the official current-class ranks from the complete actual output. A recovered object's **rank** may change even though its mask/class is untouched, because an incumbent can enter/leave its predicted class. Record this consequence; do not incorrectly assert recovered scoring entries are numerically frozen.

The new evaluator uses the actual full G1 partition registry; administratively added mask entries must not be mistaken for new feature evidence. Include small owners in semantic evaluation even if the instance manifest drops them under the existing minimum. Same payload for AP and mIoU. Prediction metadata must not carry evaluation labels.

## 8. Proposed data structures and adapter responsibilities

Use stable small JSON/NPZ artifacts, not a new general-purpose framework:

- `source_binding.json`: repository/spec/parent identities, context/asset paths, ordered cohort and parent scores.
- `evidence/<scene>/manifest.json`: parent selections, original distributions, exact FULL pair definitions and statuses; score arrays can be deduplicated in NPZ.
- `coarse/<scene>/<frame>/receipt.json`: actual C scores, dense identity and real work counters.
- `eligibility/<scene>.json`: common domain and all exclusion/cancellation counts.
- `decisions/<scene>/<method>.json`: probabilities/refs and proposed/applied labels for every selected object.
- `predictions/<scene>/receipt.json`: complete locked payloads/identities, including unchanged-data aliases.
- `evaluation/`, `pools/`, `diagnostics/`, `selection.json`, `costs/`, `publication/`.

An item is scientific KEEP only after its input path was actually executed or a valid absence was established. An unfinished worker is BLOCKED/FAILED, not KEEP. Resume compares dependency identities, reuses successful leaves, and regenerates only descendants of a changed input. Preserve original lineage and failed attempts without rehashing the entire repository at each phase.

## 9. Minimal production tests

Use approximately ten focused tests; do not add large safety testing:

1. Existing D2 coefficients/missing-source normalization and unchanged class reconstruction.
2. Exact A/C pair identity, view averaging order, and rejection of class-order mismatch.
3. F replacement and F-group blend formula; identity when A equals F.
4. SU05/SU07 equal-A-mass coefficients and their F-only equality case.
5. Paired residual constant/zero identity and use of same-view C rather than F.
6. Common-domain keeps for absent F/views/C and one-versus-two-view cases.
7. Matched hard/stable reconstruction and their historical-to-matched distinction.
8. Whole-owner protected-raw-zero cancellation; G1 recovery mask/class preservation.
9. Exact actual-partition registry, full semantic coverage, official rank and content-based aliasing.
10. Unrounded five-metric gate, no-gain fallback and deterministic tie selection.

Two real existing-scene pilots check the end-to-end binding, one coarse FULL readout and all decision/output paths. Any affected production test must run after a bug fix. Synthetic tests accompanying this package check only the mathematical specification, not server data or CUDA correctness.
