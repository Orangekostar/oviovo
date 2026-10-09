# Algorithm and interface contracts

All numeric settings are in `PROTOCOL_SPEC.json`. Choices introduced here are **proposed fixed experimental settings**, not findings from the repository or guarantees of improved AP. Symbols refer to prediction-side geometry unless explicitly marked as evaluation-only.

## A. Preserve what is already effective
Let `G1.owner_ids`, `G1.semantic_labels`, its geometry identity, and its original-source-to-evaluation projection be the baseline. The main experiment never changes source XYZ, face indices, original owners, any G1 instance mask, or G1 recovery decisions. It can change only a selected original D2 incumbent's **whole-instance class**. All unselected, recovered and unknown labels remain bitwise unchanged. Recompute official area-based ranks after a class change; do not reuse old ranks or substitute the FC similarity for them.

Reconstruct `p0` with `backbone_wave1.readouts.fuse_readout(..., "D2", ...)`, using actual source availability and frozen temperatures. Do not infer p0 from the winning class. Fully available D2 has N/Q/F weights 1/4, 1/4, 1/2. Source absence is handled by the original group normalization. No new scalar is fitted.

The full 26-scene input is the source-preserving-update parent, not the four-scene SAM-V parent. `source_preserving_update.load_scene()` is a useful reader but writes verification memos into `binding['output_root']`: an adapter must redirect those writes to the new task and retain the inherited parent references. Its legacy `units` list contains only old selected objects; construct required new units explicitly rather than assuming all newly selected IDs are present. Do not invoke its old branch-specific binder or runner.

Predictor inputs may contain fixed predicted labels, raw masks, RGB-D, poses, source scores, text embeddings and geometry. Target labels, best GT assignments, GT nearest-neighbor mappings and current evaluator output are not allowed in target selection, observation choice, scores or update logic. Keep evaluation-only data in a separate namespace.

## B. Stable predicted-surface sites and geometric observations

### B1. The two representations have different jobs
1. **Legacy raster**: reuse `SourceRowProjector` with the inherited exact depth tolerance, camera convention and full BVH. It maps each valid pixel to an original source row. `FULL(owner, frame) = valid & (G1.owner_ids[source_row] == owner)`. This remains the FC mask for all query policies. No morphological change or new segmentor is permitted.
2. **Canonical surface sites**: provide repeatable physical locations for coverage/overlap measurements, independently of which representative row a ray selects. They never repaint FULL masks or the final map.

Build canonical sites from the existing prediction mesh. Normalize negative zero coordinates to positive zero. Do not epsilon-weld nearby positions. Identify duplicate geometric triangles by their three lexicographically sorted exact XYZ tuples. For a duplicate group whose per-coordinate owner assignment agrees, count triangle area only once. If exactly coincident duplicate triangles disagree in owner assignment, mark that group ambiguous and exclude it from the coverage quadrature; keep it in the actual map and full occlusion BVH. Report ambiguous/degenerate/duplicate counts and areas. Zero-area triangles do not contribute. For each retained triangle assign area/3 to each vertex site `(G1 owner, exact XYZ)`. Sites with the same XYZ but different owners are separate, never joined across objects. This is a conservative support proxy, not corrected GT geometry.

All arithmetic for area and visibility metadata is float64. Stored original XYZ/face arrays are unchanged. Cache this preparation by geometry and G1 owner-partition identity.

### B2. Deterministic bounded area quadrature
For each selected object, sort its positive-area unique sites lexicographically by XYZ. If at most 4096, retain all sites with their areas. Otherwise use exactly 4096 midpoint quantiles of cumulative surface area: quantile q lies at `(q + .5) * A / 4096`; select the site whose right-open cumulative interval contains it (searchsorted side='right', clamp final index). Repeated selections of a large-area site merge into one sample with weight `count * A / 4096`. This gives at most 4096 locations and conserves total estimated area. It is a deterministic sampling proxy, not an exact area integral; record unsampled-site count, selected count and sampling convention. Site IDs depend on coordinate/content, not array ordering.

### B3. Direct visibility at a physical location
For point x and frame camera `T_cw = inv(T_c2w)`, compute camera coordinate y, z = y[2] > 1e-6, image coordinate `(u,v)` from original K. Reject if the floating coordinate is outside the original image; sample depth and FULL at `floor(u+.5), floor(v+.5)`, with explicit bounds (do not clamp an out-of-frame projection into the image).

Use the same unnormalized camera-z=1 ray convention as the parent: world ray direction `R_c2w @ [y_x/z, y_y/z, 1]` from the camera center. Cast against the **complete original mesh**, not only target faces. Require both measured depth D and first-hit `t_hit` to agree with z within `max(.02, .02*z)`, and require finite valid depth/hit. Thus direct geometric visibility V is distinct from the legacy representative-row hit count.

For weighting and query coverage use `O[g,v] = V[g,v] AND FULL_owner_v[round(project(x_g))]`. An atom geometrically visible but outside the FULL mask remains recorded as an alignment/boundary discrepancy, not positive semantic support. Store V and O separately. Unknown depth/occlusion/out-of-frame is unavailable, not a negative semantic observation. Batch rays with at most 65536 per call and four CPU threads. Full visibility makes no claim about the correctness of the owner's identity.

### B4. D-Surface diagnostic (four screening scenes only)
On the selected objects compare: (a) original individual source-row observation counts, (b) exact co-located site aggregation of those legacy hits, and (c) direct-site V/O counts. Report distributions of 0/1/2+ observed views, area-weighted repeat coverage, ambiguous support, and differences between physical visibility and FULL membership. Do not compare raw row counts and area estimates as the same quantity. Keep original camera/depth bounds; do not lower the two-view threshold to manufacture a positive diagnosis. This task does not relift saved SAM-V masks or change a geometry protocol.

## C. Common target and candidate plan
Use `pose_banks` unchanged: max 32 scene representatives, chronological input, .20 m translation and 15 degree rotation bins, deterministic retained indices. If a compatible parent bank exists, verify its frame/geometry/camera/depth/operator identities and reuse; the new site footprints still have their own identity. Read no panoptic GT; existing pre-insertion panoptic predictions are not needed for this protocol.

Candidate target inventory: original D2 owners with a valid frozen class, available p0, available historical F, no original support row with `raw==0`, and at least two legacy FULL masks with >=100 pixels and bbox width/height >=2. The raw-zero exclusion preserves the parent whole-object relabeling convention and must be reported explicitly; do not silently cancel a predicted edit later.

Rank these targets by `(D2 top1-top2 margin ascending, JS across available N/Q/F probabilities descending, owner ID ascending)` and select at most 16. JS is H(mean(p_e)) - mean(H(p_e)) using natural logs and the full original vocabulary. If only one source exists, JS=0. Ranking is a fixed heuristic, not a learned oracle. Record the complete prediction-side inventory and no-target reasons. Do not select recovered G1 owners or substitute targets based on model success.

After selection construct physical sites. Candidate views must satisfy the legacy FULL qualification and contain at least eight O-supported sampled sites with nonzero area. A selected object with fewer than two such views is a **KEEP_NO_TWO_PHYSICAL_VIEWS** entry for all methods, remains in the selected denominator, and is not replaced by another owner.

For each query-eligible object:
- Anchor a: qualified view of maximum original FULL pixel area; tie minimum original frame ID.
- Candidate bank: anchor plus up to seven views, farthest-point selection among **only qualified** views. Distance is translation/.20 + camera-rotation-angle/15. At each step maximize minimum distance to already chosen views; tie minimum original frame ID. Save the final bank chronologically, independent of acquisition order. All policies see the same bank (at most eight).
- Second view is distinct from anchor. Freeze all candidate metadata and supports before any new FC score is read.

The whole sequence is archived and available geometrically. This is **causal access to re-query evidence on an offline sequence**, not a causal online mapping or robot exploration claim.

## D. Four query policies: exact scores and fallbacks
Let A_g be sampled site area, O_gv its mask-supported visibility, and S_v its support set. All policies see geometry, candidate FULL areas and the same original distributions. The object center is the midpoint of the unchanged G1 support AABB.

Final ordering is always `(policy_score descending, FULL pixel area descending, original frame ID ascending)`; exact ties are not resolved with GT, future cosines, latency, or model outputs.

### AREA
`score(v) = FULL pixel count(v)` for v != anchor.

### COVERAGE
Implement the OVI paper's **object-centered surface-direction novelty proxy**, not camera-heading novelty. For each visible site, use unit direction `(x_g - object_center)/norm`; omit the degenerate zero direction. Theta=acos(z), phi=atan2(y,x) mod 2pi; bin into 180 x 240, clipped last theta bin. B_v is the set of occupied bins. `score(v)=|B_v minus B_a|/|B_v|`, zero if empty. This is a two-read adaptation of the published geometric criterion, **not** a reproduction of its online trigger, Native feature update or full system.

### VERIFY
`J_v = sum_g A_g O_ga O_gv / sum_g A_g O_ga`.
Let theta_v be the angle between the object-center-to-anchor-camera vector and object-center-to-v-camera vector. `h_v = min(theta_v / 30 degrees, 1)`; degenerate center/camera vector gives h=0 and a reason.
`score(v)=sqrt(J_v * h_v)`.
If all scores are zero, this naturally reduces to the stated area/ID tie-break, not an alternative search.

### DISAGREEMENT
Use only the anchor FULL and at most four anchor probe regions:
- c0 is the existing G1 class; c1 is p0's strongest other class, ties in original class order. Freeze the pair before acquiring the anchor.
- Split the anchor FULL bounding box into four half-open quadrants at integer midpoints; intersect with FULL. A probe is eligible at >=25 pixels and extent >=2. Use exact ordinary-FC v2 pooling on the original anchor RGB/dense tensor. No new masks, AnyUp, feature upsampling or text is used.
- `d_full = (s_a(c1)-s_a(c0))/T_F`. For each successful tile b, `d_b=(s_b(c1)-s_b(c0))/T_F` and `r_b=abs(d_b-d_full)`.
- Map a physical anchor-visible site to its pixel quadrant. `I_g = A_g * O_ga * r_b` if that tile succeeded; otherwise zero. This is localization of **relative response heterogeneity**, not a validated causal influence map. Report separately whether tile margins really include both signs; do not call a magnitude difference proof of class conflict.
- If <2 successful tiles or sum I <=1e-12, use VERIFY exactly with fallback reason. Otherwise `H_v=sum_g I_g O_gv / sum_g I_g` and `score(v)=sqrt(J_v*h_v) * (.5 + .5*H_v)`.

Thus selection asks for a different but jointly visible view of the anchor's heterogeneous region. It cannot predict the candidate's FC output by inspecting the candidate. Candidate image encoding cache existence must not influence selection. DISAGREEMENT may select exactly the same view as VERIFY/AREA: report divergence and fallback rates.

Do not relabel based only on the two competition classes. After selection, all methods use full-vocabulary output probabilities and may select a third class.

## E. Evidence acquisition and access semantics
Acquire shared anchor FULL records first; any required anchor tiles next. Seal second-view choices. Then send only the union of selected second FULL requests to the FC worker. Reuse an image's dense tensor across owners/regions when **image preprocessing tensor + physical model identity** matches. Text-independent dense data may be reused across vocabularies; a region-score cache requires the text identity and exact ordered IDs too.

Use inherited `image_tensor`, `signed_mask`, `FCSession`, and `area_fallback.region_vector` unmodified, in FP32 without autocast. Apply the parent visual head and normalized text cosines. A request key includes image, K/pose for lineage, original FULL/tile mask, pooling protocol, source support/geometry, model, text and ordered vocabulary. Do not call the hard-coded legacy worker end-to-end because its region cap and binder are for SAM-V.

Two counters are mandatory: actual union execution/cache counts; logical standalone requirements of each policy. Once another policy has acquired a candidate, that score remains inaccessible to an unrequested policy until the latter's selection is independently sealed. A compact whitelist of allowed keys per decision is sufficient; do not build a security subsystem.

Scientific unavailability: a finite zero/invalid region vector handled by the known original pooling failure categories is explicitly unavailable. If either required FULL is unavailable, that policy returns exact G1 for the object; do not substitute a different second view. A failed tile has zero priority and may trigger VERIFY fallback. Missing files, shape mismatch, CUDA failure and nonfinite unexplained output are execution failures requiring a real fix or blocked status, never free KEEP successes.

## F. Three update rules on exactly the same selected observations
Deduplicate exact observation content before aggregation using image pixel identity, mask, pose/K, physical support, model, pooling, text and ordered vocabulary; receipt IDs are not evidence. Copies that disagree on features or support under the same key raise a data error. This standard identity protection applies to **all** update controls; it is not an exclusive innovation of SUPPORT.

For each FULL observation `p_v = softmax(s_v / T_F)` over the complete original vocabulary. No class-subset normalization, temperature fitting, clipping, or per-scene weight tuning.

- MEAN: `p_new = mean_v p_v` over unique successful required FULL observations.
- AREA: `a_v = sum_g A_g O_gv`; `p_new = sum_v a_v p_v / sum_v a_v`.
- SUPPORT: `m_g = count_v O_gv` over distinct selected views; `w_v = sum_{g:O_gv} A_g/m_g`; `p_new = sum_v w_v p_v/sum_v w_v`.

If positive total support unexpectedly vanishes, fall back to MEAN and record it. If a required FULL failed, keep G1 instead (do not use this fallback to conceal missing evidence). With a unique duplicate observation, aggregate it only once.

For every updater: `p_final = .75 * p0 + .25 * p_new`, normalize numerical roundoff and take argmax in the original class order. There is no new class-margin gate. A no-query/unavailable/no-op path returns the exact existing class and p0 copy. Record probabilities, coefficients and actual label transitions separately.

SUPPORT conserves the total geometric evidence mass: sum_v w_v = sum_{g:m_g>0} A_g. It does **not** guarantee label correctness, AP monotonicity or invariance to arbitrary near-duplicate views. If two views cover the same sites, SUPPORT equals the mean; with disjoint supports, SUPPORT equals AREA. Record the prevalence of these degeneracies.

**Removed control:** two-observation pose-group equal averaging always equals the mean, whether the pair forms one group or two singleton groups. It is not a third useful scientific arm. Do not add a third read just to make this control different.

Only new observations have support-aware reweighting. Historical N/Q/F remains the immutable p0 prior. There is no claim that historical contamination can be subtracted or that historical/new overlap has been fully removed.

## G. Vocabulary and evidence-property diagnostics
For every parent incumbent with p0 and sufficient scores, fix (c0=G1/D2 label, c1=best other under the full D2 p0). For seeds 0,17,29 sort all other IDs by SHA256 of `seed|class_id`; retain c0,c1 and prefixes giving sizes 2,8,16,32,64,FULL, clipped/deduplicated to the existing vocabulary. Keep original class order within each subset. No new strings, synonyms, text encodings or GT-selected competitors.

Compute (1) original grouped per-source-softmax fusion and (2) group-equal source scores divided by their frozen temperatures, averaged before any final softmax. Preserve source absence and original groups. For the original pair record margins, strict sign changes and ties (1e-12); a newly introduced third class may legitimately win. Single-source pair ordering and scaled-score pair difference are invariants under adding other classes; current probability mixing is not. Validate the full-set current output against the saved D2. Do not compute reduced-vocabulary benchmark AP or promote the diagnostic scaled-score rule to a new main method.

Properties for a fixed observation set: exact content duplicate leaves the deduplicated result unchanged; permutation changes at most declared float roundoff; withdrawing one **new** record equals batch recomputation on the remaining new records; no remaining record yields exact p0. Adaptive choices are not required to be permutation invariant. All predictors are GT-free; GT analysis follows locks only.

## H. Output, evaluation and effect separation
Build a new `PredictionPayload` using the current G1 geometry and owner arrays. Populate the whole output class registry, including recovered owners, even when no native feature exists. Relabel only eligible selected original incumbents; enforce homogeneous owner class. No old raw-zero veto is silently removed. Recompute `native_ranks`, verify against `official_view`, then lock/save.

Use an evaluator built for the actual G1 partition and current scorer context. Reuse only with equality of actual owner arrays, labels, current six-decimal ranks, projection, vocabulary, evaluator and targets. Method names or identical rounded metrics are insufficient. Re-pool any changed ordered scene set.

Since support is fixed, class-agnostic matching is a negative control, not an improvement metric. Main mechanisms concern class-aware matches and correct/incorrect semantic changes. Match ties are explicitly recorded; do not equate score-entry counts with unique GT objects.

## I. New evidence manifest (minimum fields)
Per scene: pinned context/baselines, map/camera/site identities, full candidate frame list, target inventory with exclusions, query-eligible IDs, selected support/anchor/candidates, no-GT declaration.
Per observation: content key, FULL/probe role, owner, frame, original RGB/depth/mask identity, model/text/pooling IDs, feature/cosine record, O-supported sites/weights, availability and measured cost.
Per choice: policy, allowed evidence keys, c0/c1, anchor tile contrasts, geometric candidate scores, selected second frame, fallback reason, before-acquisition choice identity.
Per update: selected view content keys, dedup decisions, MEAN/AREA/SUPPORT weights, p0/p_new/p_final, proposed/applied label, unchanged/recovered-support audit.
Per stage: actual/expected logical coverage, unique physical work, complete pools, frozen selected controls, gate results and conditional reasons.

Store compact score vectors once with lossless references and a verified restoration procedure; don't bloat GitHub with per-element assertions or repeated copies of the same feature array.
