# Algorithm and integration contracts

These are **new implementation requirements**, not statements that these APIs already exist. Constants are normative in `PROTOCOL_SPEC.json`. Where an inherited value is specified, resolve it once from the pinned parent and record its exact value before measuring variants. Never silently substitute an arbitrary value if that inheritance fails.

## C1. Namespaces and job graph

Keep separate identifiers for:

1. current-frame 2D candidate/group ID;
2. current depth-fragment ID;
3. native global superpoint label;
4. native object owner returned by mode4 count-based membership;
5. segment-graph component ID, if exported for diagnosis;
6. final owner on the native official-export surface.

A SAM track ID does not equal a native object ID. A graph component ID does not automatically equal mode4 raycast membership. Every cross-level conversion must be explicit in a record.

A job identity includes parent input identities, method recipe, actual code/patch/binary identity, scene schedule, camera/pose/depth/panoptic identities, and dependent semantic/evaluation identities as appropriate. Model request content identity is separate from lineage request identity. Store records under `<root>/<phase>/<scene>/<map_id>/...`, and include map ID in shared-mask/projection/evaluation roots. Never share `shared_masks/<scene>` across unlike native geometries.

Suggested minimal products:

- `resolved_inputs.json`: actual roots, modes, assets, schedules and inherited recipe values.
- `native_build_receipt.json`: upstream/base patch/new patch, compiler/dependency identities and actual loaded `.so`.
- `maps/<scene>/<map_id>/map_receipt.json`: map inputs, outputs, counts, duration and status.
- frame captures: preinsert fragments, prior probe summary, decisions, integrated membership, current owner projection and requests.
- `geometry.json`, `surface_raw.npz`, native painted surface, regenerated `projection.json/.npz`.
- `sources/{N,Q,F}.json`, request/features manifests, immutable readout predictions.
- official scene rows and dataset pools, separate diagnostic ledgers.

Store complete arrays externally; compact decision summaries and failure records are sufficient in Git. All predicted artifacts must be locked before GT labels are used for diagnosis/evaluation. Evaluation projection to target coordinates is not an input to mapping or view selection.

## C2. Simultaneous depth/instance fusion (BB01_SYNC)

Inputs are aligned integer depth-region raster `D`, original 2D instance raster `P`, metric depth, intrinsics and pose. Do not change which depth-region generator was used. For each original nonzero depth region d:

1. Let `S = (D == d)` and `A = count(S)`. Apply only the native initial gate `A < 100 -> skip`.
2. For each positive instance k with nonempty intersection, compute on the **unchanged original** S:
   `I_k = S & (P == k)`, `a_k = count(I_k)`, `b_k = count(P == k)`.
3. Qualify k exactly when `a_k > .9*b_k` and `a_k < .5*A`. All quantities in this decision refer to the original region. P is an exclusive label raster, so these intersections are disjoint.
4. Emit every qualified nonempty intersection as a foreground fragment with its original current-frame group, native score1.0, and `overlap_ratio = a_k/b_k`. Do **not** additionally delete a carved fragment merely because it has fewer than100 pixels; that would be a different experiment.
5. Set `R = S \ union(qualified I_k)` once. If empty, emit no residual and perform no division by zero.
6. Recount intersections on R, excluding label0 for foreground selection. Select the positive label with greatest residual intersection. Resolve ties by spatial support order, not by numeric ID. If the maximum is at least `.2*count(R)`, emit the whole R with that group, score1.0 and residual overlap fraction. Otherwise emit background with score.5 and its measured background fraction.
7. Retain the native separation of foreground and background lists. Use a documented canonical spatial ordering within lists: first occupied row-major pixel, then full mask-support digest. Compute the usual3D points/boxes through the native conversion.

Keep `.9`, `.5`, `.2`, native semantic foreground/background placeholders and all downstream settings fixed. No border erosion, depth completion, tiny-object compensation or new classifier is allowed in this arm.

### Order diagnostic

For the first three valid scheduled frames per development scene, compare original/native visit order with reversed 2D-candidate visitation and a bijective renaming of instance IDs. Preserve raw input masks. Compare canonical `(fragment support, physical source-group support, is_thing, score, overlap_ratio)` collections. Numeric fragment IDs, owner colors and insertion indices are not sufficient evidence of a changed partition.

Also distinguish actual support/grouping differences from ordering-only differences. New object numbers that change a semantic target-cap tie break are not by themselves evidence of stronger geometry.

## C3. BB05_RATIO_GATE

In the native `getNextSegmentLabelPairWithConfidence`, replace only the accepting condition with the conjunction of its existing conditions **and** `ratio_greater_than_min`. The ratio's numeric threshold remains `label_tsdf_config_.label_register_min_overlap_ratio` from the bound native configuration. Record it. Fresh-label handling, assigned-label exclusion, subsequent candidate recomputation, object counts, graph confidence and alias merging remain native.

Do not enable this change in FORWARD/BIDIR. Those arms use their own specified directional eligibility; mixing the unused native ratio into them would add an uncontrolled fourth constraint.

## C4. Pre-insertion geometry probe for directional association

Create a new read-only Python-facing native function, for example `exportAssociationProbe(pose_c2w, depth_m)`. This function name is proposed. It must operate on the map at t−1, before any current-frame fragment is inserted or any current-frame count/graph update occurs.

Probe every pixel with finite depth `0 < z < 50m`, not only pixels labeled foreground by current CropFormer. Use the native camera ray generator and bounded depth neighborhood `[z-.2,z+.2]`, clipped to its range. Walk from near to far and return the first voxel meeting all conditions:

- a nonzero superpoint label with label confidence at least.5;
- an allocated TSDF voxel with positive measurement weight and `abs(tsdf.distance) <= 2*voxel_size`;
- the voxel-center camera-space depth differs from the current sensor depth by at most `tau(z) = .03 + .01*z` meters.

Look up that superpoint's prior owner using mode4 `getInstanceLabel(label, 0.0f, empty_assigned_set)`. Return `prior_label`, `prior_owner`, `hit_depth`, and support validity. Owner0 is not a foreground object. Unobserved, out-of-range, occluded or depth-inconsistent pixels return no foreground support. Do not allocate TSDF blocks, adjust counts, consume fresh IDs, or update the graph during this query.

This is a **depth-conditioned prior-label support probe**, not a claim of exact visibility or the original OnlinePG renderer. Its extra surface/depth tests are shared by FORWARD/BIDIR and must be counted in cost. Do not call the mutating `raycastPanopticPredictions` routine. The old read-only `raycastInstancePredictions` is a source of compatible traversal/lookup logic, but it gates on the supplied mask and does not return all needed arrays; calling it after current insertion is not acceptable.

Snapshot known superpoint→prior owner membership and alias state before insertion. During a real short trace check that probing leaves allocated blocks, label/count statistics and fresh-ID counters unchanged. Only touched-state checks are needed; no whole-scene hash every frame.

## C5. Directional object assignment

Group retained current foreground fragments by their nonzero input 2D instance ID. A local object L_i is the union of those fragment supports on the current image. Let V be valid current-depth pixels; let G_j be the positive prior-owner pixels from C4.

Define:

`a_i = count(L_i & V)`
`b_j = count(G_j)`
`I_ij = count(L_i & G_j & V)`
`F_ij = I_ij/a_i`
`R_ji = I_ij/b_j`.

The reverse denominator is **the full current depth-consistent probe support of that prior owner**, not just its intersection with local foreground proposals. No dataset category or future semantic vector is used.

For forward matching, an edge is eligible when `I_ij >= 100` and `F_ij > .2`. Benefit is `F_ij-.2`. For backward matching use `I_ij >=100` and `R_ji >.2`, with benefit `R_ji-.2`. Zero-size objects produce no edges.

Solve a maximum-total-benefit one-to-one assignment with an independent zero-benefit dummy destination for each row, so every local/global object may remain unmatched. Give forbidden edges a sentinel below any valid solution; filter all nonpositive edges from the result. Use double precision, sorted stable local-support keys/global IDs, and a pinned deterministic solver. Identical inputs must produce identical assignments. Do not inject an epsilon large enough to change a genuinely non-tied optimum.

- FORWARD accepts its forward pairs.
- BIDIR independently solves R, converts pairs back to `(local,global)`, and accepts their intersection with the forward result.
- Every unmatched local object obtains exactly one native fresh **object** ID, shared by its fragments. Existing global objects not matched this frame remain in the map. They are not removed or automatically merged.

A successful unit fixture must show that F and R can produce different assignments under valid intersection/area constraints. Running Hungarian on a single symmetric IoU matrix and its transpose fails this requirement.

## C6. Enforce object decisions in the real mode4 mapper

The Python/C++ boundary must carry a frame-local local-group→planned-owner table and the prior membership snapshot. Native methods may allocate fresh object IDs; Python must not guess or recycle them.

Implement the following common behavior for FORWARD and BIDIR:

1. Freeze prior membership before current insertion and compute planned groups from C5. New IDs allocated for unmatched groups are recorded without deleting existing owners.
2. Preserve the usual distinct superpoint labels. A matched local group may reuse an old superpoint only if its snapshot owner equals the planned old object. Unmatched groups may not inherit a superpoint already in the snapshot; they obtain fresh labels through native allocation. Background fragment association remains native.
3. Fresh superpoint labels allocated in the current frame are tagged with their planned local-object token. They are valid for their originating group, not reusable as a backdoor to a different planned group. Apply this filter in every invocation of `computeSegmentLabelCandidatesConfidence`, including calls made by greedy conflict recomputation. Do not filter only a first candidate list.
4. In mode4's `updateLabelClassInstanceConfidence`, use the planned object per local group instead of letting the first visited fragment independently choose it again. Increment ordinary label/class/object counts once as native code does. Do not reset old counts or hard-code final labels. Confirm a reused superpoint's prior dominant owner is compatible with the plan; log the rare actual post-update discrepancy rather than suppressing it.
5. Keep `IncreaseSegGraphConfidence`'s within-current-group behavior once per frame. Both branches use the same `seg_graph_confidence=3`; no new graph confidence formula in this wave.
6. Filter newly gathered pairwise superpoint merge candidates by compatible planned-owner tags. At merge execution, block a queued merge if the endpoint tags conflict, using current-frame planned tags for touched labels and the prior owner snapshot for untouched known labels. Resolve aliases transitively. Two distinct positive object tokens are incompatible; a fresh token is distinct from all old positive owners. A0/background endpoint alone supplies no positive object agreement; allow its original merge only when it cannot bridge two conflicting positive tags. Aggregate token sets across aliases so a chain through background cannot bypass the check. Do not purge unrelated old compatible pairs.
7. After frame integration, export assigned superpoints, count-based object membership, alias events and current raycast ownership. Do not claim that a planned association was realized merely because the planner emitted it. A stale Python dictionary/color remapping with unchanged actual map is not completion.

All C6 machinery is disabled in BB00, BB01, RATIO_GATE and standalone SAM arms. A combo uses exactly the selected structural machinery. The scope includes changes required for the planned constraint to survive downstream native code, not an alternative complete mapper.

Record at least: prior state ID, local group and fragment IDs, candidate intersections/coverages, forward pairs, backward pairs, accepted pairs, native allocated object IDs, candidate veto counts, alias veto counts, assigned superpoints, realized count-owner mapping, and changed canonical map supports. Store sparse tables only; do not dump full TSDF every frame.

## C7. Bounded SAM2 front-end assay

### Fixed inputs and memory

Use exactly the valid scheduled frame order. Partition it into consecutive chunks of five valid frames (last chunk may be shorter). This reset is a deliberate bounded temporal assay; it is **not** the B06 sliding-window multi-view instance-fusion method. Mapping frames are not delayed or revisited.

Run the pinned SAM2 video predictor with the original acquired checkpoint/config and evaluation mode. Derive its Python executable from the historical SAM environment receipt; do not upgrade the native mapping environment. If the official loader requires numbered JPEGs, create deterministic derived JPEGs with quality100 and no chroma subsampling from the aligned RGB arrays, record the conversion, and use the same derived bytes for RAW/GEOM. Original reconstruction RGB, depths and poses remain unchanged.

Initialize one predictor state per chunk with CPU video/state offloading as needed. Seed every nonzero first-frame CropFormer mask with `add_new_mask`, ordered by row-major support key. First-frame final raster is exactly the original CropFormer raster. Predictor warm-up counts as physical image inference. An empty-seed chunk copies CropFormer for the entire chunk and records NO_SEED; it does not crash or call an imaginary segment-everything API.

For each subsequent current index t, consume only the forward output for t. At the pinned API, `start_frame_idx=t, max_frame_num_to_track=0, reverse=False` yields only t because the end is inclusive. Do not pass1 expecting one frame, do not seed all future CropFormer masks at initialization, and do not use reverse propagation. Reset after each chunk. Never accumulate scene-long object memory or compute extra unscheduled video frames.

### RAW raster and new objects

For each seeded track, keep its raw video-resolution logit mask `M_i(t) = logit_i(t)>0`. Resolve track overlap by highest positive per-pixel logit; tie by stable seed-support key. Retain the pre-resolution masks and logits for diagnosis/GEOM filtering.

Let U_raw be the union of all positive raw track masks. For each positive current CropFormer mask C_k, define coverage `count(C_k & U_raw)/count(C_k)`. If coverage is strictly below.2, register it as a current-frame discovery candidate. This candidate set is computed once from RAW and shared with GEOM. It is not prompted into SAM in this bounded assay; it can become a seed at the next chunk's first frame. Thus discoveries still enter the current map without future information.

RAW output first assigns positive SAM pixels by the rule above, then fills unassigned pixels from discovery candidates in spatial key order. Do not run an extra semantic detector. Remap resulting positive IDs densely into a safe integer raster; mapping consumes them only as current-frame grouping IDs. Keep the original masks, track IDs and remapping table so changing numbers cannot be mistaken for a new physical object.

### GEOM filter — no extra neural inference

For t>chunk start, for each original seeded track, backproject its **previous RAW** mask using previous valid depth; transform by poses to the current camera; project and round to the nearest pixel. At collisions choose the smallest positive transformed depth. Keep only projections inside the image whose current observed depth agrees within `.03+.01*z_current` meters. This yields visible past support W_i. Invalid depth, occlusion and out-of-view projections are not negative evidence.

If `count(W_i)<100`, abstain and keep the current raw track. Otherwise compute retention `count(W_i & M_i(t))/count(W_i)`. Accept when retention is at least.5. This is a directed support-retention test, **not** IoU against a full amodal object.

For a rejected track, compare the current raw mask M_i(t) to each original current CropFormer mask. If the best IoU is at least.3, provide that CropFormer mask as its replacement; otherwise provide no replacement. For duplicate replacement requests use the mask once. Acceptance of one track must not rewrite another track's raw prediction or the next SAM memory state.

GEOM raster first assigns accepted SAM tracks by original raw logits, then replacement CropFormer masks on unassigned pixels, then the **same RAW-defined discovery candidates** on remaining unassigned pixels. Order replacement masks by descending matching IoU then spatial key. This prevents a changed discovery policy from being hidden inside the geometry test. Its fallback can still be wrong; report accepted/rejected/replaced/missing counts and actual map effects.

RAW and GEOM use identical model frames, seed prompts, chunk boundaries, raw logits and preprocessing. Do not feed the GEOM outputs into later SAM state or compare against a future frame. Only this filtering/resolution differs. On mapping input, use original native foreground/background confidence conventions; SAM score weighting is deferred.

## C8. New maps, anchors and caches

A new geometry version contains actual xyz/faces/TSDF plus raw numeric superpoint/owner arrays and official native-painted owner arrays. Recreate the original mapping-to-native-semantic eligibility on **that map**. Preserve the first-vertex triangle painting/order behavior of `native_surface_readout` and record the gap to raw numeric ownership separately. Do not accidentally perform semantic rectification under the name of geometric improvement.

The old `PredictionPayload` permits branch tags N0/S/G/Q/COMBO. Use a valid tag in a task-local adapter or a new task-local payload; do not pass an unregistered BACKBONE tag to old code and then delete its validation. A new map's anchor is the native reference for all its downstream readouts. `relabel_prediction` is permissible only **within that exact anchor**, never from historical G0 into a new map.

Cache rules:

| Artifact | Reuse rule |
|---|---|
| RGB/depth/pose/intrinsics | Same verified scene/frame bytes and convention |
| Original CropFormer / geometric rasters | Same original input and preprocessing; immutable |
| SAM raw predictions | Same ordered chunk RGB, checkpoint, config and seed-mask identities |
| FC whole-frame dense encoding | Only if actually persisted and model/image/preprocessing/precision match |
| Pooled region or six-crop feature | Exact image, target/union mask, box, crop convention, model and preprocessing match |
| N/Q/FC aggregate | Same actual used requests, weights, ordering, model/text and success states |
| GT projection indices | Same actual source and target coordinates/order, projection algorithm and threshold |
| Evaluation result | Same geometry/owners/classes/ranks, scene list, evaluator/GT identities |

An equal owner number, same model name, matching dimensions or an approximate overlap is insufficient for feature reuse. Exact physical feature reuse under a different lineage ID requires an explicit alias record linking both requests to the same physical content key. Logical acquisition is still charged. Source artifacts must identify themselves as real N/Q/FC evidence, not populate a missing source with the incumbent's probability.

FC's area-policy cap and native's at-least-two-observation admission can hide newly generated instances from final official export. Report raw owners, native-eligible owners, N/Q/F available owners and cap exclusions per map. These are part of pipeline interpretation, not justification to silently change eligibility.

## C9. Evaluation contracts

Official target-space projection is Open3D float32 nearest1 with strict distance `<.05m`. It must be computed for each changed source coordinate/order. The instance confidence used by the official current-class view is area/max-area-within-current-predicted-class, serialized to six decimals, with the actual exporter minimum-size rule. Do not replace it by neural probability or change min-region size to favor small candidates.

Dataset pools call the original released evaluator over all ordered scene manifests. Semantic mIoU/mAcc use the sum of per-scene confusion matrices. Read `namespace['overlaps']` and publish it. The established APall range is .50–.90; AP25 remains separate.

Class-agnostic diagnosis uses predicted raw numeric instance masks projected by the same coordinate procedure, with all eligible GT instance IDs from the same ignore/min-size rules. For each threshold, maximize the number of one-to-one matches in the IoU eligibility graph. Recall is matched eligible GT/ALL eligible GT, not matched-only GT. The comparison is strict `IoU > threshold` consistently with the released matching convention. Report thresholds .25/.5/.75 separately; do not call these the official semantic AP.

Per-GT best IoU includes zero for a missed target. Report the number of distinct prediction masks each GT substantially intersects and the purity/coverage of those intersections rather than only object count. Surface precision and completeness use the full reconstructed observed surface versus valid target surface at strict5cm; sampling/density conventions must be identical across variants, disclosed, and not fitted per scene.

Across map IDs, numeric owner IDs are unrelated. Report GT-indexed match gains/losses, predicted-geometry overlap associations (without GT in prediction), and changed point partitions. A new semantic label applied to the same unchanged shape is not a geometric gain. A different class-conditioned rank can change AP without changing geometry; secondary per-map-native ranks expose this.

Never pool only the successful scenes of an incomplete method under the full cohort's title. Publish complete-case descriptive subsets explicitly if useful, but incomplete methods are not eligible for full-cohort selection.

## C10. Selection, composition, completion

Every constant and candidate identity is frozen before development metrics. Select structural and front-end candidates only from fully measured respective new arms. The ranking bands and tie order are in the spec. Method differences are evaluated on the four-scene official D2 pool. Publish the strict metric ranking and the banded preference ranking.

`candidate_freeze.json` must contain both candidates and their actual config hashes before launching BBX_COMPOSE. Compose by using the selected front-end raster *then* the selected structural depth/association setting. If structural=BB01, do not also enable B05. If structural=RATIO_GATE, do not also enable directional matching. If structural=FORWARD/BIDIR, keep the native depth-fusion procedure. This is exactly one front-end block plus one structural block.

Do not require either candidate to have already beaten G0 before attempting the single composition. Require complete inputs and a nontrivial canonical map intervention for each. If all new arms in a family are final-map identities, retain their results and skip the composition with the no-intervention reason. A technical missing family is a different reason.

`selection.json` freezes the nominee among G0/two candidates/composition and the at-most-four transfer configurations before Replica evaluation. Imported prior scores are never a substitute for a new map whose representation changed. An unchanged G0 job may reuse historical artifacts only after the bridge verifies identical input/output identities and official behavior; distinguish imported from newly computed jobs.

For scientific interpretation, label an improvement in both APall and mIoU with at least one increase beyond.05pp as a measured net gain; a mixed-sign result is a tradeoff, and all nonpositive changes are no net gain. These labels describe measurements, not significance or deployment approval. A stronger-backbone mechanism additionally needs consistent raw-geometry evidence and relevant simple-control comparisons. Keep implementation completeness, data coverage and publication status separate.

## C11. Publication receipt and no excessive auditing

The publication receipt is external because it contains the final commit SHA. Only after an actual normal push and equality of the complete local/remote SHA may it contain PUSH_VERIFIED. Do not commit it and recursively change the hash it is supposed to attest.

A release manifest checks the files actually included once. Large consumed assets are hashed once per relevant job/process and cached with size/mtime/inode checks. Unrelated old artifacts need no scan. Preserve failed job logs and costs; do not repeatedly reproduce an already demonstrated failure solely for more audit output.

Real short native/GPU bridge tests are necessary; repeated generic safety audits are not. The provided reference fixture tests cannot certify the C++ mapper, GPU execution, cache availability or scientific performance.


## C12. Cost accounting under shared computation

Maintain two different ledgers: (a) actual physical work executed in this attempt, including failures/retries and loaded models; (b) standalone work required by each method recipe. RAW/GEOM both require the raw SAM predictions, even when only RAW's worker physically computed them. Every D2/FC_EQ readout requires its real N/Q/FC stages even when a preceding readout populated the cache. Use content-deduplicated required image encodings as the first cost tie breaker, not a zero-inference cache receipt.

For the timing tie breaker, attribute measured common prerequisite stages to each standalone method that needs them, add its own map/readout stages, and take the per-scene median. This is an attributable standalone estimate, not directly measured wall time of a clean isolated end-to-end rerun. Publish real execution wall time, stage timings, physical forwards, standalone obligations and missing timing components separately. Do not run duplicate expensive cold jobs solely to refine a tie-break timing estimate.

The composition no-intervention check compares canonical **raw numeric instance partitions and geometry**, not arbitrary IDs, surface colors, semantic labels or class-conditioned ranking. A change solely caused by target-cap tie ordering is logged separately and cannot establish a stronger-backbone mechanism.


## C13. Avoid an unnecessary raw-logit storage workload

Compute RAW and GEOM raster outputs from the same current raw prediction in one paired preprocessing pass. Raw floating-point logits may remain in a bounded chunk buffer; persist final exclusive rasters, bitpacked per-track masks, filter measurements and the raw neural-job identity. Do not dump full per-object float32 logits for every frame of every scene by default. Keep full logits only for a small explicitly identified diagnostic need. A reusable frontend bundle must contain both paired outputs and their shared input/code identity. If a later repair genuinely needs an unpersisted intermediate, account for the necessary recomputation rather than inventing it from a binary mask.
