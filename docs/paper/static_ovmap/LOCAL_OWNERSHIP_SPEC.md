# Codex execution task: preserve U00 candidate gains through local instance ownership

**Version:** 2026-09-14 / implementation specification, not an experimental result  
**Repository:** `Orangekostar/oviovo`  
**Reviewed source branch:** `research/ovimap-t1-attribution-v1`  
**Reviewed commit:** `98cafc4a96bf906878116ae3284da28bc974ad4d`  
**New task branch:** `research/ovimap-local-ownership-v1`

## 0. Execute this task, including the GitHub handoff

Implement and evaluate a bounded local-ownership method on the two existing Room0 FP32 prediction caches. Start from the **U00 candidate pool**, not the already relabeled and suppressed U11 pool. Preserve its masks, original classes, canonical identities, and AP-ranking scores. Change how overlapping candidates become a **unique instance map**.

Complete development, six new run/condition evaluations, a compact scientific interpretation, and a Markdown handoff. **The user explicitly requests committing and pushing the scoped code, configuration, small results, and handoff to GitHub. Do that on the task branch and verify the remote SHA.** A local-only handoff is not full delivery unless a real push/access failure is reported.

This is a new development authorization. The previous attribution task's `NO_REPAIR_JUSTIFIED` closed that bounded diagnostic exercise; it does not forbid the new implementation specified here. Conversely, do not pretend that the present method has already produced a gain.

Use an isolated worktree or continue an existing matching task branch after reading its changes. Protect unrelated work. Do not reset or force-push other branches. Record actual HEAD and relevant differences from the reviewed base. Do not reopen the entire CROVE project or repeat the old 34-condition study.

**Fixed scope:** one development scene, two distinct saved FP32 network runs, cached geometry and predictions, plus CPU processing of existing depth/pose/2D-instance observations. No new mapping, network inference, training, backbone, SAM run, image encoding, ScanNet download, or full-benchmark expansion. Source-side reprojection and evidence computation are allowed and must be timed. Zero inference does not mean zero compute.

## 1. Evidence motivating this task

Read `docs/paper/static_ovmap/T1_ATTRIBUTION_RESULTS.md`, its audit, and full-precision rows under `artifacts/static_ovmap/t1_attribution_v1/`. The following rounded percentages are historical anchors, **not new acceptance numbers**. [S1]

| Existing condition | Primary candidate AP | Repeat candidate AP | Primary unique AP | Repeat unique AP |
|---|---:|---:|---:|---:|
| Common-domain OVI/S1a-only (`AT_O_AREA`) | 20.644 | 20.644 | 20.644 | 20.644 |
| U00: candidate union, no borrowing, no fusion NMS | 23.153 | 22.081 | 6.005 | 4.067 |
| U11: old complete T1 | 20.516 | 19.810 | 12.399 | 8.352 |
| Old OVI_FILL **on U11** | 20.516 | 19.810 | 20.644 | 20.201 |

U00 added 7/6 released matched GT objects at the threshold near 0.5, but only two of those gains survived its naive unique ownership in each run. The study also observed extensive same-class owner changes invisible to semantic-class accuracy. These observations motivate local ownership, not a claim that any local smoother will work. [S1]

The earlier historical-S1a-to-T1 AP increase included geometry transfer/export and score changes. Preserve those stages and the same-domain O-only comparison; do not credit their improvement again. S1a itself is a fixed input here, not a newly proposed contribution. These experiments concern **our static fusion/output layer**, not CROVE temporal ownership or OVI-MAP's original mapping backend.

### Explicit hypotheses

- **H1:** shared, position-dependent multi-view instance evidence can assign conflicting areas better than a candidate-wide scalar or unconditional OVI protection.
- **H2:** spatial consistency can improve unique-instance quality beyond the **same** local evidence, without changing the candidate set or semantic labels.
- **H3:** the recovered unique-map gains survive both cached runs without creating a larger loss of previously correct OVI instances.

All three are hypotheses to test. Existing 2D masks are predictions, often related to OVI's own frontend; they are not an independent ground-truth teacher. Geometrically identical candidates with different classes may remain indistinguishable to this geometry/instance-only evidence. Do not hide that limitation with GT-selected decisions.

## 2. Read and bind the actual code before implementation

Read these files at local HEAD and compare directly relevant behavior against the reviewed commit. Source links appear in Section 14. New file names later in this prompt are proposed, not existing APIs.

| Existing source | Verified behavior | Reuse / required change |
|---|---|---|
| `src/static_ovmap/fusion_attribution.py` | Frozen masks/areas and canonical candidate identities; separate borrowing/NMS; `resolve_native_owners` takes one scalar per candidate and greedily claims unassigned points. [S2] | Reuse frozen pool and exact U00 identity. Preserve legacy resolver; implement a separate local family. |
| `src/static_ovmap/proposal_fusion.py` | Legacy fusion wrapper. [S3] | Preserve compatibility; do not silently change old T1 defaults. |
| `scripts/evaluation/run_static_t1_attribution.py` | Condition-specific global priorities; owner cache key uses kept IDs, priority array, and source groups. [S4] | Add a bounded new runner/explicit method registry. Include local-evidence/atom/graph identities in cache keys. Unknown method IDs must fail, not fall through to a raw rule. |
| `scripts/evaluation/diagnose_static_t1_attribution.py` | Reuses projection, exports overlapping/disjoint masks, calls released evaluator, and **asserts equality with globally ordered projected-mask assignment**. Closing checks reference old `AT_*` conditions. [S5] | Add family-aware validation and method-aware closing checks, or a small adapter reusing these functions. The present assertion would reject a genuine local map. |
| `src/static_ovmap/geometry_support.py` | Integer-pixel rays, measured-depth agreement, pose-distinct support selection. [S6] | Reuse coordinate/visibility conventions and pose-distance logic. Depth agreement establishes visibility, not correct ownership. |
| `src/static_ovmap/instance_graph.py` | Frame-local mask/owner contingency statistics and multi-view evidence aggregation. [S7] | Reuse counting ideas, not its old positive/negative graph decisions as object truth. Extend to candidate/atom evidence. |
| `src/static_ovmap/split_evidence.py` | Repeated-view evidence for splitting one native parent. [S8] | Reference correspondence/unknown semantics; do not import old split output or impose parent ownership on the new solution. |
| `src/static_ovmap/hierarchy.py` | `component_readout` pools selected semantic observations of component members. [S9] | **Not a neutral spatial-atom builder.** Do not assume a `make_surface_atoms` API exists here. Do not change semantic readout in this task. |
| `scripts/evaluation/run_static_spaceformer.py` | Saves postprocessed T0, original query IDs, raw logits, query embeddings and text. [S10] | Bind T0. No new network call. Raw logits are optional future evidence, not required for the six primary cells below. |
| `src/static_ovmap/attribution_regions.py`, `attribution_objects.py`, `released_trace.py` | Region accounting, object evidence, observation-only tracing of the original evaluator. [S11] | Reuse diagnosis. New active-conflict checks must use **U00**, not hard-coded U11 multiplicity. |
| `scripts/evaluation/evaluate_static_ovmap_readout.py` and receipt-bound released evaluator | Historical semantic and instance metric behavior. [S12] | Keep vocabulary, valid-GT policy, scores, thresholds and AP behavior unchanged. |

**Do not confuse a proposed path with a verified producer.** Locate actual frame loaders and geometry-support producers using their recorded argv and a targeted local search, such as `rg -n 'native_pixel_rays|split_from_views|geometry_support.json' scripts/evaluation src/static_ovmap`. Read the resulting relevant callers. Do not invent a `build_native_geometry_support.py` command or assume an unverified source-ledger filename. A targeted lookup is sufficient; do not scan unrelated storage repeatedly.

## 3. Bind only the assets this task consumes

Start from `configs/evaluation/ovimap_t1_attribution_v1.json`, prior input inventories, `source_handoff.json`, and the original model receipts. Historical locations are hints, not proof that files are still readable:

```text
/home/ww/oviovo_baseline_runs/20260913_static_ovmap/room0/
/mnt/shared/ww/ovimap-t1-attribution-v1/
```

Required assets:

1. Each independent FP32 run's frozen pool or bound T0 and exact native/S1a source used to reconstruct that pool. Preserve OVI readout identity and same source generation.
2. Returned `T0.npz['coord']`, canonical mask bank, original classes, query IDs and source areas. Existing `predictions`/`predictions_verified` are two postprocessing versions, **not** two extra model runs; use the verified numeric equivalence and correct metadata.
3. Original native-owner projection for geometry provenance; bound common-to-evaluation projection, original protocol, and full-precision `AT_U00`/`AT_O_AREA` results.
4. Existing measured depth images, aligned frame-local predicted instance-mask PNGs, actual poses, intrinsics, original sampled frame IDs, and loader conventions. Resolve them from old geometry-support / graph / input receipts. Do not use final G1/G2 owners as original OVI owners.
5. GT meshes/conversion only for the evaluator process, never for selecting frames, computing support, graph construction, or prediction.

Check consumed-file identity once, shapes, coordinate order, units and alignments. Do not hash every dataset/model again. Reuse projections across runs only after exact coordinate/order equality. Do not use the input RGB-D cloud's row order as the returned T0 row order.

Existing raw model assets can be recorded as available/not checked without loading them. `raw_mask_logits.npy` is roughly 1.49 GB per run and rows correspond to original queries; `T0.query_ids`, not sorted T0 row numbers, map back to it. Do not load it merely because it exists. Do not substitute SF sigmoid values versus OVI binary values as a purported calibrated cross-model local score. [S10]

If a required observation source is unavailable, complete U00_FILL and unaffected engineering/evaluation work; mark dependent local cells `BLOCKED` with exact missing inputs. Do not generate fake evidence, reopen unrelated provenance, or silently launch inference. This chat has inspected repository text and recorded asset locations, not mounted the server's large files.

## 4. The frozen experiment matrix

Implement exactly these primary additions, each on both existing FP32 runs: **3 × 2 = 6 new method/run cells**.

| ID | Prediction-only pool / classes / candidate rank | Ownership |
|---|---|---|
| `LO_U00_OVI_FILL` | Exact U00 | Qualified OVI first, then SF fills uncovered points; raw area and canonical index within a source. |
| `LO_U00_LOCAL` | Exact U00 | Local shared-view evidence on neutral small atoms; no pairwise spatial term. |
| `LO_U00_SPATIAL` | Exact U00 | **Identical atoms, evidence, unaries, feasible labels and fallback to LOCAL**; add only the fixed spatial term. |

U00 keeps all nonempty candidates already exported by T0 and all qualified OVI candidates, with borrowing OFF and **additional fusion NMS OFF**. T0's original upstream postprocessing stays fixed. Do not recover additional raw model queries, relabel candidates, resize their masks, add class-aware NMS, merge candidates, or recompute AP confidence after assigning points.

The old OVI_FILL used U11; it is not a valid substitute for `LO_U00_OVI_FILL`. Reuse old U00 and O-only results only after matching source domain, registry, masks, class IDs and serialized ranks. List reused baseline rows separately, not as new experiments. Do not rerun all old 34 conditions to populate a familiar report template.

All new overlapping-candidate manifests and masks must be equivalent to U00. Candidate AP equality is an **intervention isolation invariant**, not a claimed new gain. Evaluate the new disjoint maps separately.

## 5. Implement a concrete first local method

The following algorithm and numerical settings are **proposed engineering choices, not validated results or optimal thresholds**. Implement one configuration, write it before new GT evaluation, and do not sweep a grid. CPU-only source reprojection is permitted. Correct objectively invalid I/O or implementation bugs and rerun affected cells; do not disguise GT-driven tuning as a bug fix.

### 5.1 Deterministic shared observations

Use the original sampled mapping sequence, not every arbitrary frame in the dataset. Apply the existing pose-distance convention to select a temporally ordered pose-distinct list: a representative must differ from each previously selected representative by at least 5 cm translation **or** 5 degrees rotation. Use frame/pose validity only, not GT or candidate success. From that list choose at most 32 evenly spaced temporal indices, deduplicated deterministically. Record all chosen IDs and exclusions. Both model runs use the same frame set.

Pose-distinct is an engineering redundancy criterion, not statistical independence. A maximum of 32 frames is a computation budget, not a claim that 32 is scientifically optimal. Do not choose the best-looking views separately for each source.

### 5.2 Shared visibility and frame-mask data

Project the **complete frozen common cloud** into each selected camera. Respect receipt-bound depth units, camera-to-world convention and RGB/depth/mask registration. Transform world points to camera coordinates; use camera-z depth, positive finite z and the actual intrinsics. Use the integer pixel-center convention from the existing code, with a documented nearest-pixel rule and fixed tie-breaking. Never silently resize masks without nearest-neighbor registration and updated intrinsics.

Build one scene z-buffer per frame, independent of candidate and source. Keep the nearest common point per pixel; break equal-depth ties by original point index. A sample contributes only when measured depth is positive/finite and `abs(z_rendered - z_measured) < 0.05 m`. Invalid/out-of-view/occluded samples abstain. Do not infer foreground free space from a missing sample, or let each candidate use its own visibility buffer and see through other surfaces.

Keep original frame-local positive instance IDs. Zero/unknown is absence of usable instance evidence, not proof that a candidate is wrong. Numeric mask IDs across frames are unrelated. The visible image domain used for candidate-mask comparison is shared across all sources and restricted to valid projection/depth with a positive observed instance ID.

Build pixel-count contingency statistics from z-buffer point membership; count each image pixel once, not every point that landed on it. Dense point sampling must not manufacture more independent evidence. Reuse visibility buffers between runs when their coordinates match exactly; candidate membership calculations remain run-specific.

Do not use the OVI rendered owner value as the observation label. It may establish geometry alignment, but the identity evidence must come from the archived frame prediction. Record that these predictions can be correlated with OVI's frontend and that repeated mistakes remain possible.

### 5.3 Neutral atoms and allowed candidates

Create small prediction-domain atoms **once per run**, shared by LOCAL and SPATIAL. Start with 2 cm world-grid cells, refine by the exact frozen U00 candidate-membership signature, and split disconnected point groups within a cell/signature using 1.5 cm radius connectivity. Use actual metric coordinates; retain deterministic point-to-atom mappings.

Each atom therefore has a common candidate set `C_a`: every candidate in that set covers **every original point in the atom**. Membership signatures are representation boundaries, not assigned owner labels. Never initialize an atom label by declaring its old OVI parent immutable; an SF candidate can win adjacent atoms inside or across old OVI regions wherever its original mask supports them. Do not import old parent-scoped G1/G2 atoms or their failed merges/splits as the answer.

Points with no candidate keep owner 0. Single-candidate atoms keep that candidate. Conflict atoms have two or more allowed candidates, including SF-only conflicts and same-class conflicts. Do not restrict optimization to different-class conflicts or just old OVI boundaries.

Keep subcell connectivity computations spatially bounded; no global all-point distance matrix. A dense local cell must use a radius-neighbor/hash implementation rather than a quadratic distance allocation. Build sparse atom/candidate incidence arrays. If exact signatures produce many atoms, chunk and report the count; do not merge incompatible signatures just to reduce memory.

The U00_OVI_FILL label at every atom is well-defined because membership is constant. Use it as a reproducible fallback and a **soft preference**, not an unchangeable OVI lock.

### 5.4 Candidate-to-observation correspondence, then local support

For each candidate and frame, form its visible-pixel mask on the shared valid image domain. Compare it with each positive frame-instance mask by visible IoU. These are geometric, class-agnostic comparisons; do not use GT categories or the final borrowed label.

For scoring atom `a`, choose the frame-mask correspondence **excluding the atom's own rendered pixels** from both operands. This leave-atom-out operation prevents a tiny atom from providing the entire evidence for the identity it is then assigned. Compute it by subtracting atom contingencies from the cached candidate/frame table, not by repeatedly rendering the cloud. Missing remainder means abstention. Small objects may consequently lack usable evidence; record that limitation instead of secretly dropping the rule.

Use the best remaining IoU match only when: intersection >=16 pixels, best IoU >=0.20, and best-minus-second IoU >=0.05. A sole match has second score 0. These defaults are proposal assumptions; freeze them for this run. Ambiguous correspondence abstains instead of relying on a mask-ID tie to decide ownership.

For a usable `(a,i,f)` correspondence, let `r_aif` be its leave-atom-out IoU and let `p_aif` be the fraction of the atom's valid positive-mask pixels assigned to that matched frame mask. Other positive frame masks count as disagreement; missing/zero/occluded pixels do not enter this fraction. An atom/frame contributes at most one view-level observation, irrespective of pixel count.

Define the shared local support score:

```text
s_a(i) = (1 + sum_f r_aif * p_aif) / (2 + sum_f r_aif)
```

This is a smoothed evidence score, **not a calibrated correctness probability**. Use float64 accumulation and integer pixel counts. Store usable-view count, support mass, agreement, rejection reason counts and the contributing frame/correspondence IDs. No NaN-to-zero ambiguity: missing scores have explicit missing state.

Use the same computation for OVI and SF. Do not set all OVI local confidences to 1, add incomparable model class scores, or infer that depth agreement alone makes one instance correct. Do not add SF soft logits to only the SPATIAL condition; that would confound the LOCAL-to-SPATIAL comparison.

At atom `a`, the feasible labels are its fallback label plus candidates with at least two usable pose-distinct views. The fallback is always feasible and receives neutral support 0.5 if it lacks evidence. When no challenger has sufficient evidence, preserve fallback. This conservative restriction is part of the first hypothesis, not a proof that the fallback is correct.

### 5.5 Local unary decision

For feasible candidate `i`, define:

```text
D_a(i) = 1 - s_a(i) + beta * [i != fallback(a)]
beta = 0.05
```

Choose the minimum cost independently for LOCAL. On an exact cost tie, keep the fallback when tied; otherwise choose the smallest canonical index. Do not perturb AP scores or add source-specific epsilons. A candidate with stronger local evidence can beat OVI inside its old region; absent evidence does not authorize arbitrary reassignment.

Save the atom definitions, feasible sets, unary arrays and LOCAL owner map. These **same** objects are the input to SPATIAL. Running LOCAL at point level and SPATIAL on newly coarsened regions would add an atomization confound; do not do that.

### 5.6 A bounded spatial-coherence extension

Construct a sparse undirected graph of atom centroids using up to 6 nearest neighbors within 3 cm; symmetrize and remove self/duplicate/invalid neighbors. Radius-capped nearest-neighbor queries may return missing-neighbor sentinels: check finite distance and index bounds before indexing.

Distance alone must not create a strong same-instance edge across touching objects. For co-visible atom pairs, use their **original frame-mask distributions**, not borrowed semantic labels, to estimate frame-local same-instance agreement `q_abf`. One implementation is the dot product of their normalized positive-mask pixel histograms for that frame. It is zero when observed instance IDs differ. Use only the same selected pose-distinct frames. Require at least two shared usable views; otherwise edge weight is zero in this first version.

Let:

```text
g_ab = exp(-||x_a-x_b||^2 / (2 * 0.02^2)) * max(0, 2*mean_f(q_abf)-1)
d_a  = sum_b g_ab
w_ab = g_ab * min(n_a/max(d_a,eps), n_b/max(d_b,eps))
```

Use `eps=1e-12`. Here `n_a` is the number of original common-cloud points in atom `a`, not GT area. Unsupported edges have weight zero. Use one symmetric weight per undirected edge. This normalization bounds pairwise influence relative to atom mass and avoids implicitly multiplying it by graph degree.

Optimize the explicit energy:

```text
E(z) = sum_a n_a * D_a(z_a) + lambda * sum_(a,b) w_ab * [z_a != z_b]
lambda = 0.10
```

Initialize from LOCAL and use at most 5 deterministic sequential coordinate-descent sweeps, stopping early if no labels change. Visit atoms in a fixed geometry-derived order. Consider only the precomputed feasible labels. Accept a change only if the full local energy strictly decreases; retain the current label on ties. Record energy after each sweep. This is a bounded local solver, not a global-optimality claim. Fixed/unconflicted atoms may provide boundary context but never acquire unsupported labels.

No graph-cut package installation, giant neural model, joint feature learning, global mask erosion, connected-component post-splitting, minimum-size redistribution, or per-object GT fixes. The spatial condition changes the pairwise term only. Its smoother may damage small/nearby objects; measure that rather than assume a benefit.

### 5.7 First configuration and permitted deviations

Proposed defaults to write before new full GT evaluation:

| Parameter | Value | Status |
|---|---:|---|
| Selected frames | <=32 | New compute budget |
| Pose separation | 0.05 m or 5 degrees | Existing convention, reused |
| Depth visibility tolerance | strict <0.05 m | Existing convention, reused |
| Atom cell / within-cell connectivity | 0.02 / 0.015 m | New representation choices |
| Frame-mask remainder intersection | >=16 pixels | New correspondence choice |
| Correspondence IoU / ambiguity gap | >=0.20 / >=0.05 | New correspondence choices |
| Usable challenger views | >=2 | New evidence gate |
| Unary fallback preference beta | 0.05 | New regularization choice |
| Graph neighbors / radius | 6 / 0.03 m | New graph choices |
| Spatial lambda / maximum sweeps | 0.10 / 5 | New solver choices |

Use actual camera calibration and depth units from assets, not invented defaults. Necessary engineering changes for format compatibility or a proven implementation error must be logged with before/after reasoning. Do not silently increase frame budgets, change thresholds after seeing GT, add logits, or change the candidate pool to obtain a preferred score. A different scientific method needs a separately identified next experiment.

## 6. Integrate without weakening or defeating the existing evaluator

### 6.1 Two code traps that must be addressed

The old diagnostic runner recomputes globally prioritized projected-mask ownership and requires it to match the source owner map. That check is correct for old global rules but is **not the definition of correctness for local optimization**. [S5]

Use explicit family handling:

```text
assignment_family = global_priority
    keep the existing independent global-order/projection equality check
assignment_family = local_evidence
    validate source map membership/coverage/IDs and fixed projection directly
    never compare its result to the old global-priority solution
```

Do not put placeholder global `assignment_priorities` into a local record to make the old assertion pass. Do not delete legacy parity checks globally. Do not optimize again on the GT sample domain: solve on source points, then apply the bound nearest-neighbor projection.

The old runner also has closing checks requiring old `AT_ASSIGN_*`, `AT_RANK_*`, `AT_U11` results. Dispatch relevant checks by explicit method family/registered experiment set, or provide a small new wrapper around reusable export/evaluation functions. Do not manufacture those old files or rerun 34 old cells just to satisfy closing code.

The old generator's condition branching is not an extensible scientific method registry. New method names must be explicit and unknown names rejected; otherwise a newly named condition could accidentally execute raw ownership. [S4]

### 6.2 Preserve output contracts

For each run, new overlapping masks, kept IDs, class IDs and six-decimal rank values are identical to its U00 reference. Within the three new methods use byte-identical candidate manifests. If historical relative file paths differ, compare canonical candidate-to-mask-hash/class/serialized-score records exactly and record the path-only difference; do not confuse directory names with changed predictions. Candidate rank remains original raw **source-point area**, independent of local scores and owned area. There is no new probability/feature blending.

Unique owners are positive canonical candidate IDs (e.g. canonical index+1); 0 means uncovered. Owners must lie within that candidate's original mask. Every union-covered point has one owner and points outside the union remain zero. Do not discard candidate IDs just because their unique region becomes empty/small. The released evaluator decides its existing projected-size/ignored eligibility. Record these consequences; do not redistribute points to rescue them.

Region accounting uses the frozen **U00** support/multiplicity/original-class partition. LOCAL/SPATIAL may change ownership only in its multi-candidate regions. SF-only multi-candidate and same-class multi-owner conflicts are valid optimization targets. Keep original native geometry owner as provenance; an unknown semantic candidate is not unobserved geometry.

Reuse the exact released evaluator source and metric protocol. Keep 51 semantic vs48 instance vocabularies distinct, strict projection threshold, actual floating IoU thresholds/comparators, absent-class/null handling, minimum projected region size100 and six-decimal score parsing. Valid-GT unmatched vertices remain prediction0 false negatives. A canonical class-agnostic AP75 diagnostic is not released semantic AP75.

Reuse candidate AP/trace results when prediction inputs and protocol are identical; record the cache hit. Re-evaluate unique masks and semantic confusion for every new ownership map. Do not mark the old global commutation assertion `EXACT` for a local policy; report the actual local validation performed.

## 7. Concrete implementation work packages

Create only a small set of purpose-built modules. These paths are **proposed additions**:

```text
src/static_ovmap/ownership_evidence.py
src/static_ovmap/local_ownership.py
scripts/evaluation/run_static_local_ownership.py
scripts/evaluation/evaluate_static_local_ownership.py
configs/evaluation/ovimap_local_ownership_v1.json
tests/evaluation/test_static_local_ownership.py
artifacts/static_ovmap/local_ownership_v1/
docs/paper/static_ovmap/LOCAL_OWNERSHIP_RESULTS.md
docs/paper/static_ovmap/LOCAL_OWNERSHIP_HANDOFF.md
```

A thin evaluator wrapper is preferable to duplicating the metric implementation. Follow existing conventions when equivalent files already exist. Suggested function names below describe responsibilities, not pre-existing callable APIs.

| WP | Inputs and implementation | Concrete output | Proceed when |
|---|---|---|---|
| WP0 Bind and orient | Read the directly relevant sources and receipts; bind two pools, frame loader, metrics, masks and poses; preserve task branch state. | `input_binding.json`, frame IDs, source/version note. | Used inputs resolve; any missing input has a local impact statement. |
| WP1 Establish same-pool baseline | Reuse exact U00. Implement `LO_U00_OVI_FILL`; introduce explicit method/family metadata and new runner. | Two baseline owner files, verified candidate manifest identity. | IDs/support and legacy baseline semantics are correct. |
| WP2 Build shared evidence | `select_observation_frames`, `project_shared_visibility`, `build_membership_atoms`, `collect_local_instance_evidence`; load no model. | Shared projection/atom/evidence cache and coverage/abstention statistics. | One small real-cache smoke confirms geometry/mask alignment; no GT-dependent policy choices. |
| WP3 Local and spatial assignment | `solve_local_unaries`, `build_evidence_graph`, `solve_spatial_ownership`; reuse exactly the same atoms/unaries. | Four owner maps (two methods × two runs), energy/change logs. | Source-only ownership invariants hold; changed code passes scoped fixtures. |
| WP4 Evaluate and explain | Family-aware evaluator adapter; original AP/semantic code; object and region accounting. | Six real evaluation rows plus reused references, per-class metrics, object/region tables and costs. | Inputs are fixed and every row has actual status, not forecast values. |
| WP5 Handoff and publish | Write concise results/limitations and one next recommendation; commit scoped implementation/results/MD; push and compare remote SHA. | GitHub branch containing code and evidence; verified commit URL/SHA. | Remote verification succeeds, or a concrete push failure is disclosed. |

Dependencies are WP0→WP1/WP2→WP3→WP4→WP5. Do not turn each row into a new audit cycle. Integrate, fix actual defects, and execute. Negative scientific results still go to WP5 with the implementation and diagnosis.

### Cache and data contracts

The old global owner key hashes kept IDs, scalar priorities and groups. A local key must additionally bind coordinate/order and candidate-mask identity, run/query identity, observation frame set and mask/depth/calibration identities, atomization, evidence algorithm/version, feasible sets/unaries, graph/parameters, method family and relevant source hash. LOCAL and SPATIAL share the evidence key but have different solver keys. Changes to evidence cannot reuse old owners accidentally. [S4]

Keep one immutable mask bank per run, not one full copy per method. Keep sparse atom/candidate scores and edge arrays, plus source `owner_path`, graph/unary references, labels/ranks/kept IDs, stable candidate ledger and actual timings. Separate GT annotations into evaluation artifacts. Avoid a huge general framework or a serialized Python object requiring unsafe pickle loading.

## 8. Resource discipline: meaningful checks, not excessive safety testing

No full-repository test run, CROVE dynamic safety audit, repeated broad filesystem search, new security/fuzz campaign, exhaustive threshold sweep, repeated three-round checklist ritual, or model reinstallation. Do not repeat the previous evaluator-trace audit when its code is unchanged. Preserve necessary scientific invariants with small fixtures and a real-cache smoke.

Use the existing compatible environment from receipts. Do not merge model/mapping environments or install an unrelated stack. Keep CPU thread count explicit (start with 8, respect the shared host). Process one model run at a time unless measured memory allows safe sharing. Chunk projection and mask statistics (e.g. 65,536 point chunks); no C×N×F evidence tensor, no N×N distance matrix, no giant dense label-by-point optimizer. N denotes all common-cloud points, not just a tiny demonstration.

Use one shared z-buffer per frame and compact candidate/frame-mask contingencies. Avoid Python loops over all point/candidate/frame triples. Use sparse atom incidences and vectorized count subtraction for leave-atom-out correspondence. Cache only needed summaries; optional raw logits remain memory-mapped and unopened in this first method.

Run focused tests for these eight behavior groups; this is not a demand for any particular test count:

1. Exact U00 identity; same-pool OVI_FILL rule; no unintended label/mask/rank/NMS changes.
2. Camera-z, integer-pixel projection, z-buffer ties, invalid depth, occlusion and unknown-mask abstention.
3. Frame-local IDs/correspondence, leave-atom-out counts and insufficient-evidence fallback.
4. Atom allowed-label sets, positive owner IDs, coverage and the ability to assign SF inside an OVI region without expanding any mask.
5. LOCAL/SPATIAL identical atom/unary keys; sparse graph boundary behavior and non-increasing sequential optimization energy.
6. Family-aware evaluator: legacy global equality still checked, legitimate local output accepted under local invariants, unknown method rejected.
7. Candidate manifest identity, local cache invalidation, unique empty/small accounting, unchanged rank during ownership export.
8. Regional matrices reconstruct the global matrix including unmatched GT; same-class owner changes are not lost in semantic summaries.

Run only directly affected existing regression files in addition. One small real input smoke verifies I/O before the six full cells. A smoke is not full-scene performance. Never assert that AP must increase in a unit test. Rerun failed checks and affected experiments after genuine fixes, not all previous research.

## 9. Required results and scientific interpretation

### Table A — paired performance

For each run, show reused common-domain O-only/U00/U11 references and the three new methods. Separate overlapping candidate AP from unique AP/AP50, canonical high-IoU diagnostic and semantic mIoU/mAcc. Include eligibility/empty/ignored counts, evidence coverage, source counts and measured costs. Use proportions in machine files, percentages in display and percentage points for deltas. Do not use the historical 18.672 native-domain AP as the only comparator.

### Table B — controlled changes

Report per run:

```text
U00_OVI_FILL - U00_raw                 conservative protection on the same pool
LOCAL - U00_OVI_FILL                  local evidence beyond protection
SPATIAL - LOCAL                       pairwise consistency, same unaries/atoms
LOCAL and SPATIAL - common O-only      actual net unique-map value
```

Candidate AP should not change in any of these ownership comparisons. If it does, investigate the intervention leak before interpreting a gain. Do not add geometric transfer/S1a effects to the method's contribution.

### Table C — object gains/losses and retention

Reuse original released matching/event logic and report strict runtime thresholds near .25/.5/.75. Count which U00-added GT objects survive each new unique map, which previously correct OVI objects are lost/recovered, and what happens to same-class overlaps, small objects and large surfaces. Show actual GT IDs and candidate source IDs, not only favorable images. The previously reported 7/6 gains are an evaluation reference, **never a list of objects to special-case**.

If a candidate has a correct overlapping mask but loses unique support, record its owned fraction, boundary/overmerge/fragmentation changes and relevant conflicting candidate. A low owned fraction is not automatically wrong: duplicate masks can correctly become empty. Never promote all zero-area losses to false negatives without the actual evaluator.

### Table D — evidence and regional diagnosis

Report source support × candidate multiplicity × original-class-agreement, plus projection-unmatched. Give correct→wrong, wrong→correct, wrong→wrong, owner/class changes and integer confusion deltas. Include evidence-observed fraction, number of challenger-qualified atoms, local-score margins, changed-point fraction, no-evidence fallback fraction, graph-supported edges and graph label changes. Region confusion matrices must sum to the full valid-GT matrix.

This table must distinguish:

- no new information because observations/correspondence are unavailable;
- local evidence exists but chooses the wrong owner;
- LOCAL helps but SPATIAL damages boundaries/small instances;
- graph has no usable edges and therefore adds no changes;
- semantic classes improve while unique instances deteriorate;
- a remaining class disagreement cannot be resolved by the fixed geometry-only evidence.

Score/error diagnostic plots may use GT **after** prediction. Do not tune the rule on the held outcome or select a different policy for each object.

### Costs and decisions

Measure asset/cache preparation, selected-frame projection, atom/evidence construction, graph building, each ownership solver, export and evaluation separately; include cache-hit flags and peak memory when actually measured. Do not report warm-cache fusion time as end-to-end FPS. Model inference/training counts stay zero.

Distinguish implementation completion from method success. A useful development signal requires net unique-instance improvement over the proper O-only and same-pool protection references, alongside declared semantic/high-IoU tradeoffs and object retention. Improving only from the very poor raw U00 unique map is insufficient. Two same-scene runs are not generalization or statistical significance.

Choose an evidence-supported conclusion such as:

```text
COMPLETE_LOCAL_EVIDENCE_GAIN
COMPLETE_SPATIAL_ADDITIONAL_GAIN
COMPLETE_NO_NET_UNIQUE_GAIN
COMPLETE_EVIDENCE_NOT_DISCRIMINATIVE
COMPLETE_EVIDENCE_COVERAGE_INSUFFICIENT
PARTIAL_REQUIRED_ASSET_MISSING
```

Do not force a gain. Do not select the better model run. Do not keep adding postprocessing rules after the six cells. State which failure mechanism is supported and the single best next experiment. Future scene confirmation may use predeclared Room1/Office1 with frozen settings and necessary separate authorization/data preparation; do not launch it inside this task.

## 10. Deliverables and mandatory GitHub synchronization

Commit these scoped items, adjusted to actual implemented names:

- New local-evidence/ownership code, small evaluator integration changes, config and focused tests.
- `LOCAL_OWNERSHIP_RESULTS.md`: the four tables, interpretation, negative outcomes and limitations.
- `LOCAL_OWNERSHIP_HANDOFF.md`: exact executed commands with interpreter/env and exit codes; base/implementation identity; which cached runs/frames were used; per-condition status; reproduction steps; remaining issues; one next experiment.
- Small machine-readable binding, protocol reference, candidate-manifest parity, solver/evidence summaries, performance/effect/object/region tables, costs and test receipts.
- A large-artifact manifest with actual external paths, sizes and hashes for new owners/atoms/evidence; retain them on the existing shared storage. Do not upload giant masks/meshes, original datasets, weights, credentials or copies of raw model logits.

Do not leave the only handoff under `/mnt/shared`. The two Markdown files and small evidence must be present in the **remote GitHub branch** together with the implementation. Use relative repository links in the handoff so a reviewer can navigate the evidence from GitHub. State which large files still require the shared filesystem.

Execution order for publication:

1. Inspect `git status`; stage only listed task files, not unrelated edits or large data.
2. Commit after the recorded focused tests and actual experiments. Include negative results and `NOT_RUN/BLOCKED` reasons where applicable.
3. Push the task branch to the intended `origin` for `Orangekostar/oviovo`, without force.
4. Compare `git rev-parse HEAD` to `git ls-remote origin refs/heads/research/ovimap-local-ownership-v1`. Report `PUSH_VERIFIED` only if they agree.
5. Report the full SHA, branch, commit link and result/handoff paths in the final response. If push fails, keep the local commit and state the concrete error and `NOT_PUSHED`; do not assert remote success or repeatedly retry credentials.

The handoff need not embed its own final commit SHA. Record the reviewed base and actual execution source/config hashes inside it; provide the final commit SHA from Git in the final response. Avoid recursive commits made solely to update a self-referential SHA.

The user's explicit push request authorizes these task-scoped repository writes. It does not authorize changing unrelated branches, deleting old results or publishing restricted large assets.

## 11. Minimal final response from Codex

Report, in a compact table-first format:

- six condition statuses and actual inference count;
- LOCAL/SPATIAL unique AP/AP50/mIoU and deltas to common O-only and U00_FILL for **both** runs;
- candidate AP preservation and gained/lost-object retention;
- whether local evidence helped and whether spatial consistency added value;
- focused tests actually run, timing boundaries and remaining limitations;
- GitHub branch, full verified SHA, result MD and handoff MD paths.

Do not finish with another proposed plan while executable work remains. Do not describe expected gains as measured results. Do not call a partial asset-blocked run a completed six-cell experiment.

## 12. Why this plan does not start by using every cached signal

Mask2Former provides a useful official example of object scores combined with position-dependent mask outputs, and Open3DIS provides a primary-source example of complementary 2D/3D instance evidence. They motivate ideas; neither proves that two unrelated confidence spaces are interchangeable. [E1–E2]

For this first bounded implementation, both sources use the **same observed-mask evidence rule**. SF's raw soft masks and class probabilities remain available for a later controlled extension; they are deliberately not mixed into one method only or compared directly against an OVI confidence of 1. This keeps the result interpretable and avoids another multi-module ensemble whose gain cannot be attributed.

The new pixel correspondence, atomization, unary and graph are our proposed adaptation. Do not label the implementation as a reproduction of Mask2Former, Open3DIS, the old R4 split, or CROVE temporal ownership.

## 13. One-pass self-review before publishing

Check only these task-specific questions after execution: Do all six rows use U00? Are the LOCAL/SPATIAL unaries identical? Was local ownership validated as local rather than faked as global priority? Is candidate AP unchanged? Are GT/masks/source IDs confined to their proper roles? Are both gains and losses reported? Are code, small results and both MD files pushed and the SHA verified?

Fix actual inconsistencies. Do not launch another broad audit campaign or repeat passing benchmark cells without a changed prediction/evaluation input.

## 14. Immutable source register and evidence limits

[S1] Submitted attribution evidence, not a rerun in the current chat:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/docs/paper/static_ovmap/T1_ATTRIBUTION_RESULTS.md  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/docs/paper/static_ovmap/T1_ATTRIBUTION_AUDIT.md

[S2] Frozen candidates/global ownership:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/src/static_ovmap/fusion_attribution.py

[S3] Legacy wrapper:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/src/static_ovmap/proposal_fusion.py

[S4] Existing runner/cache/condition dispatch:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/scripts/evaluation/run_static_t1_attribution.py

[S5] Existing evaluator integration and global-priority assertion:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/scripts/evaluation/diagnose_static_t1_attribution.py

[S6] Geometry visibility and pose convention:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/src/static_ovmap/geometry_support.py

[S7] Existing frame-instance evidence:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/src/static_ovmap/instance_graph.py

[S8] Native-parent split evidence, not a neutral ownership solver:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/src/static_ovmap/split_evidence.py

[S9] Component semantic readout, not an atom constructor:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/src/static_ovmap/hierarchy.py

[S10] Original model output/schema:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/scripts/evaluation/run_static_spaceformer.py  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/artifacts/static_ovmap/room0_spaceformer/large_artifacts.json

[S11] Existing region/object/evaluator trace helpers:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/src/static_ovmap/attribution_regions.py  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/src/static_ovmap/attribution_objects.py  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/src/static_ovmap/released_trace.py

[S12] Semantic evaluation and receipt-bound protocol:  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/scripts/evaluation/evaluate_static_ovmap_readout.py  
https://github.com/Orangekostar/oviovo/blob/98cafc4a96bf906878116ae3284da28bc974ad4d/configs/evaluation/ovimap_t1_attribution_v1.json

[E1] Official Mask2Former inference implementation (conceptual reference, not our validated adapter):  
https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py

[E2] Open3DIS authors' project (conceptual reference for complementary instance evidence):  
https://open3dis.github.io/

[E3] Official SciPy radius-capped nearest-neighbor query contract, including missing-neighbor sentinels:  
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html

The source audit supports the observed pipeline limitations and available interfaces. It does **not** establish availability of all server-side frame files, predictive accuracy of the proposed score, or expected benchmark gains. Resolve operational asset details from actual receipts; report measured outcomes after implementation.
