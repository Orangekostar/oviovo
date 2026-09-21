# 04 — Q: causal, pre-encoding query utility under a common budget

## Q0. Scope and honest name

Implement a causal replay driven by CONTEMPORANEOUS native mapping states captured during the unchanged OVI trajectory. This is not final-map retrospective top-K selection, robot trajectory planning, or end-to-end real-time deployment. Geometry formation in the inspected native main loop does not consume the VLM output; therefore its time-indexed native geometry snapshots can be shared across query policies without replaying the TSDF for every policy. Verify this property in the actually patched execution path and record it.

A Q policy sees current RGB-D geometry/region metadata and previously acquired successful features only. It cannot read the final map/registry, final incumbent labels, future frames, GT, or unacquired per-view VLM features. Final registry/projection may be used only at end-of-sequence export/evaluation. A file path to a precomputed feature is NOT permission to access it before acquisition.

Q v1 uses only frozen native SigLIP with the original six crops. It tests budgeted query RANKING. It does not learn expert routing or a separate optimal stopping rule. Finite per-frame budget availability, technical eligibility and sequence termination determine skipping. This smaller formulation prevents a negative utility threshold from silently disabling the whole experiment.

## Q1. Split native candidate generation from native decisions

The current `select_views_for_frame()` both filters candidates and mutates coverage. Do not merely call it and rank the already filtered output; that cannot test acquisition outside native choices. Create a pure candidate builder preserving native technical requirements and metadata: current nonzero global owner, global area>=500, majority-local-entity area>=1000, usable current depth and a nondegenerate native crop. Preserve the native handling of majority entity0 and document it rather than silently changing the baseline. Expose local/global masks and union; retain selection-independent geometric coverage features.

Keep an unmodified native combine implementation as a control with its original side effects. In particular, the current coverage routine mutates its array before the later visible-area check; a seemingly harmless refactor must not retroactively rewrite historical control semantics. New policies use their own explicitly separate state and the same raw candidate universe. Two states are distinct: all geometrically observed surface coverage (no visual encoding needed), and coverage/history of actually acquired visual evidence. Name both; never count an unqueried view as paid semantic evidence.

Capture genuine frame-local IDs, registered segments and alias events BEFORE clearing native temporary memory. Maintain query-state aliases only as they become known at that time; merge past feature lists and spend counters by known ancestry. Do not discover final aliases early. A split whose retained feature provenance cannot be assigned to the current child is marked unavailable for that child, not blindly copied. Unresolved lineage blocks causal claims with `BLOCKED_CAUSAL_LINEAGE`, but offline S/G may continue.

For deterministic module isolation, use a per-frame result barrier for ALL Q policies: select and debit the frame's whole batch using only the pre-dispatch state, run/retrieve those paid queries, then admit successful features before the next frame. Record this controlled schedule. It is not a throughput claim about the author's asynchronous worker. Do not let another policy's cache fill the current policy state.

## Q2. Budget, readout and four policies

Main allowance B=200 target-view query ATTEMPTS per scene. One attempt costs six native crop inputs; failures cost their actual attempted work and consume their one-attempt allowance. A repeated exact request is skipped rather than repaid. For planned frame index t=0..F-1, the cumulative allowance is `floor(B*(t+1)/F)`; unspent allowance carries forward. Invalid input frames still advance planned time. The known native frame schedule F is allowed metadata; future model outputs and final geometry are not.

At each frame, quota is cumulative allowance minus attempts already spent. Rank currently technically eligible requests not previously attempted; select up to the quota. There is no per-object acquisition cap in Q v1. Keep at most the10 largest-visible-area SUCCESSFUL features for the common current semantic readout, tie by frame/request ID; all paid attempts remain in the cost ledger. Retention is not a refund. The first query of an unseen object has no cached label: its prior is uniform over the dataset vocabulary. After queries, area-weighted feature fusion followed by cosine classification is identical for all four policies. At export, an instance with no usable acquired evidence has class0; do not give it the final native label for free.

| ID | Selection from currently available requests |
|---|---|
| Q_COMBINE | requests that the native combine control would admit from its OWN paid/query state; descending overlap area then owner/request ID under the allowance |
| Q_AREA | all technical candidates, descending overlap area, deterministic IDs |
| Q_UNCERTAINTY | all technical candidates, descending normalized entropy of past acquired class scores; unseen=1; ties overlap area then IDs |
| Q_GAIN | all technical candidates, descending predicted one-step loss reduction; ties overlap area then IDs |

Q_GAIN fills its available quota even when predicted scores are negative; primary v1 measures relative acquisition priority, not stopping. This removes a confounded reject-everything gate. Q_COMBINE can use fewer requests because its native novelty gates exclude candidates; report this. Q_AREA/UNCERTAINTY/GAIN have the same raw eligibility but may differ in successful cost. Describe the main result as an equal **allowance**, not necessarily equal realized FLOPs; actual debit/cost curves decide stronger cost-dominance claims.

No teacher calls, background-only crops, S selector, G refinement or re-ranking enter these four standalone Q rows. The unbudgeted original native N0 is separately shown for context, not called an equal-budget control.

## Q3. Training-only bounded acquisition exploration

Use FIT native contemporaneous snapshots, a fresh initially empty semantic state, and seed17 random request ranking within each frame; maximum512 six-crop attempts/scene, paced as in Q2 with B=512. No acquisition target uses final-map geometry. Log candidate metadata and prefix state before selection, then the acquired feature and readout change. All image encoding for these training labels is counted as training cost, not hidden in inference speed.

On CAL collect an independent seed23 random trace with B=256 per scene for head checkpoint assessment. Do not create exploratory inference traces on SELECT/CONFIRM. Existing compatible randomized traces may be reused exactly; native-selected-only logs cannot be claimed to be unbiased exploration data.

A separate supervised target builder projects authorized annotations onto the CURRENT visible region. A target is identifiable if it has at least64 valid annotated region pixels and one allowed class accounts for>=0.8 of them. This is a training-label rule only, not query eligibility. The GT category never appears in policy features. If no identifiable target, retain the paid trace but exclude its loss target.

Before/after distributions use softmax(current native cosine/0.07), or uniform prior before any feature. Target `y=clip(NLL_before-NLL_after,-5,5)` for the same identifiable class. New queries can increase loss; keep negative labels. This is a semantic acquisition proxy, not AP decomposed into additive rewards. Do not train only on positive queries or label an unobserved action with fabricated gain.

Require>=100 identifiable FIT acquisition events from>=4 physical FIT scenes, including at least10 positive (>1e-6) and10 nonpositive events. Otherwise Q_GAIN fitting is `BLOCKED_QUERY_TARGET_SUPPORT`; the other three policies are still measured if causal capture works.

## Q4. Fixed utility head

Twenty scalar features plus20 availability bits, in this order:

1 log1p(current global mask pixels); 2 log1p(local-majority mask pixels); 3 intersection/global fraction; 4 intersection/local fraction; 5 target/bbox fraction; 6 valid-depth fraction; 7 median visible depth; 8 depth interquartile range; 9 new geometrically observed spherical-cell fraction; 10 new acquired-evidence spherical-cell fraction; 11 log1p(past successful queries); 12 log1p(past attempts); 13 frames since last successful query / F; 14 past retained-feature normalized class entropy; 15 past top1-top2 cosine gap; 16 mean pairwise retained-feature 1-cosine; 17 current overlap area / max(1,best previous paid overlap area); 18 camera translation from last paid pose in meters; 19 camera rotation from last paid pose in radians; 20 remaining attempt fraction.

All unavailable history features use0+availability0; no vectors of class IDs or scene IDs. Normalize with FIT-only statistics as in S. No image encoder, current-crop text score, teacher answer or hidden-state computation may be used to rank an unacquired request. Cheap geometry-based quality computation is allowed and timed.

Head40->32(ReLU)->1(linear), seed17, Adam0.001, weight_decay0.001, Huber(delta1) loss, batch128, max100epochs, CAL MSE every5epochs, stop after3 nonimproving evaluations, choose lowest CAL MSE/earlier epoch. Weight scenes equally. One fit, no hyperparameter grid. Clip predicted gain to[-5,5] only for numeric stability; ranking unchanged except ties. Negative predictions remain valid.

Save a prefix-sentinel test: replacing any unacquired or future cached feature with an impossible sentinel must leave all decisions prior to its actual acquisition unchanged. This is a targeted causal-integration check, not a generic security audit.

## Q5. Evaluation, optional budget curve and cache accounting

Run the four policies at B=200 on SELECT. Evaluate their final maps using the same geometry, vocabulary, ranks and released projection. Then, ONLY when Q_GAIN is eligible under file05, run Q_GAIN and the frozen best nonlearned Q comparator at B=100 and400. These four additional curve rows are prescribed, not another model sweep. Do not refit the head or change readout for the curve.

Maintain two ledgers:

- physical execution cost: all actual forwards/cache hits, model loads, tiles, inference times;
- policy logical acquisition cost: selected request attempts, successful requests and crop-equivalent cost, debited even when a feature is physically cached by another method.

A shared feature store sits behind `acquire(request_id, budget_token)`. The policy sees only admitted returned features; it cannot inspect the store. Acquired cached data can save physical computation but not a method's logical budget. Do not precompute all SELECT/CONFIRM request features for convenience. Document retained/evicted features and failed-query costs.

Report uAP, AP50, mIoU, classes/masks without observations, actual requests/crops, realized cost, and optional curves. If gain is only lower compute at similar accuracy, label it efficiency evidence. If current-map snapshots are missing and only final-map reprojected requests exist, label an offline diagnostic separately and BLOCK the causal Q experiment; never rename it online acquisition.
