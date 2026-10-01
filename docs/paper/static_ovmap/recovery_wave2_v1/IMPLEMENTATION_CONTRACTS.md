# Algorithm and integration contracts

These describe NEW task-local behavior. They do not assert that the APIs below already exist.
Numbers are fixed in PROTOCOL_SPEC.json. The prior study is immutable.

## C0. Three different namespaces

Keep these distinct: current-frame 2D group/track, native superpoint label, final global owner.
A recovery candidate is a final RAW owner. A SAM track is not a global owner. A source vector
belongs to a physical request plus proven final lineage, not a matching integer.

Keep four separate identities: map geometry; owner partition; semantic output; evaluation
context. Geometry identity contains xyz/faces/TSDF/projection. U preserves those coordinates
but changes the output owner partition by filling zeros. W changes only semantic decisions.
A/S can change the map and require new anchors/projections. Do not substitute a new partition
into an old prediction identity.

## C1. Correct interpolation in the presence of missing evidence

Compute pN,pQ,pF from each real source's own cosine vector and inherited temperature. Do not
softmax already normalized probabilities again. Let E be exact inherited FC_EQ, D exact D2.
For gamma in {1/3,.4,.45,.5}, t=6*gamma-2 and W=(1-t)E+tD. Return a copy of E or D at endpoints.
When all three exist this is the group formula. With one native source and F, E=D=(pNative+pF)/2;
with no F use available N/Q; with only F use F; with no source use the unchanged native class
and an unavailable flag (no fabricated uniform vote).

This piecewise handling is intentional. Using (1-gamma)*pG+gamma*pF on a missing-N/Q object
would not reproduce the old endpoints. Store full probabilities and actual source availability.
Tie-break argmax by the existing category order. Compare class decisions, not float strings.

## C2. Independent RAW registry and append-only exported instances

Inputs per map: raw owner R[v], old painted owner O[v], old class Y[v], geometry vertex/face
arrays, saved native observations, captured requests, alias table, and final segment-owner
pairs. Construct without opening annotation labels or target coordinates.

Candidate o must satisfy:

- o>0 exists in raw R, but o is absent from positive O;
- K_o={v:R[v]=o AND O[v]=0} contains at least100 source rows;
- K_o is disjoint from every incumbent output mask and every other candidate by construction.

Rank candidates before neural inference by (-|K_o|, hash(canonically sorted world coordinates
of K_o including the representation's repeated vertices)). Do not hash the numeric owner as
priority. Keep at most128. The source-row gate/cap are new method settings, not GT thresholds.
Log tiny/covered/cap/no-lineage/no-evidence exclusions. No cap tuning by Replica outcome.

At final export, a source-successful candidate gets the SAME K_o in all U arms where it exists.
Old O/Y remain exactly unchanged on O>0. Assign a unique positive output owner ID; preserve raw
ID when it does not collide. Otherwise allocate a deterministic new ID and record the mapping.
The raw-to-output correspondence is not inferred from color equality alone. Do not fill raw0,
expand an incumbent, merge candidates, smooth boundaries, or replace all old triangle painting.

Compute the full expanded owner/class payload and its genuine rank table. The coordinates and
projection can be reused exactly, but the output registry/masks and official ranks cannot.
Do not call the old label-only relabel_prediction on a changed owner array. Use PredictionPayload
with a supported branch tag or a new explicit compatible payload. A positive recovered owner
has one actual valid class; candidates with no evidence leave zero rows unchanged.

An output geometry improvement claim is NOT warranted here: this recovers output coverage from
a fixed raw map. Show raw geometry unchanged and added exported-instance coverage separately.

## C3. Legal visual observations and U-arm definitions

Build final segment->raw-owner sets from the actual final surface and follow the captured
alias graph. Reuse reconcile_semantic_request with the raw active set. Every recorded segment
must converge to the same candidate; ambiguous lineage is not repairable by matching owner IDs.
No GT association or final target projection may choose RGB views.

U1: exactly one SUCCESSFUL RETAINED native observation in its native saved row, corresponding
to an actual native-selected request that resolves to the candidate. Validate bbox/frame/mask
and stored feature identities. Reject color/ID ambiguity. Use the original feature and native
text/canonical FP32 classifier with the minimum-count condition changed only in this standalone
new function. Class score computation must match the old function for an artificial duplicate
single-view input, up to normal FP32 tolerance; this is a local sanity check, not a new observation.

UQ: load the stored query_scores.npz AFTER its recorded final reconciliation; require available=True,
finite complete scores in the right class order and a trace proving retained paid evidence resolves
to the same raw owner. Use argmax of the original Q scores. No new views or budget are added.

U2: fix precisely U1's selected candidate and single request. Produce its FC region feature from
the existing global target mask, signed-mask pooling and original FC projection. This is allowed
to need new pooling or a capped frame encoding. No alternate view if the mask disappears on the
dense grid. Report paired U1/U2 classification on the common-success subset; the full output still
includes their actual differences in coverage.

U3: for each capped candidate, consider all legal CAPTURED requests, not just native-selected ones.
Take max3 by (-visible_target_pixels, frame_id, physical request digest). Use area-weighted original
FC aggregation and its unchanged text prototypes. A single successful FC view is permitted exactly
as in the old FC source; no success is unavailable. No new Q/N contribution is silently added.

A request may view a larger raw object than the unowned output subset K_o. Preserve the exact
existing observed target mask and quantify the clipping ratio |K_o|/|R=o|; do not pretend that
cached pooling was performed on the smaller K_o. Rendering a fresh K_o image mask would be another
representation experiment and is out of scope. This distinction is critical to honest attribution.

Primary U3 uses no final-map reprojection to create absent requests. Report those candidates as
NO_LEGAL_CAPTURED_VIEW. A future reprojection study may be useful, but it is not this implementation.

## C4. FC reuse and deterministic acquisition budget

Physical identity includes weight files/revision, effective dense/region operators, FP32 precision,
image tensor preprocessing and exact pixels. Lineage and output-root names are not model changes.
Index the parent's dense/region receipts once; verify arrays when consumed. Parent cache
files remain read-only; new misses and receipts are written only to the new attempt. Reuse parent text
only after class order and14 templates match. Do not make a new model key merely because a
registry wrapper file changed, nor ignore a real operator change to force a cache hit.

To allocate the recovery-only cap, concatenate U2 planned requests and U3 planned requests in
candidate/view order, remove identical physical requests, enumerate distinct image tensors, and
mark cached frames first. Authorize the first32 distinct missing-frame encodings. Assign this
allowance before any new classification/GT. Freeze the comparison's initial cache inventory;
newly created files may serve already authorized requests but cannot retroactively enlarge the
32-frame allowance. For a new map, take this inventory after its standard readouts are locked.
Individual U2/U3 requests retain their preselected
views; requests beyond the cap are unavailable. A failed authorized frame does not authorize a
33rd frame or an alternative view. Cached data are not a claim of zero standalone compute.

Materialize a task-local request object/manifest for the expanded raw registry. Preserve captured
fields; use `_load_request` on genuine captured targets. The old prepare_semantic_manifest requires
native-painted active owners, so it must not be used unchanged for these orphan candidates.

## C5. Per-group native fallback throughout C++

New action representation: {local_group, action, existing_owner|null}. USE_NATIVE is not owner0
with a different spelling. Zero in the old beginBackboneAssociation immediately allocates a new
owner; this must not occur on fallback.

At frame start preserve the old state; do not pre-increment highest_instance for fallback groups.
Allocate only for explicit CREATE_NEW (not selected by the main experiments) or by genuine old
native logic during integration. Snapshot prior owner lookup at factor0 in mode4.

Required behavior across layers:

| Layer | USE_NATIVE | ASSIGN_EXISTING |
|---|---|---|
| candidate generation/recompute | execute old eligibility/count path | apply compatible planned-owner restriction |
| fresh superpoint bookkeeping | original native behavior | remember the planned global owner, not solely local group |
| mode4 instance update | execute original current-to-global logic | write the planned existing owner using original confidence updates |
| assigned-instance reservation | ordinary native behavior | allow explicitly planned follower groups to reuse this owner |
| aliases/merge evidence | no blanket prior-owner veto | protect labels actually constrained to conflicting explicit owners |

Keep hard-constraint tokens only where an accepted plan actually constrains a label. Do not load
all prior owners as unconditionally protected tokens: that changes native behavior even for every
fallback group. A merge involving two incompatible explicit tokens is rejected. A fallback label
that is later genuinely constrained inherits that token. Mixed interactions are logged. A merge
between unconstrained labels takes the old path. Use proper union/alias propagation of tokens.

The all-USE_NATIVE case must pass the serial native off comparison. A mixed case does not promise
complete global identity to native: explicit plans can legitimately affect a shared assignment set.
Record this rather than asserting every fallback pixel is independent of all constrained groups.

A2/A3: compatible follower groups share an existing owner, not necessarily a superpoint. Native
per-frame superpoint uniqueness constraints may remain where required; multiple superpoints can
belong to one owner. Do not force all fragments into one label or restore one-to-one owners through
`fresh_group==local_group` or the global assigned_instances set.

Keep existing positive confidence magnitudes, data_association2, mode4, depth fusion and TSDF update.
Do not add geometry weighting, pseudo-label learning, temperature changes, or permanent cannot-links.

## C6. Exact A1/A2/A3 planning

Use own preinsert probe only, with the already implemented depth-consistency criteria. Local groups
are the union of current foreground fragments with the same positive input instance ID. Masks are
disjoint in this frame. Let I_lg be intersections; A_l valid local area; B_g current known prior-owner
visible area. F_lg=I_lg/A_l, zero if A_l=0. Existing eligible edge: I>=100 AND F>.2.

A1: use the parent's positive-benefit Hungarian assignment on F-.2 plus per-row dummy nodes.
Accepted edge => ASSIGN_EXISTING; every other local group => USE_NATIVE. No reverse check.

A2: independently choose the eligible g with largest F for each l. Use global-prior spatial support
order for exact ties, not semantic correctness. No column capacity. Noneligible row => USE_NATIVE.
This is the simple cardinality control and has no extra specificity test.

A3: take A2 proposals. For each one require:
D_lg=I_lg/sum_h(I_lh)>=.8 (zero denominator fails), and
F_lg-max_{h!=g}(F_lh)>=.1 (a missing competitor has score0).
For each existing owner g, union all remaining local masks; require
R_g=|union_l(L_l) intersect G_g_visible|/|G_g_visible|>.2.
If R fails, return all members to USE_NATIVE. Recompute unions after specificity filtering.
No requirement that each partial mask independently covers20% of the whole object; its own
forward eligibility remains unchanged. Groups never force two old global owners to merge.

Log evidence before/after the specificity/union gate, action counts, candidate and alias vetoes,
fresh superpoints/owners, planned-vs-realized count owner, and duplicate owner followers. For each
recovered match, distinguish rejection avoided from a genuinely new correct correspondence.

## C7. S1 using only saved binary tracks and winning pixels

For each original completed frame load CropFormer C, all binary SAM masks M_i, and the existing RAW
exclusive raster. Recover W_i, the subset of RAW pixels actually won by track i, from the parent's
RAW remap and the chunk seed-ID order. RAW discovery/replacement groups are not SAM tracks.
Missing correspondence is a technical block, not an excuse to replace W_i with the binary mask.
The frame inputs, model, seeds and5-frame chunking are unchanged and already paid.

Let coverage(i,c)=|M_i intersect C_c|/|C_c| and IoU(i,c)=... . Find best-IoU partners in each direction;
a tied best is ambiguous and not attached. Only mutual unique pairs with coverage>=.8 AND IoU>=.5
are attached. Preserve every C>0 pixel and preserve distinct C groups.

For an attached i->c, add W_i intersect {C==0} to c. For an unattached track with nonempty M_i,
allow a separate object only if |M_i intersect {C>0}|/|M_i|<.2; use W_i intersect {C==0} as its support.
Otherwise add no independent fragment. Overlapping raw masks cause no new winner competition:
W_i are already exclusive. Assign new IDs deterministically; if dense uint16 remapping is needed,
preserve the equivalence classes of all C pixels, not necessarily their original integers.

At chunk first/no-seed frames return C exactly. Do not simply raise the old discovery threshold
from.2 to.8 and then carve multiple residual objects. The new policy intentionally preserves the
whole current candidate and only supplements background. It may preserve a bad CropFormer mask;
that is an empirical limitation, not guaranteed semantic safety.

Record partial-overlap situations (.2<=old SAM-union coverage<.8), protected C pixels, attached
additions, standalone additions, omitted ambiguous tracks and canonical partition differences.

## C8. S2 causal self/other ownership evidence

S2 uses its OWN mapping state before current-frame insertion, the preceding two completed native
snapshots, and the same unmodified cached SAM track history. Never consume the BB00 final map,
RAW final map, GT, or a future frame to define a past self/other mask.

Add an after-frame capture callback to store the just-integrated count-owner membership/snapshot
for use only on later frames. Reuse captured raw SAM bits/depth/pose, not S2-corrected masks, as
track history; no new prompts or SAM feedback. Reset track identity on each original chunk boundary.

A current probe pixel is known-stable when: it passed native depth validation; its superpoint
label L exists in both previous snapshots; both factor0 owner lookups and the current prior owner
are the same positive ID. Alias ambiguity/missing history => unknown. Do not compare arbitrary
raw owner IDs across unrelated maps. Valid depth uses the parent .03+.01*z consistency rule.

Warp the previous binary mask of this same track with the existing visibility warp into the current
frame. Among current known-stable pixels in that warped support, require at least100 and a dominant
owner fraction>=.8. That owner is self. Insufficient/ambiguous support => KEEP_S1. First two completed
frames necessarily lack enough history. A different 2D group ID is not a foreign owner.

For a track with actual S1 additions, define its proposed object support P_i as those additions
plus its attached current CropFormer mask (or just additions when unattached). Restrict known
support to P_i; do not charge conflicts in parts of the raw binary mask that S1 already removed.
Known self = P_i & stable & owner==self; known other = P_i & stable & owner!=self.
If known area<100, KEEP_S1. Else q=known_other/(known_self+known_other).
If q>.2, remove only W_i additions belonging to this track. Leave every original C pixel unchanged.
If q<=.2, retain S1. Unknown region is never negative. Source stability can still be wrong, so no
claim of guaranteed correctness is allowed.

There is no need for a second-best SAM score when an addition is removed: the vacated background
remains zero unless already occupied by an original CropFormer group. Do not fill it from guessed logits.
Record q, self/other counts, history sufficiency, removed additions and actual map effects.

## C9. Map-first screen and real semantic re-extraction

Use the original schedules and main TSDF algorithm for all A/S maps. Mapping may run CPU-only with
deferred features. Export/lock the raw map before accessing diagnostic annotations. Compute the
same class-agnostic recall and surface definitions as the parent. All eligible GT remain in the
denominator; raw unmatched fragments and conditional impurity are separate.

The predeclared catastrophe test applies to the FOUR-scene development aggregate, not a cherry-
picked bad frame. Finish all four geometry maps before deciding that arm's semantic spend. It is
allowed to stop earlier only for technical inability/resource exhaustion, with a different status.
A catastrophically fragmented arm has complete geometry results but unmeasured official semantic
scores. Do not copy a baseline score into those unmeasured cells.

For other arms generate native N, replay Q_GAIN, and generate FC using their own observed masks,
lineages and per-scene frozen temperatures. Q budget is200 attempted requests; do not fabricate
requests to reach200. A truly exhausted candidate universe may have fewer attempts only with a
recorded exact per-frame accounting and a task-local adapter, not a silently changed query policy.
Feature cache reuse requires exact tensors/masks; new owner IDs alone do not invalidate a physically
identical feature if a valid content alias exists. Approximate overlap alone cannot authorize reuse.

## C10. Evaluation, selection and statuses

All full-map primary outputs use the unchanged released scorer and current-class area ranks.
Old source coordinates unchanged => exact nearest projection reuse permitted. New geometry or
vertex order => new nearest projection identity. Source row count is not evaluator target count.

U changes registry: a task-local evaluator registry must cover the output universe rather than
reusing only native-eligible N0 objects. Provide mask files and manifest entries for every positive
export-eligible new owner. Metadata says N unavailable where it is unavailable. Do not add a fake
N source to satisfy the loop. Compare expanded registry length, mask support, actual manifests,
and evaluator prediction counts in one real end-to-end check.

Diagnostic equality has three levels: source probability equality; final owner/class partition
equality; scorer-context/rank equality. Reuse scoring only on the last identity. Do not print a
fixed sentence that two methods are the same without computing the differences.

Selection feasibility and gain thresholds are engineering preferences, not inference guarantees:
feasible means APall>=B-.05pp, AP50>=B-.1pp and mIoU>=B-.1pp. Gain means APall>=B+.2pp with the latter
two guards. Always display all five raw metrics. A result meeting guards with a small negative
secondary delta must not be described as five-metric domination.

Distinct statuses: COMPLETE, EQUIVALENT_PREDICTION, EQUIVALENT_INPUT, SCREENED_OUT_GEOMETRY,
NOT_RUN_RESOURCE_SCREEN, BLOCKED_MISSING_ARTIFACT, FAILED_IMPLEMENTATION, NOT_SELECTED_FOR_TRANSFER.
Separate them from scientific NET_GAIN_WITH_GUARDRAILS / TRADEOFF / NO_NET_GAIN / INCONCLUSIVE and
publication PUSH_VERIFIED / PUSH_FAILED. No arbitrary metric zero for a missing result.

## C11. Reproducibility and low-cost validation

The provided reference kernels cover mathematical intent only. Adapt them to actual source shapes,
class IDs and native API; do not import them blindly as production code. Native off/all-fallback
and explicit compatible-followers require actual native integration tests. Cached U and W require
one real expanded-manifest/evaluator check. Avoid large unrelated safety/regression matrices.

Keep a single compact consumption manifest per job and memoized input validation. Do not repeatedly
read huge hash trees. Preserve timing failures and resumed inputs. Publication must contain actual
new code and compact decisions/scores, with external raw arrays and restoration paths. Verify a
normal push with the full remote SHA. No claim of guaranteed gains or independent generalization.
