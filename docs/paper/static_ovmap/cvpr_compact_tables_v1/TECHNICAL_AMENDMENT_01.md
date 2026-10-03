# Technical amendment 01: current segment alias membership

The initial implementation freeze is
`1373d7a5eea02028774acea865fda9177165d661`. Its complete record is retained
in the freeze directory. The main controller stopped during the Native/Q leaf
for `scene0378_02`; no main compact predictions, pooled metrics or cold timing
measurements had been produced.

## Cause and correction

At captured frame 315 (completed-frame index 35), segment 54 was retired by the
recorded merge `54 -> 51`. The native snapshot still reported an old positive
owner for segment 54, while current segment 51 had owner zero. The unchanged
causal lineage guard consequently rejected the mixed `{0, 5}` ancestry of a
current owner-5 request.

The pinned native C++ `mergeLabelConfidence` calls `swapLabels(old, new)`;
`SemanticInstanceLabelFusion::mergeLabels` can retain retired statistics when
the surviving label lacks a corresponding table entry. `exportStudyFrameState`
includes those statistics and the actual merge aliases. A retired label's
remaining statistics do not establish a live geometric segment.

The new task-local `query_bridge.py` interprets every recorded segment through
the aliases available at the current frame. It preserves all membership rows
and assigns retired rows their current resolved segment's owner. It rejects
cycles, missing resolved membership and malformed snapshots. The original
positive-ancestry guard, causal replay, Q_GAIN policy, checkpoint, scaler,
budget, encoder, prompts and Native acquisition remain in use. It does not read
final owners, future frames, GT or scores to resolve current membership.

The bridge retains the original frozen `base_sources.py` bytes, dispatches the
new Native/Q leaf through its own module, and uses a distinct log basename to
preserve the original failed command and log. Existing FC execution is unchanged.
The original AST and each narrowly adapted AST are recorded in successful receipts.

## Actual impact evidence

The primary inspected all eighteen captured streams and verified every recorded
request against current canonical membership. The completed sixteen streams
have identical original and corrected request ancestor sets. `scene0378_02`
has 49 corrected request ancestries, all with complete positive live ancestry;
`scene0518_00` has no changed request ancestry. The captured files remain immutable.

Evidence in the task's shared validation directory:

- `query_alias_root_cause_audit.json`, identity
  `0d658416b9e7b2fa88b7892afc8f7a3cdba1fdf7547ac09eea9a6bf5fc2eb591`;
- `query_alias_bridge_validation.json`, identity
  `b98cefb52900d8d86a3393ff76ffb77558cdd3b956628e8d09e286afb73a6de6`.

The failed worker incurred 6,930 image inputs in 1,155 encoder batch calls and
321.25317117595114 seconds of worker wall time, including its completed Native
work and partial Q work. Its receipt, exact command and log are retained; these
payments enter the failure ledger. Identical successfully encoded tensors may be
reused under their existing model/tensor identities when this leaf resumes.

## Exposure and descendants

`main_outcomes_seen` is conservatively true because main source readouts already
exist. The primary has not observed new compact main metrics. This correction is
a documented technical amendment, not an untouched confirmation or result-based
method selection. A3, numerical constants, all cohorts and deployment
`N0_UNCHANGED` remain fixed.

Only the failed `scene0378_02` Native/Q leaf and its not-yet-complete semantic
descendants require recovery. The remaining `scene0518_00` stream uses the same
corrected interpretation prospectively. The sixteen complete source readouts,
twenty-four complete projected-view receipts and all twenty-six common maps are
preserved. No cold measurement has been reserved or replayed.

Validation includes regression coverage of the actual stale-alias failure,
persistent aliases, genuinely missing positive ancestry, cycles and missing
membership, original inference/replay preservation, worker dispatch/log retention,
and freeze gates with isolated test repositories. The final focused validation
and the same existing eight-condition development smoke are refreshed before
the amendment is committed.
