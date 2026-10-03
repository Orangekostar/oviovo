# Technical amendment 03: per-arm cold preflight

The fixed timing matrix still contains eight Replica scenes and three FC arms,
for 24 required leaves. Before any measurement reservation or GPU model load,
the controller verifies the complete scene/arm parent plan against the actual
warm receipts, candidate registries and selected request outcomes.

Office1 has an actual complete independent U2 parent. Its G1 and G3 parents are
the unchanged all-failed semantic receipts described in amendment 02. These two
leaves receive `BLOCKED_UNMEASURED` receipts: null time and parity, zero reserved
physical calls, and no executed cold call. They retain the exact failed selected
requests, masks, frame IDs, source hashes and technical reason. A missing parent
or an unproved all-failed source cannot be treated as scientific zero recovery.

The remaining 22 leaves use the original shared production `recover_fc` callable
exactly once each. It receives a resident model/text session and `cache=None`;
G arms rebuild their full geometry/view work and U2 performs its archive lookup.
The original whole-call CUDA timer, model-load exclusion, output export and
feature/support/label/rank parity checks retain their definitions. The original
reservation rule continues to forbid replay of an observed or uncertain call.

Aggregation requires all 24 fixed receipt positions. An arm mean is available
only from eight complete measurements, with denominator eight. U2 can therefore
have its full-cohort mean; G1/G3 retain null means while office1 is blocked.
Their seven measured costs remain explicitly partial evidence, never a
seven-scene substitute benchmark. The no-recovery reference alone is zero by
definition; genuine no-view U2 leaves still run and contribute measured overhead.

This revision changes `timing.py`, task orchestration and focused tests before
any real cold reservation. All 168 completed predictions/scoring records, ten
whole-cohort pools, their scientific producer sources and every office1 failure
remain unchanged. The required scientific totals remain 172/14/24. Main outcomes
have already been seen; this is a technical amendment with no untouched claim,
new parameter fitting, changed selection, geometry, model, precision or class
vocabulary. A3 remains the fixed primary method and deployment is unchanged.

Validation covers refusal of reserved-call replacement, exact 24-leaf partial
accounting, refusal of fabricated zero or GT-dependent blocks, and a controller
fixture that executes 22 distinct calls with one model load and then reuses all
24 receipts without a second call. Synthetic timing values are temporary test
inputs and cannot enter the actual result store. Current-source final validation
and the unchanged eight-condition existing-map smoke are required before the
amendment freeze; actual cold execution and its primary audit follow that freeze.
