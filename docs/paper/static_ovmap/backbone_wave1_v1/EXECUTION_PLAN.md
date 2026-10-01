# Backbone wave-1 implementation and completion ledger

Normative inputs: CODEX_FINAL_EXECUTION_EN.md, PROTOCOL_SPEC.json,
IMPLEMENTATION_CONTRACTS.md and SOURCE_EVIDENCE.md. All study scenes are exposed.
Starting code: 1ce806b22c63943843300006b5b1034ec9e1cb0b. Deployment stays N0_UNCHANGED.

## Task Graph

1. Bind exact historical inputs and four development/eight Replica schedules.
2. Layer the native-v10 source on the isolated pinned upstream, then implement
   read-only prior support, ratio switch, constrained candidates/counts/aliases.
3. Implement native/simultaneous fusion and shared order diagnostics.
4. Implement paired bounded SAM2 RAW/GEOM processing with current-frame-only
   propagation, RAW-defined discovery, past RAW geometry and counted forwards.
5. Implement fresh capture, deferred native metadata/features, per-map anchors,
   frozen Q_GAIN/FC requests, readouts and actual lineage/cache aliases.
6. Build the actual new extension; run targeted tests, one real short native
   trace and one real SAM preflight. Commit implementation before measurements.
7. Complete BB00 scene0056_00 bridge, including all three readouts/evaluator
   parity and deferred-native equivalence; diagnose once if tolerance is exceeded.
8. Measure all seven arms on four complete development schedules; compute raw
   geometric and semantic coverage/intervention diagnostics after locking outputs.
9. Freeze two candidates, run at most one justified composition, freeze nominee
   and transfer set before any Replica result. Commit the freeze evidence.
10. Measure the fixed set on all eight Replica scenes. Report complete official
    pools and within-map native ranks; distinguish incomplete conditions.
11. Generate the four reports and compact measured release from real receipts.
12. Review every normative requirement, commit/push normally, compare full SHAs
    and write the external publication receipt only after equality is verified.

## Implementation Ownership

Primary agent owns design, algorithms, debugging, native integration, scientific
decisions, testing design, final review and publication. Only explicit mechanical
leaf transformations may be delegated to mechanical_worker.

## Exact Phase Interface

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_backbone_wave1.py \
  --phase all --spec docs/paper/static_ovmap/backbone_wave1_v1/PROTOCOL_SPEC.json \
  --output-root /mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001 \
  --gpu 2 --mapping-workers 2 --mapping-threads 8 --evaluation-workers 3 --resume
```

Report/publish phases consume receipts without model/map launches. A nonblocking
coordinator lock and actual per-GPU/output locks protect running work. Interrupted
TSDF maps restart from the beginning with prior attempts preserved.

## Acceptance Evidence

| Requirement | Evidence Required | Final Local Status |
| --- | --- | --- |
| C1 namespaces and exact DAG | resolved inputs, matrix and job identities | verified, 64 exact schedules |
| C2 simultaneous fusion and real order changes | scoped tests and first-three-frame diagnostics | verified; no real first-three-frame order change |
| C3 inherited ratio only | native config/trace with per-arm switch | verified; inherited 0.0 is redundant |
| C4 read-only preinsert probe | loaded binary and touched-state before/after trace | verified in actual native trace |
| C5 actual directional/dummy matching | valid counterexample and real assignments | verified |
| C6 actual mode4 enforcement | candidates/recomputation/count/alias/realized trace | verified; zero full directional-map discrepancies |
| C7 SAM2 current-frame video/raster intervention | actual predictor preflight and reconstructed outputs | verified across all 12 scenes |
| C8 fresh anchors/features/lineage/native paint | each map capture/source DAG and bridge equivalence | verified; Q 12800/12800 successes |
| C9 official pools and raw geometry diagnostics | complete ordered pools and valid GT denominators | verified; 72 pools, 64 raw diagnostics |
| C10 selection/composition/transfer locks | pre-composition and pre-Replica freezes | verified; original nominee preserved |
| C11 true publication and bounded validation | scoped checks and full local/remote SHA equality | local checks verified; publication receipt external |
| C12 physical and standalone costs | measured stage/forward/attribution ledgers | verified with explicit missing timing/call fields |
| C13 bounded raw-logit storage | paired frontends and bitpacked track bundles | verified |
| Production scope | up to 64 actual map jobs, 192 primary rows, exact blocks | complete: 64 maps, 192 primary and 192 secondary rows |
| Four reports and compact release | RESULTS/HANDOFF/SELECTION/CLAIMS with real receipt links | generated and reviewed; 44 release entries verified |

## Input Corrections Discovered Before Measurement

- The declared attempt_001/assets receipt directory is absent. The existing
  SAM2 smoke receipt resolves the real asset root to its parent study assets
  directory. Bind both declared and resolved paths; do not fabricate a download.
- The native configured label_register_min_overlap_ratio is 0.0. Preserve it;
  a structurally inactive ratio control is a measured finding, not permission
  to replace it with a new threshold.
- native-v10 compiled global_segment_map_py.cpp adds raycast_instance_label at
  count-threshold factor 0 to the tracked capture patch. Include this exact
  source correction in the layered patch and retain the original old binary.
- The ratio arm is structurally tagged in the matrix but is explicitly a simple
  control in the mandatory scope. The normative noncontrol structural-champion
  rule therefore permits BB01_SYNC/BB05_FORWARD/BB05_BIDIR only. All seven arms
  remain measured and published. This eligibility correction was made before
  any complete development pool or candidate freeze existed.

## Targeted Validation Budget

At most about twelve new test functions, parameterized where useful. Exercise
native-off parity, simultaneous partition/grouping, directional assignment and
unmatched handling, current-only SAM invocation, raw/geom shared discovery,
fresh geometry/cache/lineage boundaries, official pooling and report behavior.
Compile modified Python and the actual scoped native target. No whole-repository
test run or historical-release-file audit is needed.

## Current Progress

- [x] Read the entire normative package and verify its supplied file manifest.
- [x] Create isolated main and upstream worktrees at their exact pinned commits.
- [x] Preserve the instruction package separately from executable study code.
- [x] Bind actual runtime/model/data assets and write resolved inputs/matrix.
- [x] Implement and validate the real native/Python/SAM/source/evaluator paths.
- [x] Complete the full BB00 scene0056_00 map/source/three-readout bridge.
- [x] Complete 28 screening, four composition and 32 frozen transfer maps.
- [x] Generate four reports and personally review C1-C13 and the compact release.

Publication is verified separately by the ordinary push and complete local/remote
SHA comparison in external `publication/final.json`. The external
`review/final_release_review.json` records the final artifact and publication
checks; a local completion ledger is not proof of remote publication.

### Scoped Implementation Evidence

The loaded isolated native extension passed its real three-frame trace. Native
partition/owner/raycast behavior matched the old extension; a single-thread
diagnostic produced exact TSDF arrays. Eight-thread TSDF differences also occur
in an old-versus-old repeat and are retained as measured parallel variation.
Both object association traces exercised 37,917 actual candidate vetoes with
read-only probes and no realized-owner discrepancies.

The pinned SAM2 video predictor completed a real three-frame GPU preflight.
The optional connected-components CUDA extension is unavailable; the official
predictor fallback is recorded. Twelve new test functions (fourteen cases)
passed. The full fresh-map/source/evaluator bridge passed: every official metric
for NATIVE_READOUT, FC_EQ and D2 differs from historical evidence by at most
0.05 percentage points. The new native pickle is not byte-exact; measured
eight-thread mapping variation is explicitly retained rather than hidden.

The preflight was rerun after the seed-support-key correction, preserving the
previous preflight receipt and its physical cost. Paired frontends cover all 200
scheduled frames of all 12 scenes: 2400 production image encodings and six
validation encodings, shared between RAW and GEOM. The completed study used two
CPU mapping jobs, eight native threads, three evaluator jobs and one neural
worker on GPU 2. GNU time records screening/composition/transfer wall and CPU
time; earlier uninstrumented phase CPU and device-only GPU event time remain
explicitly missing. FC batch-call counts are null, while actual FC image inputs,
region poolings and text inputs are recorded.

Final scientific status is COMPLETE_NO_NET_GAIN. Development champions are
BB01_SYNC and BB03_SAM2_RAW; the only composition is BBX_COMPOSE. The frozen
nominee remains BB01_SYNC. Replica D2 versus fresh BB00 changes APall by
-0.2019653pp and mIoU by +0.3401608pp. Raw R50 changes from 163/389 to 161/389;
surface F5 remains approximately 89.6891 percent. All scenes were exposed and
deployment remains N0_UNCHANGED.

The inherited office commands used a feature IPC wrapper. Twenty initial
launches failed before reconstruction. Transfer-only normalization removed the
unused wrapper while preserving scientific options, original binding and all
failed logs. The actual 32-launch red/green regression passed, and the single
retry completed all 32 Replica configurations with zero failures. No unaffected
map or neural prerequisite was restarted. Original and repaired phase costs are
both retained.
