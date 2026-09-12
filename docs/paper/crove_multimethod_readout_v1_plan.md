# CROVE multimethod readout V1 — execution revision V2

Evidence base: `dba12eeac62fbdaae9c345011b0e5fa570518e07`.
Working branch: `research/crove-multimethod-readout-v1`.
The authoritative task is `CODEX_CROVE_MULTIMETHOD_READOUT_V1_REVISED_V2_ALL_IN_ONE.md` supplied by the user.

## Frozen decisions before new trial scores

The experiment config records selection keys, numerical ties, original Apartment gates,
paired B3-to-H2 transfer, no confirm retuning, and initial method parameters.
DEV is room0 and the existing Apartment pair in both saved states. Static confirmation
is room1; room2 is an asset-failure reserve, not a score-dependent alternative.
All 200 authorized RGB and 200 depth paths, trajectory, and semantic GT mesh exist
for room0/room1/room2 on 2026-09-12. Derived confirmation inputs are not yet bound.
Previous use in Replica8 is disclosed; these are not claimed unseen pretraining scenes.

## Execution dependencies

1. T00: bind exact existing geometry, states, legal views, GT, native features/masks,
   and official M4 checkpoint. Build missing confirmation derivatives once.
2. T01: preserve four legacy evaluation sources and expose actual pointwise predictions
   through the shared metric kernel. Verify unchanged B3/H2 against old input.
3. T02: sparse independent-frame view bank and exact patch/source inverse mapping.
4. T03–T08: full-map native/S0/S2/reencoded baselines and all M1–M4 controls;
   graph patch-only isolation, shared M3 arbitration, matching M4 mean pooling.
5. T09: limited combinations, recovered-H2 attribution, costs and failure analyses.
6. T10: freeze family/task winners, evaluate room1 with necessary direct controls.
7. T11: publish code, real results, reloadable new models if trained, permitted compact
   predictions and handoff; verify remote SHA and readable remote results.

Scientific design, interfaces, implementation semantics and review remain with the
primary agent. Only explicit deterministic mechanical leaves may be delegated.

## Current evidence and limitations

- Clean starting worktree matched the required SHA; this round has its own worktree.
- Both saved `H1_B3/D1_B3` and `H2_INHERIT/D2_INHERIT` NPZ files exist under the
  20260910 entity-epoch run. Reading `current_valid_after` gives 26,958,216 rows
  each, 15,622,601 valid B3 rows and 15,625,540 valid H2 rows: exactly 2,939 restored
  and zero removed. Source-key identity and metric equivalence still need checking.
- Three A40 GPUs were visible, each reporting 45,490 MiB free at preparation time.
- The old named baseline-eval conda environment has no Python executable. Base Python
  imports NumPy, SciPy and pytest; relevant runtime dependencies must be checked.
- Official MaskAdapter source was cloned and checked out at
  `c0516d8a548d90055c3dca7f2a9b4281a4da842f` under the external reference root.
  Its README links `https://huggingface.co/owl10/Mask-Adapter` and the FC-CLIP
  adapter Google Drive object `13_sr30_Q0Geubijik0BpVC_JgyFAmyQU`.
  This does not establish weight availability or trained inference.
  Initial HF API request timed out at 30 seconds and official Drive HEAD request
  timed out at 25 seconds (curl exit 28). Network routing/cache alternatives remain
  to inspect; these two observations do not establish a permanent M4 blocker.
- Base Python passed all four existing `test_two_visit_snapshot_metrics.py` tests.
  The new configuration parses as JSON and `git diff --check` passes.
- Office raw ROS2 database, `gt_changes.csv`, and RGB-D export manifest were each
  checked and remain absent. This does not block static confirmation.

DEV_SCREENING_STATUS=PARTIAL

STATIC_CONFIRMATION_STATUS=PARTIAL

DYNAMIC_CONFIRMATION_STATUS=NOT_RUN_RAW_MISSING

No new family metric or completion claim is made at this preparation checkpoint.
