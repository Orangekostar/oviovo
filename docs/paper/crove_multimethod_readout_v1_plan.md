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

## T01 bridge evidence (2026-09-12)

Implementation commit: `88188f2` (the execution preceded commit; only automatic
formatting changed after execution started, with no metric or prediction changes).

`python scripts/evaluation/verify_crove_readout_bridge.py` completed both saved
states. Predictions were serialized before target loading and reloaded for scoring.
The common metric kernel is shared by old snapshots and new pointwise inputs.
The sidecar preserves canonical indices plus the legacy evaluation permutation,
so nearest-neighbour ties retain the original entity/source ordering.

| State | Current rows | Uncovered legacy rows | mIoU | Ghost | BG F5 | Surface F5 | Maximum metric difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B3 | 15,622,601 | 0 | 0.1358861593 | 0 | 0.3663600098 | 0.4313354117 | 0 |
| H2 | 15,625,540 | 0 | 0.1357344795 | 0 | 0.3663600098 | 0.4357097517 | 0 |

Both states preserve all 62,498 unknown-entity rows in the object evaluation branch.
The small test independently verifies supported role rewrites, owner independence,
and a changed predicted label changing actual mIoU. Twenty focused tests pass,
including the existing entity-epoch runner regression; Ruff and compilation pass.
These are baseline bridge results, not M1–M4 improvement trials.

## M4 asset resolution

Official HF access succeeded through the existing local proxy. The 1,669,238,739-byte
FC-CLIP + Mask-Adapter checkpoint matches its official LFS SHA-256; it contains all
543 backbone-prefixed and 43 adapter-prefixed tensor entries. A dedicated venv now
provides OpenCLIP 2.24.0, timm 0.9.16 and fvcore; existing CUDA torch is available.
Strict loading of all backbone entries into OpenCLIP succeeded without missing or
unexpected keys. Adapter module load and real-mask reference comparison remain.
See `model_manifest.json`; downloading weights is not recorded as a completed M4 trial.

## M4 external-mask inference checkpoint

The wrapper strictly loads the 543 encoder keys and 43 Adapter keys. It uses the
official RGB mean/std, longest-side 896 resizing and normalized padding to 32.
`F_raw` is 1536 channels and `F_head` is 768; learned activations pool `F_raw`,
then each map passes through the region projection before averaging. The ordinary
mask mean uses that same `F_raw` and region projection. Empty masks are skipped,
with explicit returned mask indices. Neither decoder labels nor void/ensemble
scores enter the interface. Text uses one fixed prompt per class for both rows.

Three real cached OVI geometric masks from authorized room0 frame 0 passed a
reference comparison against expressions loaded from the pinned official code.
These masks are a numerical-validation input, not independent instance evidence
for M3 and not a full-map M4 trial. Source-extracted head forward, both visual
projections and per-map pooling agree exactly for the same batch. Splitting mask
batches gives maximum logit error 0.0003977 and passes rtol=1e-4, atol=1e-5.
Peak allocated GPU memory was 1,758,827,520 bytes. Full result is in
`adapter_reference_check.json`.

Executed command (from this worktree):

```bash
$HOME/oviovo_baseline_builds/maskadapter/venv/bin/python \
  scripts/evaluation/verify_mask_adapter_reference.py \
  --source "$HOME/oviovo_references/modules/MaskAdapter" \
  --checkpoint "$HOME/oviovo_baseline_builds/maskadapter/fcclip_convnext_large_maskadapter.pth" \
  --rgb "$HOME/vv/dataset/Replica/room0/results/frame000000.jpg" \
  --regions "$HOME/vv/paper2/OVI-MAP/output/benchmark_20260615_ovimap/geo_seg_temp/room0/00000_mask.png" \
  --output configs/evaluation/results/crove_multimethod_readout_v1/adapter_reference_check.json
```

The output command refuses to overwrite an existing receipt. Official classes
retain attribution and Apache-2.0 license under `docs/licenses/`.

Next: prepare the shared full-surface observation bank and run complete M1/M4
readouts. The room0 native pickle contains 77 owner entries with actual `feat`,
`frame_id`, `pose`, `vis_area`, and `box_2d`; the first owner contains 10 independent
cached observations. This permits native-feature reuse but does not recover
individual crop-scale features from their six-crop average. Room1 derived OVI/S2
inputs are still to be generated. All three completion fields above remain unchanged.

### Subsequent full-map execution

Room0 trained adapter mean/learned, S0/S2, S2 patch/geometry graph, cached
diverse/quality M1 and M1-posterior patch/geometry graph are now evaluated.
See the results document for exact metrics, costs, negative results and limits.
No family winner is frozen across protocols. Static CURRENT_AWARE reuses QUALITY
because the source contains one visit; dynamic state behavior is still pending.

Native room1 GPU preparation failed with CUDA OOM while other processes occupied
the devices. The same official full-resolution frontend is now executing on CPU
for room1 and room0; first-frame outputs were verified. Commands in progress:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/prepare_crove_native_inputs.py --scene room1 --frontend-device cpu --map --output /home/ww/oviovo_baseline_runs/20260912_crove_multimethod_readout_v1/confirm/room1/native_cpu
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/evaluation/prepare_crove_native_inputs.py --scene room0 --frontend-device cpu --output /home/ww/oviovo_baseline_runs/20260912_crove_multimethod_readout_v1/dev/room0/cropformer_cpu
```

Do not launch duplicates while these handles remain live. Room0 output supplies
independent segmentation evidence for M3/M2 boundaries; owner-projection masks
are not substituted for independent segmentation. The remaining work includes
boundary graphs, M3 whole-map readouts, paired dynamic experiments, structural
background/per-crop evidence, confirmation, selection and final handoff.
