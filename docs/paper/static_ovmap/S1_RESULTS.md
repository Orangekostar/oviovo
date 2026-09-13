# 0913_1 — first measured fixed-K semantic selection result

Scope: Room0 development, immutable native geometry/instance partition, native original
SigLIP image features and Replica51 text vocabulary. This is not Replica8 or final completion.
All selection uses the retained native top-10 pool, not yet the complete executed-query history.

| Condition | Semantic mIoU | mAcc | Released semantic APall | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| B0: native stored last8 + area | 0.332857 | 0.381587 | 0.154316 | 0.343233 | 0.368233 |
| RANDOM8 seed 0 + area | 0.292899 | 0.347837 | 0.155087 | 0.343233 | 0.359900 |
| QUALITY8 + area | 0.332857 | 0.381587 | 0.154316 | 0.343233 | 0.368233 |
| S1a: quality/coverage8 + area | 0.362175 | 0.416451 | 0.186723 | 0.384900 | 0.409900 |
| S1b: native last8 + quality weights | 0.332857 | 0.381587 | 0.154316 | 0.343233 | 0.368233 |
| S1c: quality/coverage8 + quality weights | 0.362175 | 0.416451 | 0.186723 | 0.384900 | 0.409900 |
| ALL_VIEWS: all retained records + area | 0.362175 | 0.416451 | 0.186723 | 0.384900 | 0.409900 |

S1a changes 12 observation subsets. Two object labels change: instance 66 vase→indoor-plant
and instance 71 pipe→shelf. Margins and selected source-query IDs are saved for every change.
S1a's gain is selection-related; this experiment gives no evidence for an additional
quality-weighting gain. ALL_VIEWS has a different readout budget and is a separate control.
No extra image/VLM query, encoder replacement, instance filter change or geometry change occurred.

## Evidence and costs

- Independent reconstruction and strict squared-distance `< 0.05**2` 1NN reproduce all
  954,492 original projected semantic and instance labels exactly (116,029 unmatched vertices).
- Original Torch matching matches all 71 eligible object classes. Maximum fused-feature
  difference versus NumPy accumulation is 4.82e-8; output classes and projected labels are exact.
- Released B0 mask files, class/score text manifest and GT instance conversion match the original.
  The original evaluator's semantic-instance vocabulary has 48 classes; text/vertex vocabulary
  has 51. Both ID lists are recorded, rather than silently treating them as the same list.
- Released class-agnostic outputs are identical for every condition: instance mIoU 0.4589,
  mP@.25/.50/.75 = 0.8413/0.7619/0.4286, mR@.25/.50/.75 = 0.5761/0.5217/0.2935.
  These rounded released precision/recall values are not integrated class-agnostic AP.
- Enrichment prepared all 592 retained observations with no missing RGB-D frames, taking
  11.96 s including the native mesh centroid pass and relevant input hashing. 69 native-color
  objects have centroids. Quality measures the legacy context bbox, not an unavailable source
  union mask. Mask quality and independent geometry support remain null.
- Warm CPU readout is approximately 0.08–0.10 s per condition. Frontend and image-VLM historical
  cost are null; these numbers are not end-to-end throughput. Projection/AP audits are diagnostics.
- Room0's 200 RGB, depth and input mask files are bound by path/size/hash. Cached poses agree
  with original trajectory within 2.27e-7. All Replica8 RGB-D and sampled pose inputs exist.

## Reproduction and remaining scope

Source and small results are under src/static_ovmap, scripts/evaluation, and
artifacts/static_ovmap/room0_enriched. Large masks, semantic arrays and feature caches remain
under /home/ww/oviovo_baseline_runs/20260913_static_ovmap/room0. Full argument vectors are in
the `command` fields of each input/evaluation/audit receipt. Run text, original arithmetic,
projection and instance evaluation using /home/ww/miniconda3/envs/ovimap-map/bin/python;
enrichment/readout can use the default python. Readout now requires --native-binding pointing
to artifacts/static_ovmap/room0_native_binding.json to prevent assigning a different text space
to the same native image cache. Its historical identity reconstruction limitation is explicit.

41 targeted tests pass. Final audit remains open: full-query history ownership, R3 fallback,
R4 merge/split, R5 dense/AnyUp/refinement, R6 T3D, Replica8/ScanNet18 and final tables are pending.
New finding: native `cropformer_inst/temp_feats` contains 764 cached query features, matching
the mapping log. The 172 discarded records lack saved instance IDs and visible areas in their
filenames; priority is recovering original ownership/area metadata without encoder inference.
Do not infer these quantities from GT or silently promote retained-top10 results to full-history.

The native C++ map configuration sets voxel_size=0.01 in global_segment_map_py.cpp:60;
all 9,282,303 native vertices have at least two coordinates on the 1 cm voxel-center lattice
within 2 micrometres; none fit the 2 cm alternative. The historical binary's exact source/build
provenance still needs closure. ScanNet sampling
must be bound to real sequences: source uses floor((end-start)/200) for negative step, which
must not be replaced with Replica stride 10. The earlier PROGRESS.md is the initial checkpoint;
this document and current receipts supersede its pending Room0 arithmetic/projection/AP entries.
