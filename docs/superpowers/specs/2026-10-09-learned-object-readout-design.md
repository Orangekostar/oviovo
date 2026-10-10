# Learned object readout design

The user authorizes implementation, data materialization using existing ScanNet access, fixed training, evaluation and normal publication without further questions. The immutable execution pack in `docs/paper/static_ovmap/learned_object_readout_v1/` is the specification. Its numerical JSON is copied byte-for-byte to `configs/static_ovmap/learned_object_readout_v1.json`.

## Boundaries

Use the exact base `30f59c1fa3d783054dc53fc6693e1fd68e8c2a6d` and branch `research/ovimap-learned-object-readout-v1`. The full source-update parent supplies all 26 maps. A new adapter validates its actual publication and baseline rows; old binders and parents remain unchanged. Neural input namespaces contain predicted geometry, sources, calibrated frames and frozen text only. Evaluation projection/labels remain in the output/evaluation path.

The new data pipeline inventories only declared roots and documented immediate children. Official TRAIN families exclude CF18 and every discovered prior-project family, including the 14 families recorded in the authorized 2026-09-22 acquisition. Fixed SHA256 family order supplies 24 TRAIN, 4 DEV and 4 H families. Materialization uses the existing verified downloader constants and prior authorization receipt. H annotation parsing and feature generation require both seed nominations. Category hashing yields 160 positive-supervised base IDs and 40 adapter-heldout IDs.

New RGB-D is registered using sensor extrinsics, actual depth scale and unaligned geometry. Full-mesh ray/depth tests construct the declared four proposal conditions and common view prefixes. Regression uses predicted owners and inherited 32-frame captures. Physical-site IDs are shared across views; unknown tokens stay missing. Frozen image extraction writes ordinary FP32 CPU tensors. The frozen visual projection remains differentiable with respect to learned raw pooling values.

Port the pinned Mask-Adapter numerical head and ConvNeXt arithmetic with their notices. MA, VIEW, SURFACE and AUX branches share features, data draws, initialization and losses except the declared grouping/auxiliary terms. Execute exactly 10,000 main and 6,000 repeat optimizer updates; the 20-update engineering fit is discarded. Select only on equal-family DEV base loss before H and regression. Checkpoints preserve optimizer, RNG and draw position.

All seven nonparent methods replace historical F using inherited source temperatures and D2 grouping. G1 geometry, recovered labels and ineligible owners remain fixed; current-class official ranks are recomputed. Lock complete predictions before evaluation. Report H recognition separately from 234+52 whole-map rows and 18+4 ordered pools. Runtime is recorded and never selects a method.

## Implementation layout

`common.py` owns fixed config, receipts and identities; `binding.py` owns read-only lineage adapters. `inventory.py`, `sensor.py` and `data.py` own root discovery, normalized camera export and supervised proposals. `observations.py` and `features.py` own shared views/sites and immutable FC image shards. `mask_adapter.py`, `model.py`, `losses.py` and `training.py` own exact numerical heads and optimizer execution. `prediction.py`, `evaluation.py` and `selection.py` own whole outputs, official pooling and accuracy gates. `reporting.py` and `runner.py` own three table families, four reports, stage order and CLI status.

The task root is `/mnt/shared/ww/ovimap-learned-object-readout-v1/attempt_001`, backed by the capacity store. The root filesystem has insufficient cache space and is not used for new raw/dense storage. One A40 and the inherited GPU lock; data workers <=4 and evaluation workers <=3.

## Acceptance

Run the 12 supplied CPU references and bounded production checks for the 11 required groups. Run real upstream-port and frozen-FC gradient parity, two-TRAIN-object engineering fit, all fixed scientific stages and the terminal `--phase all --resume`. Publish real compact results and small trainable checkpoints, exclude licensed scans and frozen/dense assets, then prove full local/remote SHA equality in an external receipt. Missing dependencies return nonzero with precise stage statuses; they never produce fake trained heads or metrics. Completion requires a requirement-by-requirement review against all original contracts.
