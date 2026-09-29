# A7 wave-1 handoff

External run: `/mnt/shared/ww/ovimap-a7-evidence-upgrade-wave1/attempt_001`. Binding: `3036348dd8e035215c1d1543a7ef0f6f9f300e79981c8b8c887321108800b998`.

Measured: E01 RAW_EQ/UNIT_EQ/GMED, E02 GLOBAL/SAM2, C0 SO400M, E03 FC_FROZEN_REGION/OVR, E04 SHORTLIST, five controls and one frozen composition. SAM3 SPATIAL/MIX50 are blocked by actual official gated access (401); no substitute weights, zero metrics or fabricated predictions are supplied.

FC_FROZEN_REGION is an original frozen OpenCLIP backbone with matched region operators, not a full FC-CLIP benchmark reproduction. OVR loads the learned author checkpoint backbone strictly; pinned upstream operators are extracted, but the full Detectron wrapper was not executed.

## Assets and provenance

- fc_frozen: revision/code `654d0f80ff73c58e7281a3ca7dc425589049e2e1`, verified bytes 1410764773; full checkpoint SHA256 and official source in `assets/fc_frozen/download_receipt.json`.

- ovrcoat: revision/code `9fd9450d22852d269d426b521663a127f3983a4b`, verified bytes 4633915328; full checkpoint SHA256 and official source in `assets/ovrcoat/download_receipt.json`.

- sam2: revision/code `2b90b9f5ceec907a1c18123530e92e794ad901a4`, verified bytes 898083611; full checkpoint SHA256 and official source in `assets/sam2/download_receipt.json`.

- so400m: revision/code `f3b7a187cd133857ff43c0dccfe88f8268372549`, verified bytes 4582795867; full checkpoint SHA256 and official source in `assets/so400m/download_receipt.json`.


Each FC/OVR recognition model has 351,772,609 parameters (199,770,816 visual). Both branches were strictly loaded with 543 keys; these counts exclude unused segmentation heads and all shared native-map components. SO400M has 1,136,008,498 parameters and SAM2 has 224,446,642, recorded in real worker receipts. All visual models are frozen; CAL only fits scalar temperatures.


SO400M: Apache-2.0, documented WebLI training. Frozen ConvNeXt OpenCLIP: MIT model card, LAION-2B English pretraining. Native SigLIP and S2 use documented WebLI training. SAM2 code/checkpoints are Apache-2.0 and document SA-V; an exhaustive training-overlap audit is unavailable. OVR code is Apache-2.0; its inherited configuration documents COCO panoptic training and the backbone originates from LAION. The complete author-checkpoint training history and overlap with ScanNet/Replica are unresolved, not assumed clean. Pinned model cards and code licenses provide the supporting source records.

## Environment and restore

Actual Python executables, Torch/CUDA versions and package snapshots are in environments/{semantic,region,sam2}.json and workers.json. No installed environment was upgraded. Restore the exact model files and historical binding inputs on shared storage before resuming. External paths and SHA manifests describe bytes not uploaded to GitHub. The source map and original principal JSON are unchanged.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_a7_evidence_upgrade.py --phase all --split all --resume --gpus 0,1,2
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_a7_evidence_upgrade.py --phase report --resume
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_a7_evidence_upgrade.py --phase publish --resume
```

These commands run from the research worktree. Worker receipts validate fixed model/input/code identities before reuse; a changed scientific input requires a separate attempt. Full arrays, weights and RGB-D remain external; compact metrics, scalars, decisions, contracts and diagnostics are published.

## Limits

See RESULTS for complete official/frozen rankings, pooling versus scene means, intervention/fallbacks, matched controls, composition interaction and timing omissions. SAM3 remains an asset-access block. Wave1 does not execute E05–E12 or geometry reconstruction changes. Publication is verified only by an external receipt matching full local HEAD and the remote branch SHA.
