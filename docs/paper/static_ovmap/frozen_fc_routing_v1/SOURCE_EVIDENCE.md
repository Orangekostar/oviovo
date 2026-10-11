# Source evidence and limits

Inspected on2026-10-11. All repository links pinned to `76db115aa7df652a0b2dc4575df52e6a74a63549` unless an endpoint is explicitly a live ref. Blob SHA is Git content identity, not a SHA256 of a server file. Connector citation references refer to the audit conversation; permanent links make this file usable outside the chat.

No server training/inventory or full raw-scene evaluation was performed while writing the pack. Source inspection is not a proof that the new mechanism will improve accuracy.

## E01 — remote ref
- Source: https://api.github.com/repos/Orangekostar/oviovo/git/ref/heads/research/ovimap-preservation-routing-v1
- Inspected evidence: Remote ref ->76db115... at authoring; local/server worktree not inspected.
- Ref endpoint, no file blob.
- Audit conversation: `turn1015file0`, content at L2.

## E02 — whole report
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/docs/paper/static_ovmap/PRESERVATION_ROUTING_RESULTS.md
- Inspected evidence: R failed the foundation gate; G/H2/new maps were not measured.
- Git blob: `2fe2ba3866bdae46cba35450d247bd5a05e1f5fb`
- Audit conversation: `turn1024file0`, content at L2.

## E03 — teacher,RHead,GHead,absolute_route
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/models.py
- Inspected evidence: Ordinary FC helper differs from MA RHead; G has768x768 transform; quality/membership/null route reusable.
- Git blob: `67e76b312d0655f39405bafbb65e2ba19c4500af`
- Audit conversation: `turn1016file0`, content at L2.

## E04 — train_G
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/geometry_training.py
- Inspected evidence: Both R nominations must qualify in the old entry. Do not invoke for FG.
- Git blob: `b53ed27d4b76f3e4ec028697ecc5799dacd4ff1c`
- Audit conversation: `turn1019file0`, content at L2.

## E05 — bind,load_scene
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/binding.py
- Inspected evidence: PR->LR lineage,286/22 LR check, baseline aliases, old-checkout gate; new binder required.
- Git blob: `1001f207bf96a70d701421af7f9493c7b3b91b02`
- Audit conversation: `turn1018file0`, content at L2.

## E06 — ObjectLoader,prepare,lock_H2_inventory
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/data.py
- Inspected evidence: Loader adds fc_vectors, TRAIN622/DEV69/74 classes, fixed H2/real frames.
- Git blob: `59b1ec41a4a2883bebb98ceec721682ea35f7db1`
- Audit conversation: `turn1023file0`, content at L2.

## E07 — FrozenFC.__init__,phi,capture (read1-118)
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/learned_object_readout/features.py
- Inspected evidence: Exact FP32 frozen FC and nonlinear vector phi; weights frozen, input derivatives remain available.
- Git blob: `8282d8ed2249680a65d9fa1e14b3b1e4dec0d4f2`
- Audit conversation: `turn1022file0`, content at L2.

## E08 — auxiliary,item_loss
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/losses.py
- Inspected evidence: Base/teacher/membership/correspondence terms separately controlled.
- Git blob: `3a1c564ad1606d659f7c13798180c601806c08e6`
- Audit conversation: `turn1020file0`, content at L2.

## E09 — summarize,qualifies,choose
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/selection.py
- Inspected evidence: Exact A/M/C,full200 competition and .005 DEV gate.
- Git blob: `4ca15d22701314c3d4e03f6a560bfc0f3fd896c9`
- Audit conversation: `turn1021file0`, content at L2.

## E10 — require_transfer,run
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/holdout2.py
- Inspected evidence: R-specific transfer guard; fixed authorized H2 generator and384-image cap.
- Git blob: `26b091c99ec238784d2b15f58d1ec0de07422259`
- Audit conversation: `turn1026file0`, content at L2.

## E11 — run (read1-160)
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/real_proposals.py
- Inspected evidence: GT-free SAM2 generation/lifting; freeze before matching; targets>.25,high>.5; R guard must be adapted.
- Git blob: `efc0c64daac09557ebd2b9afa9d7e69c5fbdbbfd`
- Audit conversation: `turn1029file0`, content at L2.

## E12 — check
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/integration.py
- Inspected evidence: Real FC8 vector comparison, literal unchanged G1 payload and two-scene scoring.
- Git blob: `df5f243932b8d229aca4c009486d79f94d976b19`
- Audit conversation: `turn1025file0`, content at L2.

## E13 — partition_evaluator,required_coverage,evaluate_scene,run (read1-145)
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/evaluation.py
- Inspected evidence: Exact full registry/ranks/scorer, but15-method390/30 orchestration must not be reused unchanged.
- Git blob: `cb1400f7433bdd5aa43d5338c6462f754d3ac93f`
- Audit conversation: `turn1030file0`, content at L2.

## E14 — gpu_stage,main,foundation_ready (read1-180)
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/runner.py
- Inspected evidence: Separate environments and queued lock; old negative path would skip FG if reused unchanged.
- Git blob: `96cc2993afc09c3b8a0213ba3d2bd9d9a9362c5d`
- Audit conversation: `turn1027file0`, content at L2.

## E15 — lock,diagnose,gpu_dispatch
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/src/static_ovmap/preservation_routing/transfer.py
- Inspected evidence: Post-lock oldH diagnostics and selected heads; R/map-count assumptions require new orchestration.
- Git blob: `3383dc9e8eea418d34c830ebcaf52211af4d30b2`
- Audit conversation: `turn1028file0`, content at L2.

## E16 — whole report
- Source: https://github.com/Orangekostar/oviovo/blob/76db115aa7df652a0b2dc4575df52e6a74a63549/docs/paper/static_ovmap/PRESERVATION_ROUTING_HANDOFF.md
- Inspected evidence: Logical/capacity roots, FC/controller env, CUDA works despite NVML issue; provenance limits.
- Git blob: `b29f38a00f690c0c80250700f7cdc6798a29ee7f`
- Audit conversation: `turn1017file0`, content at L2.

## Literature boundaries (primary sources)

L1. Li et al., Mask-Adapter, CVPR2025. https://openaccess.thecvf.com/content/CVPR2025/html/Li_Mask-Adapter_The_Devil_is_in_the_Masks_for_Open-Vocabulary_Segmentation_CVPR_2025_paper.html
The paper studies learned semantic activation maps and mask consistency for open-vocabulary classification. Ordinary learned region pooling is prior work, not a new FG contribution. This task uses ordinary FC, not retrained Mask-Adapter.

L2. Deng et al., OVI-MAP, CVPR2026. https://openaccess.thecvf.com/content/CVPR2026/html/Deng_OVI-MAP_Open-Vocabulary_Instance-Semantic_Mapping_CVPR_2026_paper.html
OVI-MAP separates instance reconstruction from selected-view semantic extraction. FG is a supervised semantic-readout experiment on retained predicted geometry; it does not inherit a new online, geometry, or end-to-end speed claim.

## Deductions and choices, not prior measured facts

- Removing the free value matrix is a control-design choice. We have not established that this matrix caused the previous negative result.
- Separate quality/membership with a null route enables absolute rejection mathematically; it is not a guaranteed recognition improvement.
- Equal alpha mass need not give equal delta norm; the additional norm-matched diagnostic addresses this confound.
- The DEV0.5pp gate, eta ramp, group sizes, loss coefficients and seeds are frozen study choices rather than empirically optimal settings.
- The old H2 names were prelocked; whether other runs have since consumed them requires bounded lineage verification. Never automatically claim untouchedness.
