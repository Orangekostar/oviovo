# R5 dense feature branch design and execution plan

Scope: the complete R5 requirement in the original handoff. Use one released dense
method, GLA-CLIP ProxyCLIP, with its published OpenAI CLIP-B/16 and DINO-B/8 weights.
No training. SPAR/VIP are screened alternatives, not extra implemented backbones.
The original native SigLIP B0 remains separately reported. User authorized autonomous
configuration; primary agent owns this model bridge, numerical behavior and review.

## Identity and preprocessing

Pin GLA-CLIP 63d674626cb1ddc566e3177b9cb5b2dc7a212158, DINO
7c446df5b9f45747937fb0d72314eb9f7b66930a and AnyUp
351807a9c4287368732cc247f26c7c81c9139af4. Record checkpoint hashes and imports. Isolated
venv inherits read-only Python 3.11 / Torch 2.4 from oviovo-radseg; additions remain in
the new venv. The original native mapper environment and all source repositories stay
unchanged. GLA's classifier-free forward_feature is loaded without the MMSeg segmentor
constructor or vocabulary. Transient DINO QKV hooks from that method must be removed
after each call to avoid accumulating references; feature arithmetic stays unchanged.

Dense RGB input: short edge 336, aspect ratio retained to nearest 8-pixel grid dimension,
PIL bilinear resize, CLIP normalization, 224-square windows with stride 112. Use published
KV extension, proxy_sim, token_norm, mini_iters=2, initial_crit_pos=.6, dynamic_beta=.3,
dynamic_gamma=30 and temperature=1. Stitch normalized classification-preceding window
tokens into the shared DINO stride-8 feature grid. This is a declared mapping adapter,
not a claim to reproduce MMSeg's full segmentation pipeline or its dataset scores.

AnyUp: official anyup_paper.pth, use_natten=False, full raw RGB ImageNet normalization,
output_size=(680,1200), q_chunk_size=4096. Bilinear and AnyUp consume the SAME low-resolution
feature map and produce the SAME output shape. Record actual peak GPU allocation and
latency. No interpolation placeholder is allowed in the AnyUp condition.

## Paired conditions and memory

- B0: immutable original SigLIP ROI cache, native last8/min-two output.
- D0_ROI: additional encoder control using the original CLIP-B/16 global image encoder
  on each selected native query's unmasked context bbox; exclusive native slicing and
  standard CLIP resize/center-crop. It is not falsely called the native six-crop mask
  encoder, since original union-mask history is unavailable.
- D1: GLA frame features, bilinear upsample, native depth-consistent owner-mask pool.
- D2: identical frame features/pooling, official AnyUp replacing bilinear.
- D3_L01/L02/L03: D2 plus one sparse RGB-D neighborhood refinement, temperature 1 and
  lambda .1/.2/.3. Edges join only visible same positive owner with consistent depth;
  owner 0 never propagates. Use a streaming sparse/neighbor implementation, no N-by-N
  production array. Pixel nodes represent measured RGB-D surface samples; scope is
  within-frame geometry refinement, not claimed cross-frame graph propagation.

Use exactly native B0-selected query slots for eligible owners, at most K=8 per native
owner. A frame feature is shared across all selected owners in that frame. Missing
projected support is recorded rather than replaced by features from another encoder.
Keep the B0 semantic eligibility gate; additional single-query fallback stays off.
Each new space uses its own CLIP text encoder and fixed ImageNet template ensemble for
the same explicit Replica51 labels. Features from SigLIP and CLIP are never averaged.
Text matching is separate from feature extraction. T=1, pooled cosine scoring; local
prototypes are retained for the representation/storage contract, not silently added to
the classification rule. Retain K observations and four spatial local prototypes per
object; do not store full-resolution feature images for every frame.

## Execution checks

- [ ] Bind released assets and source interfaces; strict state_dict load, finite real
  single-frame features, original-backend feature parity and constant hook count.
- [ ] Test aligned window coverage, explicit label/space checks, owner/depth/visibility
  edge exclusion, unchanged unknown nodes and sparse-vs-small dense reference output.
- [ ] Implement dense_features.py, feature_refine.py, query_scores.py and model bridge.
- [ ] Run real same-slot Room0 D0/D1/D2/D3, retaining observation/prototype payload sizes,
  encoder calls, missing support, peak memory and stage timings. Inspect negative results.
- [ ] Evaluate on unchanged B0 native geometry with original mask/score exports and
  independently verified projection. Choose lambda only at the Room0 development gate.
- [ ] Review all source requirements, actual arrays/results and relevant tests; archive
  receipts, commit/push and verify remote SHA. R6/full-scene/final A–I scope remains open.
