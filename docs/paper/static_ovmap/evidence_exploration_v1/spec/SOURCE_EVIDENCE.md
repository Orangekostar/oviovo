# Source evidence and limits of the planning audit

Verified on 2026-10-07 through the connected GitHub reader and primary paper pages.
This is a source-level planning audit, not a run on the user's A40 cluster. Public
container networking was unavailable; code was inspected via the connector. No
new FC/AnyUp inference, private-data scoring or current-server file validation was
performed while preparing this package.

## A. Current implementation

All project paths below are under `Orangekostar/oviovo` at
`30a6e07eb2b2fe6f69dc9bfa8405df171ff17f9f`.

| ID | Exact path / symbols inspected | Verified fact | Implication |
|---|---|---|---|
| S01 | `docs/paper/static_ovmap/RUNTIME_PARITY_RESULTS.md` | R2_ROI_EXACT is retained; G1 cold time 18.853 s, reference 44.448 s; G1 42/53, AP12.39; U2 9/53, AP12.39; 26 geometry checks and inherited 172/14 results | Keep accelerated G1 and target its marginal quality, not another blind time/model search |
| S02 | `src/static_ovmap/runtime_parity/binding.py` | Actual v2 is resolved via parent reference, contexts, regions, predictions and metrics; records genuine U2 vs v2 operators separately | Bind actual source artifacts; do not reconstruct old results from reported numbers |
| S03 | `src/static_ovmap/runtime_parity/runner.py` | `load_common` loads xyz/faces/raw, incumbent predictions, N/Q/F, frozen vocabulary and evaluation projection; `recover` calls `build_views`, encoding, classification and export | Reuse loader, separate decision inputs from evaluator inputs |
| S04 | `src/static_ovmap/runtime_parity/views.py` | R2 selects geometry-ranked Top-3; G1 fixed prefix; `scientific_key` separates semantic input content from producer IDs | Freeze actual G1 views and use content-based cache identities |
| S05 | `src/static_ovmap/cvpr_compact/projected_views.py` (also inspected in prior source audit; rechecked call contract through current views) | FullSceneProjector keeps full-scene occlusion, homogeneous face owners and camera-z consistency | Additional control maps need full selected-frame owner rasters, not partial ROI unknowns |
| S06 | `src/static_ovmap/cvpr_compact/area_fallback.py` | Normal signed pooling is preserved. Empty support uses area occupancy and original visual head | Area fallback is already present; do not relabel it as a new experiment |
| S07 | `src/static_ovmap/runtime_parity/session.py` | Original dense FC features precede projection; coarse vectors remain FP32; U2 bypasses v2 fallback | Preserve original coarse vectors; new operators have their own cache identity |
| S08 | `src/static_ovmap/cvpr_compact/outputs.py` | `build_method_outputs` ties available sources to unconditional argmax; `construct_output` accepts a final recovery-label subset and recomputes ranks | Add a new explicit decision/export adapter; never misuse feature failure to represent deferral |
| S09 | `src/static_ovmap/cvpr_compact/evaluation.py` | Complete five metrics, expanded native registry, full-cohort class pooling, source/scorer identity checks | Preserve official scoring and count unknown/missed objects |

Direct source links:
- https://github.com/Orangekostar/oviovo/blob/30a6e07eb2b2fe6f69dc9bfa8405df171ff17f9f/docs/paper/static_ovmap/RUNTIME_PARITY_RESULTS.md
- https://github.com/Orangekostar/oviovo/blob/30a6e07eb2b2fe6f69dc9bfa8405df171ff17f9f/src/static_ovmap/runtime_parity/binding.py
- https://github.com/Orangekostar/oviovo/blob/30a6e07eb2b2fe6f69dc9bfa8405df171ff17f9f/src/static_ovmap/runtime_parity/runner.py
- https://github.com/Orangekostar/oviovo/blob/30a6e07eb2b2fe6f69dc9bfa8405df171ff17f9f/src/static_ovmap/runtime_parity/views.py
- https://github.com/Orangekostar/oviovo/blob/30a6e07eb2b2fe6f69dc9bfa8405df171ff17f9f/src/static_ovmap/cvpr_compact/area_fallback.py
- https://github.com/Orangekostar/oviovo/blob/30a6e07eb2b2fe6f69dc9bfa8405df171ff17f9f/src/static_ovmap/runtime_parity/session.py
- https://github.com/Orangekostar/oviovo/blob/30a6e07eb2b2fe6f69dc9bfa8405df171ff17f9f/src/static_ovmap/cvpr_compact/outputs.py
- https://github.com/Orangekostar/oviovo/blob/30a6e07eb2b2fe6f69dc9bfa8405df171ff17f9f/src/static_ovmap/cvpr_compact/evaluation.py

Selected Git blob identities returned by the source reader (Git blob SHA is not a
plain file SHA256; do not compare them as if they were the same algorithm):
- area_fallback.py: `18afa2856f7508d1a5014d8a6dc6282cd892183e`
- runtime_parity/session.py: `2416f1ad2c1a8a1d50076b615e021c1f6d7b5d7c`
- runtime_parity/runner.py: `f47f18a88cdb93acad53eedcf233772101aefd9b`
- runtime_parity/views.py: `33a6af15c724ada095eeae1a41126d60a8adf28b`
- cvpr_compact/outputs.py: `8b7bd06d012f4abb6706533130c73ec8c3057b25`
- cvpr_compact/evaluation.py: `42c44388c31fd9fa7a8a22dd4a63e47bfe8e980b`

## B. AnyUp: primary implementation, exact checkpoint

Paper: AnyUp: Universal Feature Upsampling, ICLR 2026 Oral. Its paper preprint and
original checkpoint predate the 2026 conference; do not describe all code as first
released in 2026.
- Primary project: https://wimmerth.github.io/anyup/
- Repository: https://github.com/wimmerth/anyup
- Source revision: `351807a9c4287368732cc247f26c7c81c9139af4`
- `hubconf.py` exposes `anyup` (paper DINOv2 ViT-S-trained upsampler) and a DIFFERENT
  `anyup_multi_backbone`. We deliberately select the original `anyup`.
- `anyup/model.py`: Q/K from image and normalized feature encoders, V is raw input
  feature; output resolution is configurable.
- `anyup/layers/attention/chunked_attention.py`: windowed MHA returns averaged
  attention, which weights unprojected V; query chunks are supported.
- `attention_masking.py`: fixed window2d masks and pixel-center indexing.
- An attempted fetch of `anyup/layers/attention.py` was 404. Tree inspection resolved
  the actual directory above; the final instruction uses the actual path.
- Original checkpoint asset 304207685, 3,540,612 bytes, SHA256
  `9d035c0f27114a6f32bdd3d8ed93b6cd39dad4b8f8bf94e69fddbfbf022901b2`, confirmed from
  https://api.github.com/repos/wimmerth/anyup/releases/tags/checkpoint
- Repository LICENSE reads Creative Commons Attribution 4.0 International. Preserve
  attribution and modification notices. The server must verify downloaded weights.

The owner/depth resampling factor in our instruction is a NEW PROPOSAL. It is NOT
present in AnyUp and is not an official API accepting an owner map. Codex must build
a small adapter and prove its neutral path matches the original.

## C. The FC feature/head boundary

Pinned upstream OVRCOAT operator:
https://github.com/nickormushev/OVRCOAT/blob/9fd9450d22852d269d426b521663a127f3983a4b/ovrcoat/modeling/backbone/clip.py

Read functions `extract_features_convnext` and `visual_prediction_forward_convnext`.
The former exposes norm_pre dense visual features. The latter accepts BxQxC pooled
visual channels, reshapes to 1x1, applies trunk.head then visual.head, and returns
text-space features. For this ConvNeXt path the masks argument is not used in the
head; it is not permission to ignore masks during pooling. Do not feed text-space
cosines or already projected 640-dimensional vectors to AnyUp in place of dense V.

## D. Idea references, not executable whole-pipeline dependencies

1. FunFact (CVPR 2026): competing functional relations are represented with geometric
   and structural factors rather than independent top scores. We borrow the need
   to distinguish available evidence from accepted hypotheses. Our target is
   recovery class/acceptance, not functional graph edges. No FunFact solver, LLM,
   posterior calibration result or performance metric is being reproduced.
   https://funfact-scenegraph.github.io/
   https://arxiv.org/html/2604.03696v1

2. GeoGuide (CVPR 2026): geometric priors guide semantic consistency in a learned
   3D segmentation model. This motivates geometry-aware representation, not a
   claim that our frozen gate implements its uncertainty distillation or mask
   reconstruction. No GeoGuide network training is authorized.
   https://openaccess.thecvf.com/content/CVPR2026/html/Tao_GeoGuide_Hierarchical_Geometric_Guidance_for_Open-Vocabulary_3D_Semantic_Segmentation_CVPR_2026_paper.html
   https://arxiv.org/html/2603.26260v1

3. OVRCOAT is already the source of reused FC operators. Its candidate-objectness
   and region-recognition motivation does not establish that our new contrast
   rule is novel. Compare direct, simple alternatives in this study.

Preference adaptation, OCH3R, active view policies and recent preprints discussed
previously are deliberately OUT OF SCOPE. They are not prerequisites and must not
be installed merely to make the task sound current.

## E. Claims not established by this audit

No new method is measured. No guarantee of improving CF18 or preserving Replica
has been established. A source-level reason for a candidate is not evidence of its
success. The proposed constants are fixed design choices, not learned optima.
Signed contrast is not a calibrated posterior; geometry-conditioned attention is
not proven feature deconvolution; frozen weights do not imply the overall historical
system was never fitted. Final claims must follow the actual completed experiments.
