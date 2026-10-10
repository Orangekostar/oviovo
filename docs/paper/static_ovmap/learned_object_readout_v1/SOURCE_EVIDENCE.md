# Source evidence and limits

Inspected 2026-10-09. All local repository references are pinned to `Orangekostar/oviovo@30f59c1fa3d783054dc53fc6693e1fd68e8c2a6d` unless otherwise stated. A Git blob SHA below identifies file contents, not a repository commit. The supplied contracts are newly designed experiments; citations do not imply their numerical choices have already been validated.

## A. Current project

| ID | Inspected source | Confirmed fact and task implication |
|---|---|---|
| E01 | `docs/paper/static_ovmap/DISAGREEMENT_QUERY_HANDOFF.md` | Last study is a four-scene screen; full 26-scene parent is source-update. Physical output storage may live under `/mnt/shared/capacity/node101/ww/`. Do not bind the small probe as a complete cohort. |
| E02 | `src/static_ovmap/disagreement_query/binding.py`, blob `71e2c68d510d223f722352ec75f80d57cefe6445` | Read-only predictor/evaluator namespaces exist; old binders hard-code protocol/branch/count identities. A new task needs its own adapter, not falsified old fields. |
| E03 | `src/static_ovmap/a7_evidence_upgrade/region_adapter.py`, blob `70e101ef4a90f0b6df3e4132572cfa619b5dec87` | Exact FC checkpoint, 800/1333 resize, padding, mask conversion, frozen parameters and AST numerical operator reuse are defined here. Raw dense vs final vector are distinct. |
| E04 | `src/static_ovmap/disagreement_query/physical_support.py`, blob `64cc4932dc16b35a213111b71404da4b4fa77cbc` | Exact duplicate-coordinate/triangle handling, area quadrature and direct depth/occlusion visibility are reusable; they do not create corrected geometry. |
| E05 | `src/datasets/scannet200.py`, blob `2f827218150ed826841ce23d3114551a4e1cc965` | Existing exported-data loader resizes color to depth and uses depth K. It is not proof raw sensor RGB and depth are registered. |
| E06 | `scripts/materialize_oviv2_scannet200_view.py`, blob `89dc4b8e80726b7d2d950ef50a19e122b8ce325b` | Selected-frame materialization and explicit source IDs exist; no verified inventory of 32 fresh supervised families was obtained here. |
| E07 | `src/static_ovmap/source_preserving_update/decisions.py`, blob `6495946e78b20c2f955948d3360179303c9c7760` | Source probabilities are formed with frozen temperatures and grouped N/Q vs F. The new F replacement must preserve this missing-source convention. |
| E08 | `src/static_ovmap/disagreement_query/outputs.py`, blob `4e9e406ddc5b6c6633568ddb7ffcc51ed8491aa1` | Whole-output homogeneous relabels preserve G1 owners/recovered regions and recompute official current-class ranks. Do not substitute a soft score-only evaluation. |

Repository file link root:
`https://github.com/Orangekostar/oviovo/blob/30f59c1fa3d783054dc53fc6693e1fd68e8c2a6d/`

## B. Frozen numerical backbone code

OVRCOAT operator checkout: `nickormushev/OVRCOAT@9fd9450d22852d269d426b521663a127f3983a4b`.

`ovrcoat/modeling/backbone/clip.py`, blob `420feddd17bb8b07db3b297e9951d9211d0840d2`:
- `extract_features_convnext`: stem, four stages, then norm_pre -> `clip_vis_dense`.
- `visual_prediction_forward_convnext`: raw `[B,N,C]` -> fake1x1 -> trunk.head -> visual.head. This numerical method does not itself disable autograd.

The current project's active encoder is its **FC_FROZEN** checkpoint, not OVRCOAT's learned visual trunk. Operator-source reuse does not mean full OVRCOAT inference. The gradient contract and pointwise projection are derived from these actual functions.

`https://github.com/nickormushev/OVRCOAT/blob/9fd9450d22852d269d426b521663a127f3983a4b/ovrcoat/modeling/backbone/clip.py`

## C. Mask-Adapter architecture control

Paper: Li et al., **Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation**, CVPR2025.

Official paper page:
`https://openaccess.thecvf.com/content/CVPR2025/html/Li_Mask-Adapter_The_Devil_is_in_the_Masks_for_Open-Vocabulary_Segmentation_CVPR_2025_paper.html`

Author implementation pinned at `hustvl/MaskAdapter@c0516d8a548d90055c3dca7f2a9b4281a4da842f`:

| File | Git blob | Verified role |
|---|---|---|
| `mask_adapter/modeling/meta_arch/mask_adapter_head.py` | `716db2b5b8ae2bc4be02b384cc2ab5d574c6d0a2` | Mask embedding, projected feature input, three ConvNeXt blocks, semantic activation maps; large path input768. |
| `fcclip/fcclip.py` | `8014152b049ce29435e12bc58db9551896f04b6a` | The actual `visual_prediction_forward_convnext_2d` uses trunk.head.norm/drop and visual.head on BHWC, not the other similarly named helper. Produce projected dense for adapter, transform predicted map logits with spatial softmax(logsigmoid), pool raw clip_feature, then frozen visual projection and head averaging. |
| `fcclip/modeling/meta_arch/convnext.py` | `bc706c0f573b9713bef5079bd59002000ecaa419` | Exact numerical ConvNeXt block/normalization behavior. |

`https://github.com/hustvl/MaskAdapter/tree/c0516d8a548d90055c3dca7f2a9b4281a4da842f`

The paper establishes a strong learned pooling precedent. It does **not** validate our multi-view physical-site grouping, losses or training data. We reuse the numerical architecture with fresh common-protocol training, not published trained performance or a hidden additional training dataset. Preserve license notices. Four maps/hidden256 and our shared loss coefficients are declared experimental settings, not an asserted copy of all author hyperparameters.

## D. 2026 inspiration, not a dependency

Khosla et al., **T-REN: Learning Text-Aligned Region Tokens Improves Dense Vision-Language Alignment and Scalability**, arXiv:2604.18573, submitted 2026-04-20.
`https://arxiv.org/abs/2604.18573`
`https://github.com/savya08/T-REN`

The abstract supports learning region-level language alignment over a frozen visual backbone. Its implementation uses a different backbone/interface, so this pack does not load its weights or claim to reproduce it. No conference-acceptance claim is required. It motivates the level of intervention, not any expected performance improvement here.

## E. Official ScanNet data and camera code

ScanNet official repository: `https://github.com/ScanNet/ScanNet`. Pin the actual source commit at server materialization and verify the relevant observed blobs; do not let a future upstream change silently redefine data.

| File | Observed blob | Confirmed fact |
|---|---|---|
| `SensReader/python/SensorData.py` | `16215cabccb16c99ce258c455350f1a5c152091c` | Sensor header includes both intrinsics/extrinsics, image dimensions and depth_shift; frames retain camera_to_world. Legacy Python2 syntax requires a deliberate port. |
| `SensReader/c++/src/sensorData.h` | `94e77e3d4f9c782e18bda35d4100ecaf306713f8` | `saveToPointCloud` transforms depth camera points to world with frame pose, and to color with calibrationDepth.extrinsic before color projection. This supports the standard-camera convention explicitly used in DATA_AND_SPLITS. |
| `BenchmarkScripts/ScanNet200/preprocess_scannet200.py` | `6f0e2bb79c61095dfb4bb89d07b5bc7a3a4a96ab` | Aggregation/segments/official TSV supply instance/class mapping. AxisAlignment is applied to geometry, which must not be mixed with untransformed camera poses. Group ID zero can be an object. |
| `BenchmarkScripts/ScanNet200/scannet200_constants.py` | `388c4975bbadb45059e5667a59012b39351eded1` | Official valid 200 IDs and labels; bind full mapping rather than raw row indices. |

The raw calibration formula is scoped to the supported standard export. Nonidentity/unknown color extrinsic needs explicit provenance, not a guessed inversion. No assertion is made that current server assets are already in this format or legally downloadable without the user's existing access.

## F. Evidence-based decisions versus hypotheses

**Established by inspected code:** frozen/raw/postprojection interfaces; numerical MA head; source probability grouping; current ownership/rank export; sensor and label mapping fields; existing physical support tools.

**Not established yet:** 32-family data availability; number of valid base/novel objects; real GPU gradient parity of the new model; whether four learned branches converge; whether new-family proposal gains transfer to predicted Replica/CF masks; whether the whole-map target is achievable. Those are real execution outcomes, not conclusions in this prompt pack.

**New design, to test:** physical-site versus view grouping; local/global feature fusion; identity membership and same-site auxiliary losses; fixed family/class split and training schedule. A scalar formula or Transformer is not credited as original merely by appearing here. Main claims require same-data, same-view, matched-parameter comparisons.
