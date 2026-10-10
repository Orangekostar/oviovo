# Data, calibration, proposals and separation contract

## 1. What is known and what must be discovered

Known: the repository contains full 26-scene parent mappings, frozen FC/text identities, ScanNet export utilities, and calibrated **parent capture** RGB-D manifests. Not known: whether 32 additional authorized ScanNet train families and their raw instance annotations are currently complete on the server. Do not state they exist until inventory verifies them.

Search, in order: explicit `--data-root` arguments; `SCANNET_ROOT`/`SCANNET_DATA_ROOT`; dataset roots in parent source/context/dependency manifests; their documented immediate `scans`/`scans_train` children. Record the resolved path for each scene. Do not recursively crawl `/`, unrelated home directories, inboxes or accounts. Hash large immutable files once, not each minibatch. If local data are missing but an already authorized dataset downloader is installed, fetch only the fixed required scene list and required file types. Dataset enrollment or acceptance of terms requires the user's own authorization; do not automate acceptance or use another person's credentials.

A suitable raw scene has RGB/pose/calibration/depth (`.sens` or equivalent exported arrays), unaligned `_vh_clean_2.ply`, corresponding `*.segs.json`, `*.aggregation.json`, the official label TSV, and official scene split lists. Exact suffixes are resolved against the pinned official preprocessing constants, not a made-up export schema. Existing training-proposal caches may be reused only if camera, class mapping, support and role identities agree.

## 2. Scene-family split: 24 TRAIN / 4 DEV / 4 H

Start from the official ScanNet v2 TRAIN list, not its validation/test list. Family key is `sceneXXXX` from `sceneXXXX_YY`. Exclude all seven current CF18 families (`scene0011`, `scene0050`, `scene0084`, `scene0168`, `scene0231`, `scene0378`, `scene0518`) and all additional families found in project lineage as prior train/tune/evaluation scenes. The existing eight Replica scenes are not used for learning or checkpoint selection. Published backbone pretraining cannot generally be audited at scene level; disclose that limitation separately.

Within each remaining family choose the complete scan with the smallest numerical suffix. Sort families by SHA256 of UTF-8 `LR1-family-20261009|` + family ID, tie by family ID. Take the first 24 as TRAIN, the next four as DEV, and the next four as H. The task is a small-data study; this rule is not claimed representative of all ScanNet. Save the full candidate list, exclusions, selected scans and split file identities.

Roles are fixed before evaluating any new model. No duplicated scan of a physical room may cross roles. A physically separate scan must never be inferred only because its `_YY` suffix differs. Before finalizing the split, unreadable/missing assets may be completed or a technically invalid scan rejected using the same sorted queue; record that reason. After model outcomes exist, no replacement or re-splitting is allowed.

Every chosen family must have at least one usable base-category object; otherwise the corresponding supervised split is insufficient. For H this is checked only after nomination and does not authorize replacing a family. There is no ad-hoc small fallback profile. Fewer than 32 eligible complete families produces `BLOCKED_TRAINING_DATA`, with the exact missing requirements and usable inventory. Do not silently move DEV/H into TRAIN or train on the exposed four-scene probe.

Post-materialization sufficiency checks, counted as original objects (not augmentations): at least 240 TRAIN base-category objects, 32 DEV base objects, 32 H base objects, and eight TRAIN base categories with usable observations. TRAIN/DEV counts are checked before training. H counts are checked only after both seed nominations when H annotations are first opened. Insufficient supervised support is a genuine limited-data block for the corresponding claim; do not change roles after training to repair an H shortfall. Do not manufacture object counts by repeating masks. This is one finite check, not an open-ended search for favorable scenes.

H ingestion is deliberately delayed: during `data` and `features`, inventory/copy its authorized source files and calibration only; do not parse its semantic/instance annotations or construct its proposal examples. The `holdout` stage, after both seed checkpoint nominations, constructs H proposals and features with the frozen data algorithm, evaluates them, and preserves any coverage limitation. No H annotation-derived statistic selects training examples, model architecture or hyperparameters.

## 3. Category separation and the actual meaning of “open vocabulary”

Use the exact 200 valid IDs/class names from the pinned ScanNet200 constants and map to the current CF18 text table by explicit IDs and canonical names. Retain this mapping. Do not treat positional row numbers as original class IDs.

Sort the 200 IDs by SHA256(`LR1-class-20261009|` + decimal ID), tie by ID. The first 40 are adapter-heldout classes; the remaining 160 are base classes. Positive TRAIN examples and CE denominators use only the 160 base classes. A heldout-class object may appear incidentally in the RGB/background or as an identity-contamination donor, but is never given positive category supervision or a heldout text loss. This supports only the claim **no positive supervision for that category in the new adapter**, not that the frozen foundation model never saw it.

DEV checkpoint/branch selection uses base classes only. H and the existing 26 maps are evaluated with the complete respective official vocabularies. H reports base/heldout/all-class recognition, denominators and missing categories; fewer than 20 heldout objects across two H families is `INSUFFICIENT_NOVEL_SUPPORT`, not a fabricated zero-shot result. It does not prevent reporting the other valid experiments.

A fixed label TSV and official `point_indices_from_group`-equivalent mapping determine the TRAIN targets. Official aggregation group ID zero can denote a real object: map group IDs to positive local object IDs (e.g. group ID + 1) and reserve zero for unlabeled, with an explicit reversible mapping. Do not discard group zero by accident. An invalid/unmapped semantic label is not a new class and not a negative example.

## 4. Correct raw ScanNet camera handling

The project's `ScanNet200Dataset` resizes RGB to the depth image and uses depth intrinsics. That loader is appropriate only for its explicitly supplied export format; it is not a general raw sensor registration algorithm. New raw data must be normalized correctly.

For standard ScanNet `.sens` with identity color extrinsic, follow the official `SensorData::saveToPointCloud` convention verified in `SensReader/c++/src/sensorData.h`:

- `T_wd = frame.camera_to_world` maps depth-camera coordinates to world.
- `E_cd = calibrationDepth.extrinsic` maps depth-camera coordinates to color-camera coordinates.
- `T_wc = T_wd @ inverse(E_cd)`.
- `z_m = raw_depth / depth_shift`; do not always assume a hard-coded scale if the sensor header provides it.

For each valid depth pixel `u_d`, `x_d = z_m * inverse(K_d) @ [u_d.x,u_d.y,1]`; `x_c = E_cd @ [x_d,1]`; sample RGB at `project(K_c,x_c)`. Bilinear RGB sampling uses in-bounds floating coordinates; invalid/out-of-bounds samples are marked unavailable, not clamped to the nearest edge. Use a color-camera z-buffer of projected depth samples to reject samples hidden by a nearer sample (same max(0.02m, 0.02*z) tolerance). The normalized aligned RGB is on the depth grid and its calibrated camera is `T_wd,K_d`; keep the measured metric depth and a color-validity mask.

The first importer supports standard sensor files with identity color extrinsic (absolute tolerance 1e-6) or an explicit normalized manifest with documented `T_wd,T_wc,K_d,K_c`. Nonidentity/unknown calibration conventions must not be guessed: require an explicit calibration adapter and provenance, or mark that scene `BLOCKED_CALIBRATION`. Known color-to-depth transforms are inverted explicitly, never chosen by maximizing semantic accuracy. A one-frame reprojection visualization/depth consistency check is engineering validation, not calibration fitting to GT labels.

Use original **unaligned** mesh and `T_wd`. Official ScanNet200 preprocessing can apply `axisAlignment` to mesh vertices; do not combine such aligned geometry with unaligned poses. If only an axis-aligned mesh is available and the transform is recorded, either undo it or multiply every camera-to-world pose by the same transform. Record the choice. No guessed ICP alignment against benchmark annotations.

A small Python 3 port of the official sensor reader may export only selected frames; the published Python 2 script is not assumed runnable unchanged. Keep sensor bytes/checksums and compression support explicit. For already normalized parent regression captures, preserve their exact image/depth/camera convention to avoid altering historical baseline inputs. New properly registered TRAIN images and inherited regression images may have preprocessing-domain differences; report these rather than retroactively changing the parent maps.

## 5. Bounded but substantive supervised examples

Per new family, consider at most 96 finite-pose frames. From all finite frames form pose representatives using 0.20m/15 degrees; when there are more than 96, take equally spaced representative indices using floor(j*(n-1)/(96-1)); when fewer, keep them. Never replace nonfinite poses with identity.

Choose at most 64 original valid objects per family, sorted by SHA256 of `family|original_object_id`. An object must have at least 100 original mesh vertices, nonzero surface area, and at least two qualified calibrated views with >=100 visible target pixels and bbox dimensions >=2. No size or class preference based on expected model performance. Walls/floors are included or excluded according to the same valid-ID/instance rules, not a hand list of easy categories.

Generate four **known supervised proposal conditions** for each object in TRAIN/DEV/H:

1. `clean`: original annotated object surface support.
2. `truncate`: keep 70% of its area along a deterministic signed axis.
3. `append`: append nearest points of other annotated instances within 0.05m of the object, capped at 20% of original area.
4. `truncate_append`: apply truncation then append the same donor set defined against the original object.

Area means vertex quadrature mass derived from triangle areas, not pixel count. Axis is one of +x,-x,+y,-y,+z,-z chosen by stable hash. Use one fixed hash direction per original object in TRAIN, DEV and H; all branches and seeds see the same static perturbation suite. Different corrupt masks are not counted as additional original objects. Truncation threshold is the first area cumulative quantile reaching 0.70, tie by original vertex index. Donor candidates sort by nearest-target distance then vertex index; include a prefix with cumulative area <=0.20*original area. If none exist, mark `NO_DONOR` and retain the resulting support; do not search another scene. Same-class neighboring instances are valid contamination examples for **identity** membership, never a different semantic-class negative.

Render these supports from the full unaligned scene mesh using nearest-hit depth/measurement checks, not by rendering only the target object and exposing occluded surfaces. GT is allowed to construct TRAIN/DEV/H diagnostic proposals and labels. Nevertheless no GT membership, class ID, donor ID, original instance ID, or corruption type goes into the predictor's input tensors. The supervised target is stored separately and is consumed only by the loss or later evaluation.

For TRAIN/DEV/H first make a common per-original-object view bank: intersect geometrically qualified views whose rendered masks meet the FULL minimum for all four pre-generated proposal conditions, then choose the fixed prefix using clean-support geometry. An object with fewer than two common views is omitted from this supervised proposal corpus for every method before training; record the exclusion and recheck sufficiency. This deliberate use of clean annotation to define a diagnostic corpus does not apply to real-map inference. TRAIN supplies paired clean/corrupted views of the same original object. The step draw schedule cycles the corrupt type `truncate,append,truncate_append`; clean is always its paired reference. All four conditions use that same view bank, so no model-dependent or corruption-dependent view selection is introduced. Cases that cannot produce the minimum two common input views are logged during manifest construction. If corruption causes local missing tokens, use padded masks/model-defined fallback; do not remove failures differently for different methods.

DEV/H clean and three corruptions form four records per original object. Report metrics both averaged per original object and per condition; do not treat the four correlated records as four independent objects. This is **proposal-level recognition under controlled perturbations**. It is not a claim that these masks have the same distribution as OVI's predictions, and H is not an independently mapped benchmark.

## 6. Identical view prefixes and physical sampling

For each original supervised object, or each real predicted-map proposal at regression, build a bank of up to eight views before neural classification using the rules above. Anchor is largest qualified FULL pixel area, tie earliest frame. Subsequent view maximizes additional visible physical surface area not covered by earlier choices; tie FULL area, then frame ID. This is a fixed geometric baseline, not a new learned query policy. Use the same ordered bank for FC2/FC4/FC8 and all learned methods. If only 2–7 views exist, use `min(requested, available)` without copying a view. Padding is excluded from every average/attention.

For existing 26 benchmark maps the frame pool is the inherited 32 pose representatives, no additional GT-selected frames. The proposal is the complete original existing D2/G1 owner mask, NOT a GT-corrected or oracle union. Neural eligibility is fixed from: D2 incumbent, valid historical F and D2 prior, no protected raw-zero rows, >=2 qualified views. Eligibility never depends on a new model's class, predicted confidence, correctness or checkpoint. Unselected/unavailable owners keep G1 identically in every method.

Use `canonical_sites` and bounded area quadrature to account for duplicate source rows; do not epsilon-weld adjacent objects. Plan visibility on up to 4096 physical sites. For neural local tokens use 48 deterministic area-quantile sites inside the proposal and up to 16 context sites from other support within 0.05m. Reserve padding for missing sites; do not duplicate sites to manufacture support. Sites are shared across views by physical coordinates, not by independently sampled per-view indices.

At each view test geometric visibility with full-scene occlusion and measured depth. A visible site need not occur in two views to be an input token; only the correspondence auxiliary loss requires two real views. Unknown/invisible is missing, not evidence against the target. A valid context site can be outside the FULL mask, and its occupancy feature then reflects that. No GT membership is used to choose context sites during regression.

Project original pixels through the exact FC resize/padding transform. With original coordinate u, resized width Wr and original width W, use `u_r=(u+0.5)*Wr/W-0.5`; normalized `grid_sample` coordinate is `2*(u_r+0.5)/Wp-1` for padded width Wp, similarly for y. Use bilinear, align_corners=False and an explicit valid mask; padded/out-of-bounds samples do not become clamped border tokens. Retain original pixel and feature-grid metadata for a small overlay check.

## 7. Storage and data identity

Image caches hold exact frozen FP32 raw dense D and optionally pointwise projected U with model/code/preprocess/RGB hash. Same RGB may be used for many masks, but changing mask/support changes region/token identities. Do not infer raw dense features from final cosine vectors. Recompute only missing required images once.

Prepare bounded feature shards and stream objects; do not cache all 32 families on GPU. A CPU LRU may use min(32 GiB, half the observed available host RAM), recorded in the run, to avoid repeatedly decompressing the same frozen features. Cache policy does not alter tensor values or sample order. At the declared maxima, TRAIN/DEV/H uses at most 32*96=3072 distinct image encodings, regression at most 26*32=832. The scientific upper bound is 3904 images, usually reduced by reuse; repeats reuse the same immutable features. Engineering or retry encodings are separately counted. No precision change or lossy cache downcast is authorized.

Before large writing estimate cache size from the first actual dense shape, reserve available space and redirect to an explicit task storage root if needed. Forty GiB free is a minimum initial check, not a guarantee the entire raw dataset fits. Do not delete parent artifacts after ENOSPC. No private/GT dataset payloads are pushed to GitHub; publish split IDs and hashes, not raw scans.
