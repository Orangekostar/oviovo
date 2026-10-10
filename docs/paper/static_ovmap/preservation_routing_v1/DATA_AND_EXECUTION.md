# Data, input contracts, and bounded execution

## 1. Reuse the completed experiment, not just its printed results

Source anchor `6767eb90c2fd6999621b356b87269c012821713a` has split identity
`0c1209d9d3887f317251fd7909393329ce804cfa1d42294936addd22d1d56fda`.

TRAIN families are the exact24 parent entries (reported622 base-positive objects/74 observed positive categories). DEV entries are `scene0105_00`, `scene0143_00`, `scene0211_00`, `scene0620_00` (reported69 base objects). Old H entries are `scene0202_00`, `scene0368_00`, `scene0037_00`, `scene0635_00`; their results are already exposed. Do not regenerate the split, remove difficult objects, or move old H into TRAIN.

Bind `split_manifest.json`, `class_split.json`, `data_profile.json`, `features/train-dev.json`, parent generated object manifests and their inputs/targets. Verify actual counts/identities; a discrepancy is not repaired by editing the expected number. The full source/update lineage contains G1/D2 and FC/text identities. Follow actual manifests and PathResolver mappings, not a guessed universal dataset path.

Read-only source paths remain parent paths. Any call that writes `features/text_mapping`, verification memos or diagnostics must use a new task output root. Parent `FrozenFC` accepts a binding-shaped object but writes into its output root; therefore create a real adapted binding with separate `lr_parent_root` and new `output_root`. Do not point an old runner at the parent and falsify its task name/counts.

For base-class training, use only `.base=True` originals. Category competitors remain all160 parent base prototypes even though74 have positives. Record three evaluation groups where labels exist: observed-positive base IDs, zero-positive base IDs, and adapter-heldout IDs. No new positive category supervision for heldout IDs. They may occur in RGB/context as before, and membership supervision is identity supervision, not novel-category supervision.

## 2. Exact same observations for R and G

Reuse parent `ObjectLoader.load()` outputs:
- raw/projected image grids,
- binary proposal masks produced with the exact signed-mask resize/padding,
- local raw/unit features and eight geometric scalars,
- view/local validity masks and original frame IDs,
- ordinary per-view FC vectors.

Parent sites48 inside+up-to16 nearby context, shared across views, remain unchanged. No in-batch resorting by success, no GT-adjusted boxes, no remeshing to improve correspondence. TRAIN/DEV/H-old each retain the original four clean/truncate/append/truncate_append conditions. Corruption masks are correlated versions, not four independent objects. The parent stores labels/memberships in separate targets files; the new forward signature must not consume these.

The teacher cache key includes input object/support, exact view prefix, FC model/text, image transforms and pooling identities. Prefix2 teacher is not prefix8 teacher. Teacher clean correctness is a TRAIN-side supervised loss control, never a DEV/map inference feature. A policy that copies teacher predictions only when GT says correct is prohibited at inference.

Do not create new maps or re-extract N/Q. Existing calibrated TRAIN/DEV RGB and inherited map RGB use their recorded conventions; maintain and disclose the preprocessing-domain difference. Do not retroactively resize/align benchmark images to make them resemble TRAIN.

## 3. H2: genuinely new family-level confirmation, conditional on viable training

Before training, read the pinned official ScanNet v2 TRAIN list already bound by LR. Exclude all parent exclusions, all32 TRAIN/DEV/oldH families, every CF18 family and any additional named project exposure found in the finite lineage manifests. Take the family key before `_YY`; different scans of the same room are not independent.

Order remaining families by SHA256 of UTF-8 `PR1-H2-20261010|`+familyID, tie familyID. Pick the lowest valid scan suffix within each; take four families. Before model outcomes, a technically invalid sensor/header/mesh may be rejected with a recorded reason; no semantic count or model score enters that decision. Missing files may be obtained through the already authorized finite downloader. Lock selected names and file identities before scientific R training; after outcomes no replacement even if H2 lacks novel categories.

Initial H2 handling is names/existence/calibration only. Do not parse instance/semantic annotations, construct H2 examples or extract annotation-dependent features until all R/G checkpoints and nominations for both seeds are frozen. If R qualification fails, no H2 download/feature inference is required and H2 is `NOT_TRIGGERED_NO_FOUNDATION`.

When enabled, use the exact parent sensor calibration and data algorithm: unaligned official mesh with corresponding poses; measured depth scale/extrinsic RGB registration; up-to96 fixed pose representatives/family; up-to64 valid original objects; same four perturbations and common view-prefix selection. Call adapted data functions with H2 role, never change the original LR role metadata in-place. Do not substitute raw RGB resizing for registration. Additional FC images <=4*96=384.

H2 has full200 prediction vocabulary. Report original objects, four correlated conditions, families and categories. At least32 original eligible objects is the planned informative scope; fewer is `INSUFFICIENT_H2_SUPPORT`, with actual outputs still reported and no relocking. Novel-subgroup claim requires20 originals across at leasttwo H2 families; otherwise its generalized claim is `INSUFFICIENT_NOVEL_SUPPORT`. Neither shortage authorizes hiding objects or replacing families.

H2 is **new-family supervised-proposal recognition**, not a new independently reconstructed full-map benchmark. No H2 result changes model, threshold, data split, method nomination or seed. If the dataset cannot be acquired, finish existing valid results and label confirmation technically blocked.

## 4. Real predicted proposals: limited post-lock transfer diagnostic

This tests whether gains on synthetic perturbations carry over to actual model masks. It must not become a new training recipe in this run.

### 4.1 Fixed frames and segmentation

Use the four existing DEV families. From each parent's sorted retained frame bank choose indices0 and floor((n-1)/2), deduplicated. These are fixed before looking at masks, labels or head scores. At most8 original registered RGB images.

Use already bound SAM2.1 Hiera-L (`facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4`, config `configs/sam2.1/sam2.1_hiera_l.yaml`). Resolve exact existing checkpoint/environment from the prior SAM-V/SAM2 probe's asset receipts or explicit CLI overrides; verify identity. Do not load SAM-V, require a new foundation model, or change the FC environment.

Invoke `build_sam2(..., apply_postprocessing=False)` and `SAM2AutomaticMaskGenerator` with every spec setting explicit: grid16x16, batch64, predicted-IoU>=0.8, stability>=0.95, stability offset1, threshold0, box/cropNMS0.7, crop layers0, crop overlap512/1500, no m2m, multimask enabled, no min-region cleanup, binary output. Frozen eval/inference, FP32 (no autocast), separate SAM2 environment. Count image encoder calls and prompt batches;8 images does not mean8 mask decoder prompts.

Generate without GT points/boxes. Retain masks with>=100 pixels and bbox extents>=2. Sort by model predicted IoU, stability, pixel count descending then mask digest ascending; keep at most16 per frame, <=128 total. Lock all retained masks before GT matching. The same masks serve every readout. This is SAM-generated-proposal evidence, not an OVI frontend replacement.

### 4.2 From proposal to common multiview inputs

For each retained mask take the source rows actually visible within it from the existing full-mesh source-row observation. Deduplicate rows; do not expand to the entire GT owner. Source geometry is the mesh, but annotation ownership must not be used in the predictor support builder. Render this exact row subset over the same<=96 DEV candidate frames using measured depth/full-scene occlusion; choose up-to8 views using parent largest-area/new-area prefix rule. The seed frame's predictor mask is this consistently rendered support, and both the original SAM mask and resulting reprojection are stored separately. This diagnostic explicitly measures the resulting 3D-lifted predicted proposal.

Build physical sites with proposal-vs-context flags, not GT owner identities; use exact-coordinate grouping and parent quadrature with48 proposal/16 context sites. No owner-based oracle repair. Fewer than2 geometric qualified views is a recorded common unsupported proposal; never replace it by a different SAM mask selected with GT. All heads evaluate all common usable proposals, and report the generated -> liftable -> multiview -> matched coverage funnel.

Only after proposal inputs and head outputs lock, compare the row subset with official eligible GT instance supports on the **same raw mesh**. Best class-agnostic IoU>.25 assigns a diagnostic target (tie smallest official GT ID); stratify (.25,.5] and>.5. Other cases are undefined, not automatic category errors. Multiple predicted proposals may map to the same GT: report both proposal-level and equal-GT-averaged metrics, not inflated independent-object counts. Per-frame cap remains prediction-only.

The frozen FC grid collection may use existing DEV data; at most4*96=384 distinct context images. No gradients, checkpoint selection or replacement threshold is fitted on this diagnostic. It is not a new unseen-family confirmation; its families are DEV. Generator technical failures are explicit, whereas truly empty/few outputs are valid coverage outcomes. Do not rerun the generator with looser settings because too few masks match.

## 5. Old H and complete-map regression

Old H may be re-evaluated **only after both seed choices or an intentional early-stop declaration**, for retrospective comparison; it cannot admit a failed R arm to G training. Keep all original labels/conditions and disclose exposure.

For the26 map regression, reuse the exact parent complete eligible owner set, eight-view prefix, physical samples and FP32 features. The original LR cohort was not limited to the later64-object probe; do not accidentally substitute that small list. Preserve parent noneligible owners and all G1 restored owners. New source vectors replace F through the same N/Q/F grouped probability rule and frozen F temperature. Missing source rules remain identical. No additional whole-D2 blend, class-specific threshold or new confidence gating.

Map class order is the dataset's actual frozen order (Replica versus ScanNet200), not the TRAIN200 table. New embeddings stay D-dimensional and score the respective text prototypes. Guard zero-update equivalence at both levels: residual initialization equals same-input FC8; a literal no-relabel map equals G1. These are different references.

## 6. Resources and resume

Prefer reuse of ordinary FP32 dense/features and cached teacher/reference vectors. New H2 <=384 images, real-proposal context<=384; same frame shared by many masks is encoded once. If parent image caches are unexpectedly absent, exact recovery of originally required tensors is allowed, separately counted, up to the old3904-image cap. Maximum aggregate FC bound=4672, not an expectation of this many new encodings. Do not downcast caches or claim zero physical inference before measuring.

Keep host LRU <=min(32GiB, half available RAM). Estimate required storage from real shapes; initial40GiB is not a guarantee. Redirect new outputs using explicit storage roots, keep parent symlinks read-only, never delete licensed inputs or previous evidence after ENOSPC.

In normal path R/G updates total<=30,000; no large cold-timing matrix. Log actual training, feature capture, proposal generation, inference, payload construction, evaluation and packaging separately; do not sum nested clocks or interpret cached inference as cold latency. Missing historical failure time remains null. A slower accurate result is still eligible.

Preserve optimizer/scheduler/RNG/draw position and frozen base/initial reference hashes. A trainable head cannot resume against a different frozen Rstar. A public inference weight loader must locate all required small dependencies explicitly and validate frozen FC/text identities without containing their weights.
