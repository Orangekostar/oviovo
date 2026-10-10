# Exact model and training contract

All numerical settings below are experimental choices, not claims that these values are optimal. Implement them once and freeze before scientific training. No benchmark-conditioned changes, unnamed approximations or code path used only for tests.

## 1. Frozen feature operator and tensor conventions

Let `D_v` be the FP32 raw ConvNeXt tensor `clip_vis_dense` after `visual.trunk.norm_pre`, shape `[1,C,Hd,Wd]`. Let `phi` be the exact existing numerical visual projection:

```text
[B,N,C] -> [B*N,C,1,1] -> visual.trunk.head -> visual.head -> [B,N,D]
```

C and D are measured from the bound FC_FROZEN checkpoint, not guessed from a source comment. D equals the width of every frozen text prototype. `phi` can be nonlinear: `phi(sum w*x)` is generally not `sum w*phi(x)`. Never interchange these operations for convenience.

Image extraction is frozen/no-grad. FC parameters (including phi), text prototypes and their temperature remain frozen. The application of phi to a **learned weighted raw tensor** during training must remain in autograd, with phi's parameters `requires_grad=False`. Do not use the existing inference worker's outer `inference_mode` around the trained model.

For Mask-Adapter input U use the actual upstream `visual_prediction_forward_convnext_2d` helper: `head.norm(D)` -> permute BCHW to BHWC -> `head.drop` -> `visual.head` -> permute back. Do not L2-normalize this dense U. This dense path is distinct from the raw-vector phi API; verify its output on the bound frozen model rather than assuming equality with a convenient approximation. The same file also has a differently named helper; do not substitute it merely by name. Cache U only if its checkpoint/projection/dtype identity matches. For local token embeddings use normalized phi(d_vg) as separately defined below. A projected dense grid and an already pooled single vector are not interchangeable.

Ordinary FC region vector uses the existing signed +/-1 mask, original hard-support pooling when nonempty, and **area_fallback v2 only on originally empty support**, then phi and L2 normalization. This preserves the actual reference operator. Standard multiview FC2/4/8 is `normalize(mean(unit_region_vectors))` across the selected valid prefix; classification is cosine to the exact frozen text. Zero aggregate/nonfinite results are explicitly unavailable or implementation failures according to the common output contract, never silently filled with a successful score.

## 2. Strong learned 2D control: LR05_MA_8

Port the numerical `MASKAdapterHead` and ConvNeXt block from `hustvl/MaskAdapter@c0516d8a548d90055c3dca7f2a9b4281a4da842f`. Strip framework registration only. Keep three ConvNeXt blocks, mask embedding, final maps and the described arithmetic. Input channel is the measured projected D (the upstream `_large` path expects 768); if the actual FC checkpoint gives a different D, the pack is incompatible and needs a pre-training amendment, not loading mismatched weights.

Parameters: `mask_in_chans=16`, hidden `num_channels=256`, `num_output_maps=4`, drop_path=0, standard upstream LayerNorm eps/gamma initialization. The count of four maps is our controlled protocol setting; do not label this a reproduction of the paper's complete dataset/training recipe.

For each view:

1. Feed U_v and the supplied binary proposal mask to MASKAdapterHead; resizing/downscaling is as in upstream numerical code.
2. Resize its map logits to `(Hd,Wd)` if necessary. For each of four maps, use `softmax(logsigmoid(logits), dim=spatial)` as in upstream FC-CLIP integration.
3. Pool **raw D_v**, yielding four vectors `b_vm` in R^C.
4. Apply frozen phi separately to the four raw vectors, average the four projected vectors, then L2 normalize to `z_v^MA`.
5. `z_MA=normalize(mean_v z_v^MA)` for the available fixed prefix.

Do not prepend the full upstream segmentor, use GT masks at regression, or import official pretrained MA weights trained on additional datasets. All learned branches receive the same freshly trained MA initialization. All share our controlled corrupt-proposal training and classification loss. Report this as **Mask-Adapter architecture, retrained under our protocol**, not “official Mask-Adapter numbers.”

## 3. Proposed hierarchical multi-view readout

LR06/07/08 have identical trainable layer shapes. LR06 groups local tokens by view; LR07 and LR08 group by shared physical site. LR08 differs from LR07 only in auxiliary loss weights. A model's grouping is part of its identity.

### 3.1 Input tokens (no GT or semantic ID features)

Use the same up-to-64 physical sites for all views. For each visible `(v,g)` sample raw feature `d_vg` from D_v using the coordinate transform in DATA_AND_SPLITS. Compute `u_vg=normalize(phi(d_vg))` for embedding; this phi application can be precomputed because d_vg is frozen.

Eight real metadata scalars are: proposal-mask occupancy sampled with bilinear interpolation (0..1); absolute measured-depth residual divided by max(.02,.02*z), clipped to [0,1]; three components of the unit point-to-camera direction; three object-centered coordinates divided by the proposal AABB diagonal. Degenerate diagonal uses 1e-6. No absolute translation, scene name, owner ID, class index, candidate rank, GT purity or corruption type is input. Invalid/invisible tokens are masked, never zero-valued real tokens.

Use trainable `E: Linear(D+8,128)->GELU->Linear(128,128)->LayerNorm(128)`; its output is `h_vg`. Reliability scalar is `r_vg=Linear(GELU(Linear(h_vg,128)),1)`. Both VIEW and SURFACE variants have this same head; BCE supervision is enabled only in LR08. Reliability is not advertised as a calibrated probability unless evaluated separately.

### 3.2 Local group pooling

For every group j, softmax r over its valid members gives a_vg. Compute

`hbar_j = sum a_vg*h_vg`

`h_j = LayerNorm(hbar_j + FFN(hbar_j))`, with FFN `128->256->128`, GELU.

`b_j = sum a_vg*d_vg` (raw C-dimensional value).

In VIEW grouping all sites from the same view are members. In SURFACE grouping all views of the same physical site are members. The post-pool nonlinear FFN is deliberate: simple linear regrouping can collapse to equivalent global averaging and would not test the intended mechanism. Groups and members have no numerical-ID positional embedding. Permuting observations/sites/group order, while permuting their masks and correspondence consistently, must preserve the result within declared floating-point tolerance.

### 3.3 Global per-view path and object pooling

Retain the four raw MA pooled values b_vm for every view. Their trainable keys are

`h_vm^global = LayerNorm(Linear(normalize(phi(b_vm)),128))`.

Use four learned object queries q_m of dimension128, initialized Normal(0,0.02). All newly added Linear/Conv weights use their PyTorch default initializer under the prescribed seed; LayerNorm uses weight1/bias0. The MA head preserves upstream gamma initialization. No class-specific parameter table is created. For each query, attention `softmax(q_m·h/sqrt(128))` is computed **separately** over local groups and over all global MA view-map keys. This prevents a larger number of local groups from changing global-vs-local mass purely by token count.

For a query m:

`b_m^local = attention_pool(local raw b_j)`;

`b_m^global = attention_pool(global raw b_vm)`;

`b_m^fused = .5*b_m^local + .5*b_m^global`.

`z_fused = normalize(mean_m phi(b_m^fused))`.

Final representation:

`z = normalize((1-eta)*z_MA + eta*z_fused)`.

At a training forward use `eta=0.5*min(1, completed_branch_optimizer_steps/200)`; the first forward has eta0, and after200 updates eta0.5. At validation/inference use eta0.5. The one-off gradient check uses eta0.5 so local-path gradients are actually exercised. This is shared by LR06/07/08, not chosen per dataset or object. If no valid local tokens exist, return exactly z_MA with a recorded `NO_LOCAL_SUPPORT_MA_FALLBACK`. Do not fabricate geometric evidence. If global MA output is nonfinite, fail the prediction rather than hiding it behind an oracle fallback.

No separate class logits or learned class lookup is introduced: any supported frozen text vector can be scored by the final D-dimensional embedding. Do not claim that this interface alone establishes unseen-category accuracy.

## 4. Losses and supervision budget

For normalized object embedding z and normalized frozen base-class text vectors t_c, classifier logits are `z·t_c / 0.07`. This training temperature is common and fixed, not fitted. Regression source integration uses the inherited F temperature; the two roles must be recorded explicitly. Do not change a trained branch's deployment temperature after seeing Replica/CF.

Every original object training item has clean and corrupt variants of the same preselected view prefix. All four learned methods use:

`L_base = .5*CE(z_clean,y) + .5*CE(z_corrupt,y) + .10*(1-cos(z_corrupt, stopgrad(z_clean)))`.

This consistency coefficient/recipe is our shared protocol, not a claim to duplicate the full published MA training loss. Each clean/corrupt valid object contributes once; missing views do not change its training weight. If a corruption is a no-op it remains as such, with counts disclosed, not replaced by a more favorable donor.

LR08 adds:

- `0.10 * L_membership`: mean binary cross entropy with logits r_vg for valid local sites whose true membership is known in TRAIN. Label1 means the site's visible surface belongs to the target GT instance, label0 means it belongs to another known instance, label-1 means unknown/ignored. Same-class other instances are label0 here for **identity purity**, but are never used as semantic contrastive negatives.
- `0.05 * L_correspondence`: mean `1-cos(h_vg,h_wg)` for valid observations of the **same physical site** belonging to the target, across different views. Within each object deterministically use at most128 pairs (sort by site, then view IDs); average per object first. No pairs -> zero loss plus denominator0. Do not align different parts or group tokens merely because their class label matches.

Losses read training annotation sidecars only. LR06/07 do not receive hidden purity labels; LR05 and all multi-view branches share clean/corrupt CE and consistency. Thus LR08-vs-LR07 isolates additional supervised geometric objectives, not a different dataset. Report active pair/positive/negative denominators, not just a summed loss that could be zero for all examples.

LR08 may be worse: forcing noisy point correspondences can harm semantics. Keep it as a tested hypothesis, not a mandatory final module.

## 5. Fixed optimization schedule

One A40, no latency objective. FP32 new heads and phi, no autocast/TF32 (set matmul/cudnn TF32 False). The image trunk and text remain frozen and in eval mode. Heads train normally. FP32 caches are shared across all seeds and branches.

Optimizer: AdamW(lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=.01), gradient clip norm1, effective batch16, microbatch1, accumulate16. Weight decay applies to ordinary matrix/conv weights; biases, normalization scale/bias and learned object-query vectors use zero decay. Use the same grouping policy for all optimizers. LR schedule: 100-update linear warmup from0 toinitial, cosine to0.1*initial at phase end. It restarts when a branch optimizer restarts. No lr search or validation-driven optimizer change.

Data draw schedule is generated before training using the seed, samples uniformly over base classes present in TRAIN and then uniformly over original objects within the class. For every item deterministically choose view-count cycle2/4/8 and one corrupted support condition. The exact examples/order/view prefixes/static corruption choice for branch step s are identical across LR05/06/07/08 within one seed. Do not order the data by scene success or GT correction counts.

Seed17:

1. MA warmup: 2000 updates using L_base. Save last as the shared branch start; do not choose a different warmup per method.
2. Fork four branches, each 2000 updates, reset optimizer/RNG phase state under identical schedules. LR05 continues MA; LR06/07/08 inherit the same MA and initialize their additional layers identically from generator seed `(17+1009)`. LR07 and LR08 starts are bitwise equal before training except loss config.
3. Validate each branch at500/1000/1500/2000 updates on the fixed DEV base objects and all four proposal conditions. Choose the checkpoint with minimum equal-family mean per-original-object [0.5*CE(clean)+(CE(truncate)+CE(append)+CE(truncate_append))/6]. A tie within1e-12 chooses earlier step. No DEV benchmark AP, H label, Replica/CF score or latency enters checkpoint choice.
4. Nominate the proposed branch with best DEV criterion among LR06/07/08, tie simpler LR06 then LR07 then LR08. The learned simple control is LR05. Record a descriptive margin vs FC8 and LR05; it is not a promise of whole-map gain.

Seed29 repeats a 2000-update MA warmup and two 2000-update branches: LR05 and the already nominated architecture. Choose each seed29 checkpoint by its own DEV scores using the same rule. The proposed architecture cannot change after seed29 or H. Main scientific updates =2000+4*2000=10000; repeat=2000+2*2000=6000; total16000, excluding the explicitly discarded 20-step engineering fit. Repeats do not justify independence at the object/scene level.

Store step, model/optimizer/scheduler states, RNG states, sampler index, exact config/data hashes, loss values and elapsed time. Checkpoint filenames and a registry identify seed, branch and step. Keep selected and last checkpoints plus the common warmup, not every minibatch snapshot.

## 6. Real regression readout, not post-hoc best-case fusion

All seven nonparent inference methods produce one new F cosine vector for each common eligible existing owner. Form the parent N and Q probabilities with their inherited temperatures. Use the new F cosine with inherited T_F, then call the **same D2 grouping rule**: mean available N/Q probabilities as one group, mean it with F as another group. With all sources this is `.25*N + .25*Q + .50*F_new`. No `.75 oldD2 + .25 new` extra mixture in this task.

The common eligibility requires original historical F, a valid D2 prior, no protected raw-zero support, and >=2 geometrically qualified views. It is fixed before new predictions. Noneligible owners keep G1 exactly. Recovered G1 owners never get new labels. Owner partition, coordinates, triangle topology and target projection remain fixed. Candidate labels come from the full official vocabulary, no GT class shortlist.

Different K methods use prefixes, not hand-picked views. All available frozen/model outputs are computed even when the proposed class equals old class, so effect/coverage statistics are honest. Fallback only for genuinely absent scientific observations as predeclared; nonfinite logits or a crashed model is a failed run. Recompute current-class official area ranks and use the single real output for semantic and instance evaluation.

This isolates representation and views. If a new standalone F representation improves but D2 integration degrades, report that separation; do not add a new learned gate or fusion grid after observing it.

## 7. What counts as novelty evidence

The external Mask-Adapter head, normal attention and a two-level average are not new contributions by themselves. Potential evidence is: a learned view model beats equal pooling; physical-site grouping beats identical VIEW grouping with the same inputs/parameters; targeted supervised objectives improve corrupted-proposal recognition and transfer to real predicted maps. Each must be demonstrated, not asserted.

An optional diagnostic after all predictions are locked may permute cross-view site correspondences while keeping every feature and view unchanged, reporting representation degradation as an input intervention. It is not a main candidate and cannot be used to choose checkpoints. Do not interpret a bad shuffled input alone as proof of improved real accuracy.
