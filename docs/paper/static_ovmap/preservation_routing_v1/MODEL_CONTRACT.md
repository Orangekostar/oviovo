# Numerical model, loss, and training contract

All coefficients below are **new fixed experimental choices**, not verified optima or guarantees of metric improvement. This contract deliberately keeps the backbone, feature transforms and supervised corpus unchanged.

## 1. Existing numerical operators are the reference

Reuse `learned_object_readout.features.FrozenFC`/`ObjectLoader`, with a task-owned binding/path adapter. Measure/assert parent raw C=1536 and projected/text D=768. Raw `clip_vis_dense` is FP32 after `norm_pre`; the MA input uses the parent's pinned `visual_prediction_forward_convnext_2d`, without extra L2 normalization. For learned pooled raw values, phi is the existing vector visual projection, potentially nonlinear. **Pooling raw features then applying phi is not interchangeable with pooling projected features.**

The image trunk and text/phi parameters stay frozen/eval. Raw feature capture and teacher construction are no-grad; learned MA pooling -> phi must remain in autograd. `ObjectLoader.load()` already returns ordinary per-view FC unit vectors alongside model inputs; expose these explicitly to the new readout. Never manufacture them from class cosine scores.

Teacher embedding for an input with k real views:
`z_FC = unit(mean_v frozen_region_vector_v)`.
Use the parent signed-mask + original hard-support pooling; only originally empty coarse support uses the parent v2 area fallback. Prefixes are2/4/8 while training and8 while DEV/H/H2/main inference. Missing views are not duplicated. Nonfinite/zero vectors are technical failures, not labels or oracle fallback opportunities.

All classes use the parent ordered frozen prototypes. No new class table, dataset ID, absolute scene position, GT category shortlist or trainable text encoder. FC text normalization is not silently changed. The following modules produce embeddings, not class-specific output layers.

## 2. Stage R — matched 2D readouts

### 2.1 MA path

Use the existing `ReadoutHead('NONE')` arithmetic without parent trained weights: three ConvNeXt MA blocks, hidden256, mask input16, four activation maps, zero drop-path. As in the parent:
`weights = softmax(logsigmoid(map_logits), spatial)`;
pool raw D separately under each map;
apply phi to each pooled value;
average projected map vectors per view and normalize;
mean real-view unit vectors and normalize to `f_theta(x)`.

The original frozen image/text backbone is shared. Within a seed, all four R arms start from an identical freshly initialized trainable MA state. Training each separately must reset optimizer and RNG; a previous arm's training must not advance the next arm's sample sequence.

### 2.2 Direct and functional-residual paths

Let `f_init(x)` be a **frozen copy of the exact initial MA parameters**, before any scientific updates. It always uses eval mode, deterministic numerical transforms and the same input x. Store the state bytes/hash, not only an RNG seed. There is no dropout in this experiment.

| Arm | Output | Additional teacher-preservation objective |
|---|---|---|
| PR_R0_DIRECT | `f_theta(x)` | no |
| PR_R1_RESIDUAL | `unit(z_FC(x) + (f_theta(x)-f_init(x)))` | no |
| PR_R2_KEEP | `f_theta(x)` | yes |
| PR_R3_RESIDUAL_KEEP | `unit(z_FC(x) + (f_theta(x)-f_init(x)))` | yes |

At initialization the difference is zero, so residual arms match the same-input FC within the declared FP32 tolerance. Compute the parentheses as written; do not use `z_FC+f_theta-f_init` with a different summation order in the reference check. Do not detach the trainable f_theta or return the teacher through an initialization-only shortcut: the first backward must reach MA weights.

This avoids the double-zero failure of a zero residual times a zero gate. It also keeps the four R arms' trainable architectures identical. Residual arms require an extra **frozen** MA reference at inference; count its stored parameters and compute separately. Direct arms are a matched scratch retraining under this schedule, not a numerical reproduction of the old LR05 warmup experiment. Export the initial reference for every residual checkpoint; never load old failed MA weights as the reference.

An optional cache of f_init/teacher vectors is allowed only with exact image/mask/prefix/reference hashes; retaining dense grids and recomputing no-grad is also acceptable. No cache depends on the ground-truth category. Keep the reference model on CPU or GPU as appropriate without changing values.

### 2.3 Common losses and conditional preservation

For every TRAIN original base-class object, draw one clean input c and one fixed corruption b with the same requested prefix. Parent base-class CE uses all160 base text competitors and temperature0.07:

`L_common = 0.5 CE(z_c,y) + 0.5 CE(z_b,y) + 0.10[1-cos(z_b, stopgrad(z_c))]`.

Compute detached clean teacher `t_c=z_FC(c)` and `m=1[argmax_full200(t_c dot text)==y]`. This is a TRAIN-only supervised predicate. It must not use DEV/H/H2/map correctness, nor a new-head prediction. Construct a small sidecar with object/prefix identity, teacher class and m; the predictor API must not accept it.

`L_preserve = m * {0.5[1-cos(z_c,t_c)] + 0.5[1-cos(z_b,t_c)]}`.

R2/R3 use `L_common + 1.0 L_preserve`; R0/R1 use `L_common`. Average per original item, then batch; do not renormalize only over teacher-correct examples, which would change the effective coefficient. Teacher-wrong objects keep genuine category supervision and common student consistency, but no strong teacher-target penalty. Log teacher-correct denominator and preservation losses separately. This does not guarantee preservation for novel categories, and clean FC can be wrong in ways not represented in TRAIN.

No KL over heldout text prototypes, no positives for the40 heldout categories, no teacher-predicted labels substituted for y. The full200 teacher argmax is used only for deciding whether an already supervised **base** example receives an embedding penalty; record this limited use of the whole vocabulary.

## 3. Stage G — separate quality from membership

### 3.1 Frozen base and common local architecture

Only enabled if Rstar passes the identical DEV qualification in both seeds. For each seed freeze that seed's Rstar, including its initial-reference dependency. Let its output be `z_B`. For all G arms Rstar is eval/no-grad and receives the same input. It is not finetuned by the local branch.

Reuse the parent up-to64 physical sites, raw local vectors d_vg, unit phi(d_vg), eight metadata scalars and validity masks. The eight metadata fields remain occupancy, clipped depth residual, point-to-camera direction(3), object-centered normalized coordinates(3). No target membership or ID enters forward. Replace invalid padded values by zero before the encoder, and mask them from every normalization and loss.

Common trainable architecture in every G arm:
- `E: Linear(D+8,128), GELU, Linear(128,128), LayerNorm(128)` -> h_vg.
- **Quality** Q: Linear128->128, GELU, Linear128->1 -> q_vg.
- **Membership** M: separate Linear128->128, GELU, Linear128->1 -> m_vg.
- Group FFN128->256->128 with GELU and residual LayerNorm, as in parent.
- Four object query vectors dimension128, Normal(0,0.02).
- `W0: Linear(D,D,bias=False)`, initialized all zeros.

All newly initialized Linear/LayerNorm modules otherwise use their PyTorch defaults under the seed. Instantiate all modules even in controls where membership supervision/routing is off; disclose allocated trainable parameters versus parameters with actual gradients. Q and M do not share their final output and M is not fed as Q. Shared encoder allows indirect auxiliary effects, exactly what G3 controls.

VIEW groups contain visible sites from one view. SURFACE groups contain visible views of one physical site. All arms see identical tokens/masks; no numerical site-ID positional embedding. Permutation of views/sites with masks/correspondences moved consistently must preserve inference up to FP32 tolerance.

### 3.2 Intra-group quality pooling

For group g's valid members:
`a_vg = softmax(q_vg within g)`;
`hbar_g=sum a_vg*h_vg`;
`k_g=LayerNorm(hbar_g + FFN(hbar_g))`;
`b_g=sum a_vg*d_vg` (raw C-dimensional values).

`rho_pred_g = mean_valid sigmoid(m_vg)`.
Use a plain mean for group membership, not the quality softmax, so the quality head cannot select only the member with the easiest membership score. Empty groups are removed, not assigned rho0 as an observation. G1/G2/G3 use effective rho=1; G4/G5 use rho_pred. This is the only G4-vs-G3 forward difference.

### 3.3 Absolute routing with a fixed null alternative

Let J be the number of present groups and query index m. Define bounded object compatibility:
`e_mg = 4*tanh(query_m dot k_g / sqrt(128))`.
For controls and candidates alike add one fixed null logit0 and divide the total local evidence scale by J:

`l_mg = e_mg + log(rho_g) - log(J)`;
`[alpha_null, alpha_groups] = softmax([0, l_m1, ..., l_mJ])`.

A zero supplied rho in diagnostic tests means negative infinity, not an arbitrary positive clamp. Predicted sigmoid rho is handled stably via log operations. The1/J corrects the otherwise confounded prior local mass for8 VIEW groups versus up-to64 SURFACE groups. With all e=0 and rho=1, total local attention is0.5 regardless of J. Controls therefore have the same null path; null's mere existence is not an extra advantage only for G4.

`v_g=unit(phi(b_g))-z_B`;
`delta_raw = mean_over_4_queries sum_g alpha_mg*v_g`;
`delta = 0.5*W0(delta_raw)`;
`z_out=unit(z_B+delta)`.

Do not normalize delta_raw or delta. W0 has no bias. No trainable global MA path bypasses rho/null; all local corrections go through the same absolute gate. The frozen base may still contain contamination, so reduced local contamination is not a guarantee of clean final evidence.

At initialization W0=0, output equals the frozen base. On the first step W0 must get a gradient; earlier local layers may legitimately get zero then. By subsequent steps a real nondegenerate engineering batch must exercise the local encoder, Q and queries. M requires an enabled loss or routing to have gradients. Do not “fix” valid first-step zero upstream gradients by breaking initialization.

No valid local tokens -> return z_B exactly and log count0. Never manufacture correspondence or duplicate a padding site. If the predicted representation becomes zero/nonfinite after a finite update, record failure; do not hide it with an accuracy-dependent baseline fallback.

**Claim boundary:** at fixed other scores/values, reducing one rho reduces its own coefficient, and uniformly vanishing rho removes local correction. The vector norm and semantic accuracy need not vary monotonically because group vectors can cancel and final embeddings are normalized.

### 3.4 Supervision matrix

| Arm | Grouping | Membership BCE | Predicted rho in routing | Correspondence |
|---|---|---|---|---|
| G1 | VIEW | 0 | no | 0 |
| G2 | SURFACE | 0 | no | 0 |
| G3 | SURFACE | 0.10 | no | 0 |
| G4 | SURFACE | 0.10 | yes | 0 |
| G5 | SURFACE | 0.10 | yes | 0.05 |

All G arms use L_common plus L_preserve iff the same-seed Rstar used it. Those losses apply to final z_out; z_B is frozen.

Membership BCE applies to M logits, not Q. It averages known valid tokens within each original item (clean/corrupt combined), then items equally. Target membership1, other known instance0, unknown-1 excluded; same-class neighbors are identity negatives, never semantic negatives. Use the unbalanced known-token mean as a fixed matched experiment, while reporting positive/negative denominators; do not automatically rebalance after observing results.

G5 reuses the parent's deterministic at-most128 same-target-site pairs per original item across clean and corrupt (never cross variant site banks). Loss is mean `1-cos(h_vg,h_wg)` for actual corresponding visible target sites, coefficient0.05; no pair -> zero loss plus pair count0. Do not force different parts to match merely because the class is identical.

## 4. Fixed optimization and execution

One A40; frozen FC/text and all trainable new modules FP32; no autocast, TF32 off. AdamW lr1e-4, betas(0.9,0.999), eps1e-8, weight_decay0.01; no decay for biases, norms or learned queries. Global grad clipping1. Effective batch16, microbatch1, accumulate16. Each phase has2,000 updates, no inherited warmup. Use the parent's phase LR shape explicitly: steps indexed0..1999, steps0..99 `lr=1e-4*step/99`; remaining cosine decay to0.1*initial at1999. The zero-LR first update is counted transparently.

Within each seed use parent `draw_schedule` order over **base** original objects: uniform present class then uniform original object; prefix cycle2/4/8 and corruption cycle as in parent. Generate separate TRAIN sidecars per prefix. Reset draw/RNG phase state for every arm; do not reuse evolving optimizer state from a different treatment. R0–R3 initial trainable MA states match; G1–G5 local initial states match. Shared base/reference hashes are explicit.

Checkpoints at0(diagnostic),250,500,1000,1500,2000 for DEV metrics. Every valid candidate checkpoint must be atstep>=250. Do not convert a random untrained head into a chosen trained result. Log scalar components and denominators each update; retain full component curves, DEV tables, gradients every100 steps, selected+last checkpoints and frozen initial/base states. Do not checkpoint whole frozen FC. Track discarded/replayed work on resume.

No post-result hyperparameter sweep. A production incompatibility permits a documented pre-science implementation correction, not dataset-conditioned algorithm changes.

## 5. Finite mechanistic checks

1. Real parent FC vector and text parity; no new normalization.
2. R residual initialization equals same-input teacher; first backward reaches actual MA weights through frozen phi.
3. Teacher-wrong TRAIN item receives exactly zero preservation gradient; teacher-correct item receives the declared one.
4. Quality and membership heads are distinct; constant within-group quality shift leaves local weights unchanged.
5. G null normalization is independent of J for uniform energies; zero rho all groups yields null1.
6. Fixed-key rho intervention reduces that group's alpha and prevents renormalization back to unit local mass.
7. W0-zero base identity and subsequent local-path gradients; no base/backbone parameter movement.
8. No valid tokens, masked NaNs and view/site permutation handling.
9. Training membership and GT never enter prediction inputs; losses and unknown denominators work.
10. Same scientific config resumes optimizer/RNG/data order; checkpoint export/load roundtrip.
11. Real fixed-partition output, protected/recovered class preservation, official rank rebuild and ordered-pool coverage.
12. New H2/real-proposal roles, GT-free candidate lock, and no outcome-based replacement.

Small reference tests are included. They establish mathematical behavior only; perform the corresponding limited production checks. Do not demand nonzero gradients in every parameter on the first zero-initialized G step or count tensor elements as independent tests.
