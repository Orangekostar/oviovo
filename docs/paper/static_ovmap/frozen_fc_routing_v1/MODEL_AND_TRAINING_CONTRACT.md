# Model and training contract

All equations describe new experimental settings. None claims guaranteed classification or AP improvement. Implement one production model, shared by optimization, inference and interventions.

## 1. Inputs and the actual frozen reference

Use the parent PR ObjectLoader adapter over LR banks. It appends `fc_vectors` to the parent feature inputs. The new head consumes only:

| key | shape | meaning |
|---|---|---|
| fc_vectors | V x 768 | exact ordinary FC region vectors, already unit normalized |
| view_valid | V | true observed views, never replicated padding |
| frame_ids | V | observation identity, not a learned numerical feature |
| local_raw | V x S x 1536 | raw ConvNeXt feature samples |
| local_unit | V x S x 768 | frozen unit phi applied to each sample |
| metadata | V x S x 8 | existing occupancy, depth residual, direction3, centered-position3 |
| local_valid | V x S | real valid observations |

S is at most64, view count at most8. The old loader may still read raw/projected grids to obtain these inputs; the new head never invokes MA or uses the full projected grid. Targets, class indices, dataset/family/owner names, teacher-correct flags and corruption types do not enter `forward`.

`FrozenFCBase(inputs) = unit(mean(fc_vectors[view_valid]))`, with zero trainable parameters. Use the actual `region_vector()` ordinary hard-mask/area-v2 fallback result, not a newly approximated mask average. All input vector values and phi/text identities remain frozen.

The old vector phi is nonlinear. For trainable weights `a`, use **unit(phi(sum a*x_raw))**, never sum a*unit(phi(x_raw)). The raw image extraction and per-token frozen input embeddings can be no-grad; the weighted raw -> phi path must retain its derivative with respect to a. `phi.requires_grad_(False)` freezes parameters, not input derivatives. Keep all numerical operations FP32, no autocast/TF32.

## 2. Identical head shapes in all four methods

All heads contain exactly these trainable layers (216,194 parameters with measured768-D text):

```text
embed: Linear(776,128) -> GELU -> Linear(128,128) -> LayerNorm(128)
quality: Linear(128,128) -> GELU -> Linear(128,1)
membership: Linear(128,128) -> GELU -> Linear(128,1)
group_ff: Linear(128,256) -> GELU -> Linear(256,128)
group_norm: LayerNorm(128)
queries: Parameter[4,128], Normal(0,0.02)
```

There is NO trained MA, no frozen initial MA, no trainable linear mapping of 768-dimensional values, no global class lookup, no trained text head. Hidden/key transforms may learn HOW to pool, but the output values remain in the frozen FC space. Use the same modules even where membership is untrained/unused; report both total architecture and actually participating parameter counts.

Initialize all arms under `seed+1009` with the same reset sequence. All ordinary Linear/LayerNorm layers use the parent defaults, except the final membership Linear's weight and bias are zero in EVERY arm. This gives rho0.5 initially in FG3/FG4 rather than arbitrary initial purity. Do not load any PR/LR trained head. Verify identical initial state dictionaries once per seed.

## 3. Local groups

After dropping invalid views, mask invalid local data with zero before the embedding MLP. Invalid slots never participate in softmax, group counts or losses. Compute `h = embed(concat(local_unit,metadata))`, q=quality(h) and m=membership(h).

FG1 groups valid sites of the same view. FG2/3/4 group valid views of the same physical site. Remove groups without valid members. For group j:

```
a_j = softmax(q of valid members)
hbar_j = sum a_j*h
key_j = group_norm(hbar_j + group_ff(hbar_j))
b_j = sum a_j*local_raw
u_j = unit(phi(b_j))
v_j = u_j - z0
rho_j = mean(sigmoid(m of valid members))
```

Rho is an UNWEIGHTED mean across valid member observations, separate from q; do not let quality reweight its own membership targets. It is a predicted target-membership proxy, not a calibrated probability claim. Same-class other-instance points are membership negatives, not semantic negatives. Their supervision is still class-agnostic instance membership.

If no local observations are valid, return z0 exactly and `NO_LOCAL_SUPPORT_FC_FALLBACK`. No fabricated token. A differentiable zero anchor may keep a no-information training item in the batch without inventing a gradient; log such items. Do not filter them after seeing labels.

## 4. Group routing and preserved baseline

For query k and group j:

```
e_kj = 4*tanh(dot(query_k,key_j)/sqrt(128))
rho_effective = rho_j   in FG4
rho_effective = 1       in FG1/FG2/FG3
w_kj = exp(e_kj)*rho_effective/J
alpha_kj = w_kj / (1 + sum_h w_kh)
alpha_null_k = 1 / (1 + sum_h w_kh)
```

J is the number of observed groups. This division prevents the local mass rising merely because SURFACE has more groups than VIEW. The null option's logit is fixed0 and value is **no change**. Group energies are bounded so direct positive-weight arithmetic is stable; this also permits exact rho=0 without log(0) backward issues.

```
delta = eta * mean_k(sum_j alpha_kj * (u_j-z0))
z = unit(z0 + delta)
```

No separate normalization, bias, projection or LayerNorm on delta. The local correction is not forced to unit length. `eta = .5*min(1, completed_optimizer_updates/200)` for training; at all normal evaluation/inference, eta=.5. The 0-step diagnostic explicitly uses eta0. Keep a graph-connected zero at eta0 so the first zero-LR step behaves as specified; do not demand nonzero classification gradients at this first step.

No-local support, eta0, or all effective rho0 must return the same FC base values (an exact return with a graph-connected zero is allowed). A missing required FC vector is a data problem; a nonfinite representation or phi output is an implementation/runtime failure, not a reason to reuse a good label silently.

### A useful numerical bound, not an accuracy theorem

Because sum_j alpha_kj = r_k <1, the pre-normalized vector is

`(1-eta*rbar)*z0 + eta*mean_k(sum_j alpha_kj*u_j)`.

With eta<=.5, it is a convex combination with at least half the pre-normalization mass on z0. With energies in[-4,4] and rho<=1, r<=exp(4)/(1+exp(4)), so its norm has the lower bound `1-2*eta*r >= 1/(1+exp(4))` at eta=.5 for unit z0/u. This prevents exact cancellation under the declared conditions. It does **not** preserve the argmax, margins or whole-map AP. Do not label it a no-regression guarantee.

## 5. Losses and target isolation

Use exactly the existing base loss on clean and one fixed corrupt variant:

`Lbase=.5*CE(zclean @ base_text.T/.07,y)+.5*CE(zcorrupt @ base_text.T/.07,y)+.10*(1-cos(zcorrupt,stopgrad(zclean)))`.

Class supervision uses the exact160 base text rows. The40 adapter-heldout classes provide no positive classification targets; full200 competition is used in DEV recognition. Context may include heldout objects, as before; do not claim no foundation-model exposure.

FG1/FG2: L=Lbase. FG3/FG4: add `.10 * mean_BCEWithLogits(m,target_membership)` over known valid tokens pooled across this original object's clean and corrupt conditions. First reduce per original object, then mean over the batch. Unknown targets=-1 are excluded; empty known denominator -> differentiable zero, with denominator0 recorded. No geometric loss for class labels, no correspondence term, no teacher-correct/teacher-preservation term.

Log CEclean, CEcorrupt, student consistency, membership BCE, total, true known/positive/negative/unknown counts, local observation count, gradients at the declared sparse intervals, and actual optimizer steps. Computing a BCE scalar for a disabled arm must not leak its gradient. For FG3 the membership task may influence shared h indirectly; FG4-vs-FG3 specifically tests direct use in routing in addition to that supervision.

## 6. Schedule and checkpoints

One available bound A40. Effective batch16; microbatch1, accumulation16. AdamW lr1e-4, betas(.9,.999), eps1e-8, weight decay.01 for matrix weights; no decay for biases, normalization or queries; clip norm1. Reuse parent `learning_rate` exactly (initial0,100-step warmup,cosine to1e-5 by2000). No extra warmup model.

Reuse `draw_schedule(train_objects,seed,updates=2000)` over the ORIGINAL complete parent list, whose function chooses base-positive objects. All arms within a seed use the identical schedule bytes. Prefixes cycle2/4/8, corrupt condition schedule is inherited. Each arm has32,000 original-item draws and64,000 clean/corrupt forwards; these are not independent new objects.

Run all four seed17 arms,2,000 updates each, including negative arms. Evaluate DEV base full200 predictions at250/500/1000/1500/2000. At0, record FC identity only and exclude it from all selection. Save selected and last optimizer state, RNG/sampler and scalar component curves. A100-step resume checkpoint interval is sufficient. Replayed uncheckpointed steps are separately counted; existing valid checkpoints resume exact counters/state and input identities.

If any seed17 arm qualifies, repeat all four with seed29, otherwise no repeat. Training sums are8,000 or16,000; no dynamic epoch extension, hyperparameter grid or third seed. Different validation-selected steps are disclosed in paired contrasts; also provide the already logged same-step DEV contrasts to distinguish checkpoint choice from architectural effects.

## 7. Post-lock interventions

See DIAGNOSTICS_CONTRACT.md. Crucially, MATCHED_MASS preserves query-wise local probability mass, **not correction vector norm**. Include the separately labeled norm-matched check instead of confusing the two. These use the same learned head and no optimization; they cannot win the model selection or appear as extra trained methods.
