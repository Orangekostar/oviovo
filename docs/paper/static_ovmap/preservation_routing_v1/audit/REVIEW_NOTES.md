# Pre-delivery review notes

This review checks the instruction pack and small numerical examples, not server training.

## Corrections made while designing the final protocol

1. Replaced ambiguous zero-output-plus-zero-gate residual initialization with the explicit function difference `f_theta-f_init`. It preserves the same-input frozen FC and permits MA gradients on the first step. The frozen reference is a real inference dependency, not free hidden state.
2. Separated same-view FC identity from whole-map G1 identity. Existing FC8 differs from G1, so passing a representation identity check cannot establish whole-map preservation.
3. Kept exact parent TRAIN/DEV proposals, classes and feature arithmetic; no simultaneous data expansion or change of calibration in the main factorial.
4. Separated relative view-quality logits from absolute membership logits. Membership routing is applied across groups with a fixed null alternative.
5. Added division by effective group count J to avoid a hidden VIEW-vs-SURFACE local/null prior difference. J counts all valid groups even when rho is diagnostically zero.
6. Prohibited normalization/nonzero bias of the gated residual before adding it to the base. Otherwise low route mass can be undone.
7. G3 and G4 have the same membership supervision, modules, starting state and null route. Only direct membership routing differs. G5 adds correspondence separately.
8. Explicitly allowed valid first-step upstream zero gradients for the zero-initialized G residual matrix; by later steps the real local path must be exercised.
9. R selection uses full200 competition on DEV base objects. CE is a tie-break; original heldout-category labels are not used for selection.
10. Teacher correctness is TRAIN-only and clean-prefix-specific. It must never be an inference-time oracle or a penalty forcing teacher mistakes.
11. Added a deterministic early negative path after8,000 or10,000 updates. G is not trained on a base that failed the corresponding two-seed foundation gate.
12. Normal-path direct G pairs are repeated in both seeds. Only Rstar is repeated; the four-way R factorial cannot be claimed fully replicated.
13. H-old is labeled exposed. H2 names are reserved pre-outcome and annotations opened after all selections. Genuine low support is reported, not fixed by outcome-based replacement.
14. Added a strictly post-lock real-proposal diagnostic using only8 automatic SAM2 image calls. No GT prompts, no training on these masks, no claim that this is a new mapped benchmark.
15. Map expansion is9 seed17 +6 seed29 paths:390 rows and30 ordered pools. All four R arms remain reported at proposal level; only Rstar gets the declared map expansion.
16. Required publication of per-component training curves and every DEV checkpoint summary, correcting the previous release's limited public curve coverage.
17. Removed latency selection and large cold-timing grids. Runtime and memory remain descriptive costs.
18. Technical blocks and completed negative gates have different exit/status semantics. A missing activated H2 or model interface must not become fabricated completion.

## Remaining experimental uncertainties

Conditional preservation may still inhibit useful changes. Frozen random-reference subtraction may be less effective than other residual parametrizations. Group membership can be mispredicted, and the frozen base itself can contain pollution. H2 data/novel support and retained parent caches are not verified in this authoring runtime. None of the mechanism identities guarantees top1, AP or mIoU improvement.

No production code, trained head or GPU accuracy result was created during this review. The included code is deliberately restricted to arithmetic and toy CPU autograd checks.
