# Pre-delivery review and corrections

Reviewed on2026-10-11. This is a review of the executable specification and equation reference, not a completed GPU experiment.

## Resolved issues

1. **Wrong immediate parent assumptions:** the latest PR completed with no new full maps. New binding explicitly follows PR -> LR -> source-update lineage. It does not require a successful R nomination or390 PR outputs.
2. **False frozen base:** RHead(False) still invokes MA. New FrozenFCBase computes ordinary FC8 directly; exported FG weights have no MA/R/initial-reference dependency.
3. **Free semantic transformation:** all four arms remove the768x768 value matrix. No claim is made that this matrix caused the old regression; removal makes this experiment more identifiable.
4. **Frozen phi gradient:** raw weighting remains differentiable through frozen phi; image features/text/base values remain frozen. Pool-then-phi order is tested with a nonlinear surrogate, and production CUDA phi checks remain mandatory.
5. **Initialization:** all arms use the same state initialization including membership last-layer zero init. Eta0 and first zero-LR step are intentional; production engineering uses active eta, not a misleading step0 nonzero-gradient requirement.
6. **Membership versus quality:** separate heads and unweighted group membership; FG3/FG4 share supervision and state shape. Disabled membership is not silently trained by a loss computed only for logging.
7. **Group-number confound:** divide positive group mass byJ so simply having more groups does not increase local-vs-null mass.
8. **Mass is not norm:** matched-mass preserves query-wise alpha totals but can change delta norm. A separate explicitly diagnostic norm-matched intervention handles this; zero-control cases are not silently included.
9. **No absolute accuracy guarantee:** the convex-mixture numerical bound is only a nonzero-representation property. It does not preserve argmax or final AP.
10. **D1 oracle firewall:** GT target-only filtering never controls training scope, stage gates or actual predictions. All four main arms run even if D1 is negative.
11. **Selection scope:** fixed69 DEV base originals with full200 competitors; all-class/heldout metrics are post-lock diagnostics. Old H is exposed, H2 is conditionally post-lock. No micro/family metric conflation.
12. **Repeat/full stage graph:**8000 main updates are unconditional. Any main signal triggers8000 repeat updates; only the same nominated architecture's repeat qualification activates286 map rows/22 pools. A different repeat winner cannot rescue it through a post-hoc switch.
13. **Real-proposal target availability:** class matches do not create per-site GT purity. Main real-proposal diagnostics require locked-score outcomes/interventions; site-purity metrics are NOT_COMPUTED without verified separate mapping.
14. **Actual publication:** code, curves, selected heads and reports must be pushed normally. The external receipt follows observed CLI exit and full SHA equality; no self-referential future commit SHA inside a commit.
15. **Finite work:** twelve focused production validation groups, no full-repo safety suite or cold-timing grid. Runtime and memory are secondary records, not rejection criteria.

## Deliberately not claimed or resolved by this pack

- It has not loaded the actual server FC/SAM2 models or trained a new head.
- Current presence and relocation of server caches, licenses and H2 data must be checked during binding.
- H2's historical nonuse across all other possible experiments cannot be guaranteed; check the known finite lineage and report conflicts without changing names.
- The fixed local-site bank may contain insufficient semantic information or noisy correspondence. D1 and the matched controls test that possibility; they do not assume success.
- Source-code inspection and passing synthetic tests do not establish novel scientific contribution. Matched recognition/whole-map gains are required.

The JSON audit and reference test log state exactly which checks actually ran. The package contains no invented server result, learned checkpoint or GitHub publication.
