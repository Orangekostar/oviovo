# CROVE Paper Integrity Audit

Date: 2026-08-20
Mode: full
No-invention policy: enforced

## Artifacts Checked

- Paper-story, A5, A6, and dual-center design specifications.
- Markdown and LaTeX benchmark templates, populated views, and token registry.
- Checked-in T1/T4 result JSON files and static result summaries.
- Pipeline-figure prompt and baseline execution report.
- Closest-work literature report and primary-source links.
- T1 protected-source, non-interference, and table-package verifiers.

Unavailable for audit: a complete manuscript, rendered paper PDF, bibliography,
appendix, and final frozen T2/T3/T4 result package.

## Claim-Evidence Matrix

| Claim | Status | Authoritative evidence | Integrity decision |
| --- | --- | --- | --- |
| CROVE has a competitive static mapping foundation | Partially supported | 33 verified T1 registry cells; Replica and ScanNet CROVE result JSON files | Supported for the populated metrics and listed comparable baselines only; OVI-MAP and several ScanNet baseline cells remain unresolved |
| Temporal maintenance leaves the cumulative T1 path unchanged | Supported | Protected-source verifier passed; `tests/oviv2/test_t1_noninterference.py` passed 8/8 | Safe guardrail claim, not evidence of dynamic superiority |
| Occlusion does not reduce object existence | Supported at mechanism level | Neutral evidence transition in `temporal_lifecycle.py` and focused unit/runtime tests | May be stated as an algorithmic invariant; benchmark benefit still requires ablation evidence |
| Visible absence and reversible ownership remove stale state | Partially supported | Runtime and ledger contracts plus mechanism tests | Table 2 and Table 3 outcome claims are not supported because every relevant result cell is unfilled |
| Role-separated association prevents lookalike identity leakage | Partially supported | A5 specification and active/dormant gate tests | Behavior is implemented, but no source-bound false-re-ID or held-out identity result proves empirical benefit |
| Qualified reappearance improves dynamic recognition | Partially supported | A6 implementation contract and tests | The cited Apartment diagnostic is development-only and lacks a checked-in result binding |
| Dual-center readout reduces trajectory bias | Partially supported | Dual-center diagnostic specification reports two position comparisons and a TP substitution | The behavior is not implemented and the diagnostic values are not bound to a checked-in artifact |
| CROVE improves dynamic current-state mapping over baselines | Unsupported | T2 registry contains 70 unfilled cells | Withhold from Abstract, Introduction contributions, Results, and Conclusion |
| CROVE has bounded online overhead | Unsupported for the proposed method | Baseline T4 has 36 verified cells, but Khronos and both CROVE rows remain unfilled | Baseline timing may be reported; no CROVE efficiency comparison is currently valid |
| Online open-vocabulary current-state mapping is the core novelty | Overstated | DynaMem, Where Did I Leave My Glasses?, DualMap, Khronos, and related work | Reframe to the narrower state-compatible causal evidence mechanism |

## Numeric Consistency Findings

### Submission-Blocking

1. The central evaluation is absent from the registry:
   - T2: 70 `UNFILLED`, 0 verified.
   - T3: 48 `UNFILLED`, 0 verified.
   - T4: 36 verified baseline cells and 36 `UNFILLED` Khronos/CROVE cells.
   - S1/S2/S3: 70/30/30 `UNFILLED` cells.

2. T1 is not a complete comparison table. It contains 33 verified cells, 22
   unfilled cells, and 5 protocol-level N/A cells. The populated CROVE row is
   numerically consistent with the checked-in Route 1 and ScanNet result JSON
   files, but it cannot support a general SOTA claim.

3. The A4/A5/A6 development values used in design documents are not linked to
   checked-in result JSON files or registry bindings:
   - object F1 0.401 and dynamic F1 0.0095;
   - 200 identities, 47 one-frame tracks, 48 dynamic identities, 469 geometry
     epochs, and 1556 lifecycle transitions;
   - 1.288/1.424 m fused-center errors, 0.559/0.266 m current-center errors,
     and the 996 to 1489 diagnostic TP change.

These values may remain explicitly labeled as development diagnostics, but they
must not enter a submitted result table or quantitative claim until source-bound.

### Major

1. `BASELINE_RUN_REPORT.md:14` identifies the 20260719 class-agnostic result as
   the current OVIV2 Table 1 source. The registry actually binds the CROVE/OVIV2
   row to the 20260721 Route 1 result and the ScanNet Stage 4 result.

2. The reproduction command at `BASELINE_RUN_REPORT.md:42` still imports the
   superseded 20260719 result and omits the current 20260721 and ScanNet result
   sources. Re-running it would not reproduce the checked-in populated table.

3. Public naming is inconsistent. The populated tables display `CROVE`, while
   the story, static summaries, result metadata, figure prompt, and stable token
   prefix use `OVIV2`. The stable internal key may remain `OVIV2`, but the paper
   needs one explicit alias policy and one public display name.

4. The story targets AAAI 2026, while the current writing/review target is AAAI
   2027. Event-specific policy and format must therefore be re-established from
   the final target event rather than inherited from the old specification.

5. Table 1 describes all entries as native predictions, while the CROVE result
   summary describes Route 1 as a composition of a frozen Stage 3 map, semantic
   replay, an instance head, and geometry stabilization. The caption or mode
   must disclose this evaluation path precisely enough to make baseline
   comparability reviewable.

6. The pipeline-figure prompt still presents the old OVIV2 dual-readout story
   and A0-A4 evidence chain. It omits A5/A6 role-conditioned association,
   geometry epochs, identity-qualified motion, and the proposed evidence
   admissibility principle. It must not be used for the final paper unchanged.

### Verified Numeric Statements

- CROVE Replica-8 0.338 mIoU, 0.414 mAcc, 0.580 f-mIoU, 0.371 AP25,
  0.125 AP50, and 0.886 F@5cm match the verified Route 1 result after rounding.
- CROVE Replica-7 0.332 mIoU and 0.124 AP50 match the same result.
- CROVE ScanNet200-5 0.317 mIoU, 0.281 AP25, 0.081 AP50, and 0.760 F@5cm
  match the verified Stage 4 result after rounding.
- The reported OpenFusion, ConceptGraphs, DualMap, and same-hardware T4 values
  match their registry-bound checked-in result files.
- The Replica semantic deltas against OpenFusion and instance/geometry gaps
  against ConceptGraphs in `OVIV2_REPLICA_RESULTS.md` are arithmetically correct.

## Citation Metadata Findings

- No manuscript bibliography or BibTeX database was supplied, so duplicate
  keys, author metadata, DOI/arXiv fields, venue strings, and citation rendering
  cannot be audited.
- The literature report uses primary-source links for the 15 retained works and
  clearly marks OASIS-Map as an under-review arXiv work rather than an accepted
  publication.
- The paper package must add bibliography records for at least Khronos,
  DynaMem, Where Did I Leave My Glasses?, DualMap, POCD, OASIS-Map, OVI-MAP,
  and Consistent Instance Field before citation integrity can pass.

## Citation-Context Findings

- The old story's broad related-work grouping no longer supports the intended
  novelty claim. DynaMem and Where Did I Leave My Glasses? belong in the
  Introduction because they establish the task itself.
- Consistent Instance Field prevents claiming the first separation of
  visibility/existence and persistent identity.
- DualMap's abstract/concrete maps have planning/detail roles; they must not be
  conflated with CROVE's cumulative/current readouts.
- OVI-MAP supports the static mapping foundation only and cannot support a
  temporal maintenance claim.

## Severity

- Submission blockers: missing central T2/T3 evidence, no complete manuscript,
  no bibliography, and unresolved core novelty positioning.
- Major: stale reproduction report, method-name drift, outdated AAAI target,
  ambiguous Table 1 protocol wording, and obsolete pipeline figure.
- Minor after blockers close: normalize display labels, result aliases, units,
  rounding language, and internal-versus-public identifiers.

## Safe Edit Suggestions

1. Keep `OVIV2` only as the stable internal token/result key and use `CROVE` as
   the public paper name, with one explicit alias declaration in artifact docs.
2. Designate `benchmark_tables_baselines.md/.tex` as the populated paper-facing
   view and `benchmark_tokens.tsv` as its sole provenance authority.
3. Update `BASELINE_RUN_REPORT.md` to the actual current registry source set and
   derive its reproduction command from verified registry rows.
4. Replace the old current-state novelty sentence with the evidence-admissibility
   claim only after the mechanism specification is approved.
5. Source-bind all development diagnostics or remove their numbers from
   paper-facing prose.
6. Regenerate the pipeline figure prompt from the approved mechanism; do not
   patch the old dual-readout graphic incrementally.
7. Keep all dynamic superiority language disabled until formal T2 and causal
   ablation results are imported through the registry.

## Verification Performed

```text
verify_oviv2_dual_readout_development_gates.py: PASS
tests/oviv2/test_t1_noninterference.py: 8 passed
tools/benchmark_tables.py check: PASS
tests/test_benchmark_table_package.py: 20 passed
JSON scorecard parse and AAAI score recomputation: PASS
git diff --check: PASS at audit time
```

## Next CCFA Owner

- `ccf-experiment-designer`: freeze the mechanism-specific T2/T3 evidence
  matrix and baseline/ablation contract after design approval.
- `ccf-paper-writer`: rewrite the paper story, related work, and public naming
  only after the approved mechanism and real result boundaries are fixed.
- `ccf-submission-checker`: check AAAI 2027 format, anonymity, bibliography, and
  artifact package when the manuscript exists.

No result, delta, citation acceptance status, or SOTA conclusion was invented.
