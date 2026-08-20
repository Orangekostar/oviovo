# AAAI Review Simulation: CROVE After Core Implementation, Before Final Results

Diagnostic simulation only. This is not an official AAAI score, acceptance
probability, or multi-reviewer consensus.

## Review Scope

- Paper type: model-method.
- Materials reviewed: method specification, implementation, focused tests,
  benchmark registry, provenance gates, and closest-work report.
- Unavailable materials: complete manuscript, appendix, final Apartment and
  Office T2 package, and matched mechanism ablations.
- Simulated reviewer profile: not provided.
- External verification: primary closest-work sources and repository evidence
  were checked.

## Paper Summary

CROVE addresses current-state open-vocabulary mapping under object motion,
occlusion, disappearance, and reappearance. It factorizes persistent identity,
existence, current pose, dynamic state, geometry epoch, ownership, temporal
readout, and a frozen cumulative map. A closed admissibility matrix restricts
each causal evidence type to compatible state mutations. The implementation
adds qualified identity motion, protects persistent prototypes from weak
matches, and exports current observations without changing the cumulative T1
path. The unresolved question is whether these restrictions produce a
competitive and uniquely attributable T2 gain over simpler visibility and
state-update rules.

## Strengths

1. The method now has a coherent technical principle rather than a component
   list.
   - Evidence: state-compatible routing design; `temporal_evidence_router.py`.
   - Why it matters: the matrix yields explicit, testable mutation boundaries.

2. The implementation is fail-closed and causally scoped.
   - Evidence: exhaustive router tests, runtime behavior tests, prefix replay,
     and source/provenance gates.
   - Why it matters: weak identity, occlusion, and rejected motion cannot be
     silently promoted into incompatible state updates.

3. Static non-interference is unusually well controlled.
   - Evidence: protected-source verifier and T1 non-interference tests; the
     current focused suite passes 225 tests.
   - Why it matters: dynamic gains can be evaluated without trading away the
     established cumulative map.

## Weaknesses

### Critical

**C1. The distinct novelty consequence remains unestablished.**

- Primary dimension: Novelty.
- Location/evidence: closest-work report and unfilled ablation table.
- Why it affects the decision: SuperMap already combines online open-vocabulary
  tracking, visibility states, pruning, reactivation, and semantic fusion.
  CROVE's narrower typed-mutation distinction is credible but not yet shown to
  cause a benefit unavailable to an untyped update.
- Resolution condition: compare matched untyped-routing, geometry-only, and
  fused-center variants and report the predicted failure classes.
- Gate: `CORE_NOVELTY_UNESTABLISHED`.

**C2. The central empirical result is incomplete.**

- Primary dimension: Evidence.
- Location/evidence: T2/T3 placeholders in `benchmark_tables_baselines.md`;
  optimized Apartment evaluation still running; Office held out.
- Why it affects the decision: the central claim concerns dynamic current-state
  quality, so implementation and unit-test evidence cannot substitute for
  official Obj./Dyn./Chg. F1 and current-map quality.
- Resolution condition: pass the preregistered Apartment promotion gate, freeze
  the method, run Office once, and source-bind the formal tables.
- Gate: `DECISIVE_EVIDENCE_MISSING`.

### Major

**M1. The final paper story is not yet reviewer-visible.**

- Primary dimension: Clarity.
- Evidence: the unified formulation exists in a design specification, while a
  complete manuscript and updated pipeline figure are unavailable.
- Impact: a reviewer may still encounter the older component-centered story.
- Suggested action: make the state tuple, admissibility matrix, and three
  prohibited cross-state mutations the Method and figure backbone.

**M2. Closest-work positioning has not entered the manuscript.**

- Primary dimension: Related Work.
- Evidence: the literature artifact includes SuperMap and mechanism-level
  comparisons, but no complete paper text is available.
- Impact: broad novelty wording would be contradicted by direct prior art.
- Suggested action: state that current-state mapping is established and claim
  only the typed evidence-admissibility contract plus frozen cumulative branch.

### Minor

| # | Primary dimension | Location | Issue | Suggestion |
| --- | --- | --- | --- | --- |
| 1 | Reproducibility | Final package | Frozen result bindings are pending | Publish commit, config, algorithm, and artifact hashes after promotion |
| 2 | Clarity | Pipeline figure | Existing prompt predates the full router | Regenerate it from the state/evidence matrix |
| 3 | Evidence | Ablations | Stress cases are specified but not populated | Include occlusion, lookalike, reappearance, and fused-center failures |

## Seven-Dimension Scorecard

| Dimension | Status | Weight | Score | Evidence / Location | Main Concern |
| --- | --- | ---: | ---: | --- | --- |
| Significance | ASSESSED | 15% | 4.5 | Story specification; literature report | Important but established task |
| Novelty | ASSESSED | 20% | 3.0 | Routing design; router implementation; SuperMap comparison | Unique consequence lacks matched evidence |
| Soundness | ASSESSED | 20% | 4.5 | State factorization; closed matrix; runtime tests | End-to-end failure boundaries pending |
| Evidence | MISSING_IN_PAPER | 25% | 2.0 | T2/T3 placeholders; running Apartment evaluation | Central result and ablations incomplete |
| Clarity | ASSESSED | 8% | 4.0 | Unified design specification | Complete manuscript and figure unavailable |
| Related Work | MISSING_IN_PAPER | 7% | 3.5 | Primary-source closest-work report | Not integrated into manuscript |
| Reproducibility | ASSESSED | 5% | 5.0 | Source, causality, replay, and T1 gates | Final result bindings pending |

## Scientific Rating

- Raw weighted score: 3.490/6.0, rounded to 3.5.
- Coverage: 100%, complete for the supplied implementation-stage package.
- Plausible range after gates: 2.5-2.5/6.0.
- Active scientific gates:
  - `CORE_NOVELTY_UNESTABLISHED`, cap 2.5. Resolution: direct mechanism
    contrast and matched ablations.
  - `DECISIVE_EVIDENCE_MISSING`, cap 2.5. Resolution: frozen T2/T3 evidence.
- Final Overall Score: 2.5/6.0.
- Recommendation: Borderline.

Implementation raises Soundness and makes the novelty candidate concrete, but
neither active gate can be removed before the running evaluation and targeted
ablations finish.

## Assessment Confidence

- Material completeness: 1/2. Core code and design are available, but final
  results and a complete manuscript are not.
- Verification depth: 2/2. Implementation contracts, tests, provenance gates,
  and closest primary sources were checked.
- Reviewer-profile domain match: 0/1. No explicit profile was supplied.
- Confidence: 3/5, Medium.

The score is a reliable diagnosis of the current package, not a review of the
eventual frozen submission.

## Compliance / Policy Status

- Format: NOT_CHECKED.
- Anonymity: NOT_CHECKED.
- Event-specific policy: NEEDS_POLICY.
- Notes: no final paper PDF or event package was available. These statuses are
  separate from the scientific score.

## Ethical / Safety Concerns

None identified within the reviewed materials.

## Questions & Suggestions

1. Does typed routing beat an otherwise identical untyped visibility/log-odds
   update under the same frontend?
2. Which observed failure reduction comes from prototype isolation, routed
   motion, and dual-center readout individually?
3. Does the optimized branch pass the preregistered Obj./mIoU/ghost guardrails
   while improving Dyn. or Chg. F1 by at least 0.01?
4. Can the final paper show exact cumulative non-interference alongside the
   temporal gain without presenting T1 as the main novelty?

## Rebuttal Priorities

| Priority | Concern | Can rebuttal address it? | Required evidence |
| ---: | --- | --- | --- |
| 1 | Missing central T2 result | Unlikely | Frozen Apartment and Office metrics |
| 2 | SuperMap mechanism overlap | Partially | Exact contrast plus matched untyped-routing ablation |
| 3 | Attribution across routes | Unlikely | Geometry-only, fused-center, and prototype-update ablations |
| 4 | Manuscript positioning | Fully | Narrow contribution wording and mechanism comparison table |

## Summary for AC

CROVE now presents a coherent typed evidence-routing mechanism with strong
causal, provenance, and static non-interference controls. The implementation is
technically convincing, but the decisive dynamic results and ablations are not
yet available, and SuperMap creates substantial direct overlap that requires a
narrow, experimentally differentiated claim. The current diagnostic score is
therefore 2.5/6.0 with Medium confidence, capped by novelty and evidence gates.

Final Overall Score: 2.5 / 6.0 — Borderline
Assessment Confidence: 3 / 5 — Medium
