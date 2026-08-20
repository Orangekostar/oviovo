# AAAI Review Simulation: CROVE Before Core Optimization

Diagnostic simulation only. This is not an official AAAI score, acceptance
probability, or multi-reviewer consensus.

## Review Scope

- Paper type: model-method.
- Materials reviewed: paper-story specification, dual-center design,
  benchmark-table source, implementation contracts, tests, and literature
  report.
- Unavailable materials: complete manuscript, appendix, and final formal T2/T3
  result package.
- Simulated reviewer profile: not provided.
- External verification: closest-work papers and repository evidence checked.

## Paper Summary

CROVE studies causal open-vocabulary mapping for scenes whose object existence,
pose, and geometry change over time. Its current implementation separates
persistent identities, geometry epochs, lifecycle state, reversible background
ownership, and cumulative versus temporal outputs. Static mapping is already
competitive, but the central dynamic evidence is incomplete and the available
Apartment diagnostic has about 0.01 dynamic F1. The most important unresolved
question is whether these mechanisms amount to one novel inference principle
and produce a competitive current-state map under occlusion, partial views, and
same-class reappearance.

## Strengths

1. The problem is important for embodied agents that must act on the current
   scene rather than historical accumulation.
   - Evidence: benchmark-centered story, lines 13-18; literature report,
     Summary.
   - Why it matters: stale state directly corrupts localization, change, and
     NOT_FOUND decisions.

2. The implementation has strong causal and reproducibility discipline.
   - Evidence: T1 non-interference tests, frozen-source gates, deterministic
     provenance, and explicit evaluation placeholders.
   - Why it matters: the paper can distinguish genuine temporal improvement
     from static-map regression or protocol leakage.

3. The diagnostic work identifies a concrete readout failure.
   - Evidence: the dual-center specification reports 1.288 m and 1.424 m fused
     centroid error versus 0.559 m and 0.266 m current-pose error, and a
     diagnostic TP increase from 996 to 1489.
   - Why it matters: it gives a falsifiable mechanism-level hypothesis rather
     than relying only on aggregate metric tuning.

## Weaknesses

### Critical

**C1. The core novelty is not currently established.**

- Primary dimension: Novelty.
- Location/evidence: literature report Summary and Closest-Work Clusters;
  current benchmark-centered story.
- Why it affects the decision: DynaMem, Where Did I Leave My Glasses?, DualMap,
  and related systems already cover online open-vocabulary current-state map
  maintenance. Khronos, POCD, OASIS-Map, and Consistent Instance Field cover
  major parts of temporal state, stationarity, identity, and visibility.
- Resolution condition: formulate one state/evidence compatibility principle,
  specify how it differs from the closest mechanisms, and validate its unique
  consequences.
- Gate: `CORE_NOVELTY_UNESTABLISHED`.

**C2. Decisive dynamic evidence is missing.**

- Primary dimension: Evidence and Evaluation.
- Location/evidence: T2 and T3 in `benchmark_tables_baselines.md`; dual-center
  diagnostic specification, lines 23-35.
- Why it affects the decision: the paper's main claim is dynamic, whereas the
  final dynamic tables remain placeholders and the current development
  diagnostic is not competitive.
- Resolution condition: complete the held-out T2 result and direct mechanism
  ablations, while preserving the frozen T1 result.
- Gate: `DECISIVE_EVIDENCE_MISSING`.

### Major

**M1. The method reads as a component list rather than one technical insight.**

- Primary dimension: Technical Soundness.
- Evidence: lifecycle, role-separated association, reversible ownership, and
  dual-center readout are specified in separate documents and contracts.
- Impact: a reviewer can interpret the method as engineering aggregation.
- Required action: derive all transitions from an explicit evidence
  admissibility matrix over persistent identity, current pose/epoch, existence,
  ownership, and cumulative geometry.

**M2. The closest-work contrast is not yet reviewer-visible.**

- Primary dimension: Related Work.
- Evidence: the closest-work analysis exists only in the new literature report.
- Impact: the Introduction and Method currently cannot defend why CROVE is not
  DynaMem-style deletion plus re-identification and a second centroid.
- Required action: add a mechanism comparison against Khronos, DynaMem, Where
  Did I Leave My Glasses?, POCD, OASIS-Map, and Consistent Instance Field.

### Minor

| # | Primary dimension | Location | Issue | Required action |
| --- | --- | --- | --- | --- |
| 1 | Clarity | Existing story and method specs | `dual`, `current`, and `causal` are used at different abstraction levels | Define one state tuple and one update notation |
| 2 | Evidence | T2/T3 templates | No direct naive-alternative rows | Add single-frame deletion, shared association gate, fused centroid, and irreversible ownership |
| 3 | Reproducibility | Final result package | Final configuration/result bindings are absent | Publish the frozen configuration and artifact hashes with completed tables |

## Seven-Dimension Scorecard

| Dimension | Status | Weight | Score | Main evidence | Main concern |
| --- | --- | ---: | ---: | --- | --- |
| Significance | ASSESSED | 15% | 4.5 | Story design; literature report | Important but established task |
| Novelty | ASSESSED | 20% | 2.0 | Closest-work clusters; dual-center diagnostic | Headline overlaps prior work |
| Soundness | ASSESSED | 20% | 3.5 | State separation and causal contracts | No unified inference principle yet |
| Evidence | MISSING_IN_PAPER | 25% | 2.0 | T2/T3 placeholders; A6 diagnostic | Central dynamic result missing |
| Clarity | ASSESSED | 8% | 3.5 | Story specification | Component list obscures insight |
| Related Work | MISSING_IN_PAPER | 7% | 2.5 | New literature report | Closest contrasts not integrated |
| Reproducibility | ASSESSED | 5% | 4.5 | Gates, tests, provenance | Final paper configuration absent |

## Scientific Rating

- Raw weighted score: 3.0/6.0.
- Coverage: 100%, complete for the supplied material package.
- Plausible range: 2.5-2.5 after active gates.
- Active gates:
  - `CORE_NOVELTY_UNESTABLISHED`, cap 2.5: resolve with a formalized and
    differentiated evidence-routing mechanism plus direct ablations.
  - `DECISIVE_EVIDENCE_MISSING`, cap 2.5: resolve with competitive held-out T2
    and mechanism-specific T3 evidence without T1 regression.
- Final Overall Score: 2.5/6.0.
- Recommendation: Borderline.

The task is significant and the artifact discipline is strong, but novelty and
evidence are the two decision-dominating dimensions for this model-method paper.
Both scientific gates remain active; static T1 quality cannot discharge either
one.

## Assessment Confidence

- Material completeness: 1/2. Major design and artifacts are available, but no
  complete manuscript or final dynamic result package was provided.
- Verification depth: 2/2. Repository contracts and primary closest-work papers
  were inspected.
- Reviewer-profile domain match: 0/1. No explicit reviewer profile was supplied.
- Confidence: 3/5, Medium.

The main limitation is that this is a pre-submission package diagnosis rather
than a review of the final manuscript and frozen result set.

## Compliance / Policy Status

- Format: NOT_CHECKED.
- Anonymity: NOT_CHECKED.
- Event-specific policy: NEEDS_POLICY.
- Notes: no paper PDF or target-event submission package was available; these
  statuses do not affect the scientific score.

## Ethical / Safety Concerns

None identified within the reviewed materials.

## Questions & Suggestions

1. Which observable evidence is legally allowed to mutate each latent state,
   and what failure occurs when that rule is violated?
2. Does evidence routing outperform a faithful DynaMem-style single-frame ray
   deletion and a Where-My-Glasses-style stationarity baseline under the same
   frontend?
3. Does the dual-center rule reduce temporal localization error and dynamic F1
   error without changing any cumulative T1 bytes?
4. How does the system behave under full occlusion, partial views, same-class
   lookalikes, non-rigid motion, and displaced reappearance?

## Rebuttal Priorities

| Priority | Concern | Rebuttal sufficiency | Required evidence |
| ---: | --- | --- | --- |
| 1 | Core novelty overlap | Unlikely without method/result change | Formal evidence-routing principle and closest-mechanism ablations |
| 2 | Missing T2 evidence | Unlikely without experiments | Held-out dynamic/change metrics and failure analysis |
| 3 | Component-list perception | Partially | Unified equations, admissibility matrix, and a naive-combination ablation |
| 4 | Related-work omission | Fully | Accurate Introduction and mechanism-comparison table |

## Exit Criteria For The Next Review

These are rescore conditions, not predicted results. Meeting them would remove
both active gates and yield a conditional weighted score of approximately
4.1/6.0 under the same model-method rubric.

| Dimension | Current | Minimum rescore target | Evidence required |
| --- | ---: | ---: | --- |
| Significance | 4.5 | 4.5 | Keep the current problem boundary and embodied-use motivation |
| Novelty | 2.0 | 3.5 | Formal state/evidence compatibility mechanism, closest-work contrast, and targeted failure ablations |
| Soundness | 3.5 | 4.0 | Complete transition equations, explicit assumptions, safety invariants, and failure boundaries |
| Evidence | 2.0 | 4.5 | Frozen Apartment/Office T2, matched baselines, mechanism ablations, confidence intervals, and failure analysis |
| Clarity | 3.5 | 4.0 | One state tuple, one admissibility matrix, one pipeline figure, and traceable claims |
| Related Work | 2.5 | 4.0 | Fair mechanism-level positioning against the six closest threats |
| Reproducibility | 4.5 | 4.5 | Preserve current provenance gates and bind final code, config, and results |

The final rescore must use the completed manuscript and real frozen results.
Static quality or proxy diagnostics cannot substitute for any condition above.

## Summary for AC

CROVE addresses an important current-state mapping problem and has unusually
careful causal/provenance controls. However, the current headline overlaps
several recent methods, and the supplied package lacks the decisive dynamic and
ablation results. The resulting diagnostic score is 2.5/6.0 with Medium
confidence, capped by unestablished core novelty and missing decisive evidence.

Final Overall Score: 2.5 / 6.0 - Borderline
Assessment Confidence: 3 / 5 - Medium
