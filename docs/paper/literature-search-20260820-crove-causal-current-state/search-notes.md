# Search Notes

## Scope

- Mode: standard.
- Purpose: novelty grounding, closest-baseline discovery, idea optimization,
  and reviewer-risk diagnosis.
- Target: AAAI model-method paper at the AI/ML/CV and robotics boundary.
- Search date: 2026-08-20.
- Private manuscript wording was not used in public queries.

## Safe Queries Used

- online dynamic open-vocabulary semantic mapping RGB-D;
- current-state object memory moved appeared disappeared;
- visibility occlusion absence object lifecycle semantic mapping;
- persistent identity dynamic scene understanding;
- multi-session object association partial views semantic correspondence;
- online dynamic spatial memory open-vocabulary object search;
- static and dynamic readout non-interference semantic map;
- 3RScan moved-object association and evolving-scene reconstruction.

## Sources Checked

- Robotics: Science and Systems proceedings;
- CVF Open Access;
- arXiv full papers and stable records;
- official project pages for venue and artifact status;
- DBLP only for publication metadata confirmation.

Twenty-three candidates were screened. Fifteen were retained in the scored
table. Secondary screened works were LOST-3DSG, FindAnything, HOV-SG,
Fusion++, MaskFusion, Co-Fusion, ReFusion, and RIO/3RScan. They remain useful
background or protocol references but are not the closest mechanism set.

## Excluded Sources

- Policy-excluded publishers and venues were removed from the candidate and
  citation lists.
- Search snippets, third-party summaries, inaccessible papers, and untraceable
  PDFs were not used for paper claims.
- Workshop-only LOST-3DSG was screened but not prioritized over the stronger
  directly overlapping conference and journal papers.

## Evidence Boundaries

- Numerical results are reported only when stated in a primary paper and are
  not compared across incompatible protocols.
- OASIS-Map is an arXiv preprint marked under review on its project page; no
  acceptance status is inferred.
- The absence of a paper using the exact phrase “typed evidence routing” is not
  proof of novelty. The proposed route still needs mechanism-level comparison
  against Khronos, DynaMem, Where Did I Leave My Glasses?, POCD, and OASIS-Map.
- Existing CROVE Apartment proxy values are development diagnostics, not paper
  results.

## Handoff Notes

### For idea optimization

- Retire “online open-vocabulary current-state mapping” as the novelty claim.
- Center the method on state/evidence compatibility and causal provenance.
- Treat role-separated association, visible absence, reversible ownership, and
  dual-center readout as consequences of one inference rule.
- Consider bounded multi-hypothesis identity only if the minimal evidence-routing
  method cannot improve official dynamic/change metrics.

### For experiments

- Add direct naive alternatives: single-frame ray deletion, one shared identity
  gate, one fused centroid, and no reversible ownership.
- Add adversarial stress tests for full occlusion, partial view, same-class
  lookalikes, non-rigid motion, and reappearance at a displaced pose.
- Preserve Apartment-only development and one-shot Office transfer.
- Add 3RScan moved/static association only after the TESSE-CD method is frozen.

### For writing

- State the task as established by prior work.
- Use a mechanism-level contribution sentence and an evidence-admissibility
  matrix in the Method section.
- Avoid “first,” generic “dual map,” and generic visibility/identity separation.

### For review

- Current decisive risks are core novelty overlap and missing formal T2/ablation
  evidence.
- A reviewer should require direct comparisons or faithful simple baselines for
  DynaMem-style deletion and Where-My-Glasses-style stationarity.

## Checklist Status

- Public queries: complete.
- Source exclusions: applied.
- Candidate deduplication: complete.
- Venue/source/type/relevance metadata: complete.
- Quality scoring: complete under CCFA anchors.
- Closest-work gaps and rescue route: complete.
- Reusable report folder: complete.
- Novelty verdict: conditional; requires implementation and evidence.

## Repository-to-Claim Audit

This audit distinguishes behavior already present in the repository from the
unimplemented publication-level mechanism.

| Candidate claim element | Current repository evidence | Remaining gap |
| --- | --- | --- |
| Occlusion is not absence | `temporal_lifecycle.py` makes occluded, out-of-view, and depth-unknown evidence neutral; lifecycle and runtime tests cover it | Evidence provenance and admissible mutation targets are not expressed through one shared contract |
| Active continuation and dormant re-ID have different roles | `temporal_association.py` keeps active/uncertain matches inside the local gate and requires qualified appearance/semantics before the wider dormant gate | The role rules remain association-local rather than consequences of a general state/evidence factorization |
| Rejected motion cannot contaminate retained geometry | `temporal_runtime.py` starts a new epoch after qualified rejected motion and otherwise avoids integration; temporal epoch tests enforce purity | Temporal export still reads the fused entity centroid, so the geometry-safe state is a biased current-position estimator |
| Object/background ownership is reversible | `temporal_background_ledger.py` stages, commits, and reclaims provenance-bound ownership records | Ownership evidence and lifecycle evidence are audited separately rather than under a common witness schema |
| Cumulative static mapping is protected | T1 protected-source and non-interference gates bind the frozen cumulative path | This is a strong guardrail but not a dynamic novelty claim |
| Current pose and cumulative centroid are separate readouts | The dual-center design has a complete causal specification | It is not implemented: `temporal_runtime.py` still uses `_centroid(entity)` for observed `TemporalExportSample` values |
| Identity-qualified active displacement complements weak geometry | Qualified association diagnostics and dormant re-ID dynamic promotion already exist | The active branch still advances dynamic state only from geometric motion confidence |

The highest-leverage behavior change is therefore the already diagnosed
dual-center plus identity-qualified active-motion rule. The publication-level
optimization is to make that rule, visible absence, role-separated identity,
epoch purity, and reversible ownership instances of one explicit admissibility
model. A bounded multi-hypothesis identity buffer should be added only if the
same-class false-re-ID stress test remains a dominant error after this minimal
mechanism is measured.
