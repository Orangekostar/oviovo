# OVIV2 Benchmark-Centered Paper Story Design

## Objective

Rebuild the AAAI manuscript around one scientific claim: an online open-vocabulary semantic map should represent the current scene state rather than an irreversible accumulation of past observations. OVIV2 is the proposed method; the main benchmark suite is the evidence spine that tests this claim through controlled scene changes, stale semantic-state release, revealed-background recovery, geometric ghost diagnostics, causal ablations, and online efficiency.

Replica and ScanNet200 are not headline evidence. They are a prerequisite calibration showing that the map has competitive static semantic and geometric quality before dynamic maintenance is evaluated.

## Fixed Scientific Position

- **Paper type:** method paper with a benchmark-centered evidence structure.
- **Target venue:** AAAI 2026.
- **Problem:** causal dynamic open-vocabulary semantic mapping from posed RGB-D streams.
- **Fundamental question:** how can an online map remain semantically consistent with the current world after objects move, disappear, or become occluded, without retaining stale state or damaging static map quality?
- **Root failure of prior accumulation:** positive-only fusion can add evidence but cannot reliably revoke stale state; treating missing observations as removal confuses occlusion with absence; irreversible object ownership entangles identity, semantics, and geometry.
- **Core insight:** current-state correctness requires signed, visibility-grounded evidence and reversible state. Geometry, semantic belief, entity identity, and current voxel ownership must be maintained as related but distinct quantities.
- **Method consequence:** independent TSDF geometry, dense semantic evidence, persistent entity posteriors, signed visibility, reversible ownership, and dense semantic recovery after ownership release. The verified runtime has no semantic-triggered TSDF de-integration.
- **Central bounded claim:** within posed indoor RGB-D streams and the selected main benchmark protocols, OVIV2 provides a causal mechanism for keeping open-vocabulary maps aligned with the currently observable scene after environmental changes. The final strength of this claim is determined only by completed dynamic results.

## Rejected Storylines

1. **Static-fusion-first story.** Rejected because Replica mIoU validates semantic map quality but does not test stale-state removal or background recovery.
2. **Benchmark-paper story.** Rejected because the principal technical content is the OVIV2 mapping method; the benchmark is its evidence structure rather than the sole contribution.
3. **Equal static and dynamic story.** Rejected because it creates two competing main claims and makes the dynamic problem appear to be an optional extension.

## Causal Story Arc

1. Open-vocabulary maps enable robots to query previously unspecified concepts.
2. Existing maps mainly accumulate observations and therefore answer what has ever been seen, not necessarily what is currently true.
3. Dynamic scenes expose three coupled failures: stale object semantics remain current, occlusion is confused with absence, and old geometry can obstruct recovery of newly revealed background.
4. These failures arise because geometry, semantics, identity, and ownership are updated as if they were one persistent state.
5. OVIV2 separates these states and uses signed visibility evidence to release current ownership, after which dense evidence labels the currently reconstructed surface. Ghost geometry remains an empirical diagnostic unless an explicit geometry update is added.
6. The main benchmark then tests the method in the same causal order: static foundation, physical change, current-map quality, mechanism attribution, and online cost.
7. The conclusion states only what completed benchmark results support and treats missing evidence as explicit result slots rather than inferred success.

## Claim-Evidence Ladder

| Order | Reviewer question | Evidence | Supported claim | Failure indicator |
| --- | --- | --- | --- | --- |
| 1 | Does OVIV2 begin from a usable map? | Replica-8/7 and ScanNet200-5 semantic and geometric evaluation | Dynamic maintenance is not compensating for a weak static mapper | Low mIoU or geometric F-score |
| 2 | Does the map recognize changed regions and objects? | Official TESSE-CD Apartment/Office metrics | OVIV2 improves dynamic open-vocabulary map maintenance | Object, dynamic, and change F1 |
| 3 | Does the current map release stale semantics, and what geometric errors remain? | Common current-map metrics on TESSE-CD | Signed visibility and reversible ownership improve semantic current-state quality; ghost and background metrics bound geometric behavior | Current mIoU, ghost rate, background F@5cm, recovery frames |
| 4 | Which mechanisms cause the gains? | Cumulative causal ablation | Negative evidence, ownership release, and dense recovery address distinct semantic failures; geometric effects require separate evidence | Static mIoU guardrail, change F1, ghost rate, background F@5cm, recovery frames |
| 5 | Is the online cost bounded? | Same-hardware efficiency evaluation | Lifecycle maintenance has measurable online overhead | throughput, latency, memory, map size |

Static quality is intentionally first but receives the least narrative emphasis. Its role is to establish a floor, not to prove the central claim.

## Section-Level Redesign

### Title

Retain OVIV2 but make the title name the dynamic current-state problem. “Causal Dual-Layer Fusion” alone overemphasizes architecture and under-specifies the scientific goal.

### Abstract

Use five moves: current-state problem, accumulation failure, signed/reversible insight, compact method, dynamic benchmark evidence. Mention static evaluation in at most one subordinate sentence and include no static numbers. Preserve explicit result slots for missing dynamic benchmark names, principal comparisons, and the final evidence-bounded conclusion.

### Introduction

Use five paragraphs:

1. Open-vocabulary mapping is useful only if queries describe the current world.
2. Explain the accumulation failure through relocation, removal, and occlusion.
3. Identify the root representational problem: geometry, semantic belief, identity, and ownership have different temporal meanings.
4. Introduce the signed-evidence and reversible-state insight, then summarize OVIV2 as its implementation.
5. State contributions as problem/formulation, method, and benchmark-driven evidence. Static experiments appear only as a prerequisite calibration, not as a contribution or headline result.

### Related Work

Organize around the gap rather than method families alone:

- open-vocabulary maps mainly optimized for accumulated semantic coverage;
- dynamic and long-term maps model change but do not necessarily maintain open-vocabulary semantics for the current scene state;
- visibility-aware representations require an explicit distinction between observed free space, occlusion, and missing observations.

End each subsection by identifying the unresolved current-map maintenance requirement that OVIV2 addresses. Remove wording that narrows the paper to static semantic mapping or expands the main claim to language-query and long-term identity tasks.

### Problem Formulation

Define the output at every time step as a current-state map. Formalize causality and distinguish:

- currently present surface and semantic state;
- absent, occluded, and unobserved evidence states;
- persistent identity versus current voxel ownership.

State evaluation targets here so the benchmark follows naturally from the formulation.

### Method

Reorder and rewrite method motivation according to failure modes:

1. independent geometry prevents semantic lifecycle updates from corrupting reconstruction;
2. dense and entity evidence maintain coverage and identity;
3. signed visibility distinguishes removal from occlusion;
4. reversible ownership releases stale semantic authority;
5. dense semantic recovery labels currently reconstructed surfaces after ownership release;
6. uncertainty-aware export combines dense coverage with current entity evidence.

The method section must state that the verified runtime does not perform semantic-triggered TSDF de-integration. Ghost rate and background F@5cm are diagnostic outcomes, not guaranteed effects of ownership release. Any later explicit geometry-reclaim mechanism requires its own description and ablation row.

Implementation details may report Replica development settings, but they must be described as globally frozen mapper parameters rather than the conceptual center of the method.

### Experiments

Replace the current static-first bulk structure with the following sequence:

1. **Questions and protocols.** State the five reviewer questions represented in the evidence ladder and define common causal inputs.
2. **Static capability calibration.** Compress Replica/ScanNet semantic, instance, and geometry evidence into one subsection and, if space requires, one combined table. The interpretation is only that OVIV2 enters dynamic testing with a competitive mapping foundation.
3. **Dynamic current-map maintenance.** Make TESSE-CD the first headline table. Discuss object/dynamic/change F1, ghost rate, and recovery delay.
4. **Causal ablation.** Add signed visibility, reversible ownership, and dense semantic recovery in the same order as the method. Measure change F1, ghost rate, background F@5cm, and recovery frames; static mIoU is only a guardrail. A geometry-reclaim row is included only after that mechanism exists in the verified runtime.
5. **Efficiency and qualitative analysis.** Report lifecycle overhead and visualize the pre-change map, intervention, first revisit, and recovered map.

Supplementary protocols for open-vocabulary state queries, temporal identity on 3RScan, and absence calibration are outside the current rewrite. They do not appear in the main-paper contribution list, evidence ladder, experiment narrative, or conclusion. Main-table ablations must not depend on their metrics.

Dynamic result cells that are not yet available remain explicit manuscript placeholders. No numerical improvement or superiority claim is written until the corresponding result is supplied.

### Discussion and Limitations

Discuss where current-state maintenance fails: proposal misses, false absence under partial visibility, pose error, and delayed observation of vacated space. Static AP weaknesses are secondary limitations. Do not say dynamic behavior is merely future evidence; instead state precisely which main benchmark cells are incomplete and withhold only the associated claims.

### Conclusion

Return to the distinction between accumulated history and current state. Summarize the signed-evidence/reversible-state principle and the evidence categories. Do not lead with Replica numbers, implementation completeness, or a list of unfinished work.

## Terminology Contract

- Use **dynamic open-vocabulary semantic mapping** for the task.
- Use **current-state correctness** for the main capability.
- Use **current map** for the semantic and geometric state evaluated after each environmental change.
- Use **current voxel ownership** for the reversible spatial assignment.
- Use **visible absence** only when free-space evidence supports removal.
- Use **occluded** when a closer observed surface blocks the expected entity surface.
- Use **stale state** as the general error; use **stale false positive** and **ghost surface** only for their benchmark definitions.
- Use **static capability calibration** for Replica/ScanNet evidence.

## Required Manuscript Consistency Checks

- Abstract, Introduction, Experiments, and Conclusion state the same central claim.
- Every contribution in the Introduction maps to at least one benchmark table or ablation.
- Static results never support a dynamic claim.
- Method components are introduced in the same order as their causal ablations.
- Occlusion never produces negative evidence in the description.
- Missing numbers remain explicit result slots; no result is invented.
- OVIV2 is used consistently; legacy project names do not appear in the manuscript.
- Limitations bound the central claim without redefining the paper as static mapping.

## Completion Criteria

The rewrite is complete when the LaTeX manuscript has a dynamic current-state title and story, the experiment section follows the main-table evidence ladder, Replica/ScanNet occupy only a calibration role, supplementary query/identity/reliability protocols are absent from the main narrative, all unsupported dynamic outcomes are visibly reserved for real measurements, legacy static-centered conclusions are removed, and source-level terminology and claim-evidence checks pass.
