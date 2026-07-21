# OVIV2 Main-Paper Story Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the OVIV2 AAAI manuscript so dynamic current-map correctness is the single central claim and the main benchmark tables form its evidence ladder.

**Architecture:** Preserve the existing LaTeX project, method equations, citations, and verified static numbers. Rebuild the narrative around accumulated-state failure, signed visibility, reversible ownership, dense semantic recovery, and explicit geometric diagnostics. Organize experiments as static capability calibration, TESSE-CD dynamic evaluation, causal ablation, and online efficiency; exclude supplementary query, temporal-identity, and calibration protocols from the main-paper story.

**Tech Stack:** AAAI 2026 LaTeX, `natbib`, existing `references.bib`, local CCFA paper-writing guidance, source-level shell checks.

---

### Task 1: Load The Writing Contract

**Files:**
- Read: `/home/ww/.codex/skills/ccf-paper-writer/references/venue-guides/index.md`
- Read: `/home/ww/.codex/skills/ccf-paper-writer/references/venue-guides/aaai.md`
- Read: `/home/ww/.codex/skills/ccf-paper-writer/references/exemplars/index.md`
- Read: `/home/ww/.codex/skills/ccf-paper-writer/references/length-budget-policy.md`
- Read: `/home/ww/.codex/skills/ccf-paper-writer/references/output-style-policy.md`
- Read: `/home/ww/.codex/skills/ccf-paper-writer/references/citation-workflow.md`
- Read: `/home/ww/.codex/skills/ccf-paper-writer/references/prose-quality-guardrails.md`
- Read: `/home/ww/.codex/skills/ccf-paper-writer/references/research-writing-patterns.md`
- Read: `/home/ww/.codex/skills/ccf-paper-writer/references/section-modules.md`
- Read: `/home/ww/.codex/skills/ccf-paper-writer/references/writing-checklists.md`

- [ ] **Step 1: Read every required writing reference completely**

Record the venue limit, abstract expectations, section roles, claim-evidence constraints, and relevant AAAI exemplar cards. Do not introduce new citations because this rewrite does not add a new related-work family.

- [ ] **Step 2: Freeze the terminology and evidence boundary**

Use `dynamic open-vocabulary semantic mapping`, `current-state correctness`, `current map`, `visible absence`, `occluded`, `current voxel ownership`, `ghost surface`, `background recovery`, and `static capability calibration`. Exclude state-query, 3RScan temporal-identity, and absence-calibration claims from the main narrative.

### Task 2: Rebuild The Front Half Around Current-Map Correctness

**Files:**
- Modify: `docs/paper/oviv2_aaai/main.tex:21-70`

- [ ] **Step 1: Replace the title**

Use a title that names the task and central mechanism: `OVIV2: Causal Current-State Maintenance for Dynamic Open-Vocabulary Semantic Mapping`.

- [ ] **Step 2: Rewrite the abstract**

Use this five-move order: current-state requirement; failure of positive-only accumulation; separation and signed-evidence insight; OVIV2 mechanism; TESSE-CD evidence with explicit pending-result markers. Mention static evaluation in one subordinate sentence without static numbers.

- [ ] **Step 3: Rewrite the Introduction**

Create five paragraphs with these roles:

1. open-vocabulary maps are useful only when they represent the current environment;
2. movement, removal, and occlusion expose the limitations of accumulated evidence;
3. the root problem is the conflation of geometry, semantics, entity evidence, and current ownership;
4. OVIV2 implements signed visibility and reversible ownership while retaining dense semantic coverage;
5. contributions cover the problem/formulation, causal method, and main-benchmark evidence ladder.

Do not report Replica numbers in the Introduction. Describe static experiments only as a prerequisite calibration.

- [ ] **Step 4: Realign Related Work**

Keep the existing citation keys and three subsections. Rewrite each closing comparison around current-map maintenance: accumulated semantic maps lack explicit state revocation; object-centric maps do not by themselves distinguish visible absence from occlusion; dynamic maps motivate change handling but leave open-vocabulary current-map correctness as the target gap.

- [ ] **Step 5: Expand the Problem Formulation**

Define the map at time `t` as independent geometry, semantic/entity evidence, derived current ownership, and entity registry. State that evaluation concerns the current semantic/geometric state after each intervention. Define visible absence, occlusion, and unobserved regions before their use in the method.

- [ ] **Step 6: Verify front-half consistency**

Run:

```bash
rg -n 'Replica-8 reaches|held-out Replica|state query|NOT_FOUND|3RScan|temporal identity' docs/paper/oviv2_aaai/main.tex
```

Expected: no static headline sentence and no supplementary-task claim in the Abstract or Introduction.

### Task 3: Align The Method With The Dynamic Failure Chain

**Files:**
- Modify: `docs/paper/oviv2_aaai/main.tex:71-145`

- [ ] **Step 1: Rewrite Overview and invariants**

Explain each state by its temporal role: TSDF geometry records observed surfaces; dense evidence maintains semantic coverage; entity evidence aggregates compatible observations; current ownership is revocable. State that this separation makes stale-state removal possible without semantic lifecycle decisions rewriting geometry.

- [ ] **Step 2: Retain the geometry, dense, association, and memory mechanisms**

Preserve equations, frozen parameter values, and citation keys. Revise their opening and closing sentences so they support the dynamic failure chain instead of presenting static semantic fusion as the destination.

- [ ] **Step 3: Strengthen signed visibility and reversible ownership**

Make this the conceptual center of the method. State the conditions for present, visible absence, occluded, and unobserved states; explain why only visible absence creates negative evidence; connect ownership release to stale semantic-state suppression and dense recovery. State explicitly that the verified runtime has no semantic-triggered TSDF de-integration.

- [ ] **Step 4: Reframe cross-layer fusion**

Present uncertainty-aware fusion as preserving semantic coverage after ownership changes, not as the paper's primary novelty. Keep the existing fusion equation and frozen scale.

- [ ] **Step 5: Generalize implementation details**

Describe room0 as the development split used to freeze globally reused parameters. Remove language implying that Replica-7 and ScanNet200 are the endpoint of the study.

- [ ] **Step 6: Verify mechanism terminology**

Run:

```bash
rg -n 'present|visible absence|occluded|unobserved|negative evidence|reversible|background recovery' docs/paper/oviv2_aaai/main.tex
```

Expected: every central mechanism appears in both formulation/method and later dynamic evaluation.

### Task 4: Rebuild Experiments Around The Main Benchmark Tables

**Files:**
- Modify: `docs/paper/oviv2_aaai/main.tex:146-237`

- [ ] **Step 1: Replace Datasets and Protocol with Evaluation Questions and Protocols**

Introduce five questions: static foundation, official dynamic change quality, stale semantic-state release and geometric recovery, causal mechanism attribution, and online cost. Describe Replica/ScanNet first but explicitly label them static capability calibration. Describe TESSE-CD Apartment as the development sequence and Office as held out. State common causal inputs, frozen validation-only thresholds, official GT usage only during evaluation, and macro/per-sequence reporting.

- [ ] **Step 2: Compress static evidence**

Keep one compact static table with the verified Replica and ScanNet values already present. Merge geometry interpretation into the same subsection. State only that OVIV2 enters dynamic evaluation with competitive semantic and geometric quality; retain the weaker instance AP as a boundary, not a second static result story.

- [ ] **Step 3: Add the headline TESSE-CD table**

Create a two-panel `table*`:

- official Apartment/Office object, dynamic-object, and change F1;
- common current mIoU, ghost rate, background F@5cm, and recovery frames.

Use rows for frozen OVI-MAP, frozen ConceptGraphs, DualMap, Panoptic Mapping with shared masks, Khronos open-set, Khronos GT-semantics oracle, and OVIV2. Mark every unavailable number explicitly and avoid comparative prose until results exist.

- [ ] **Step 4: Add dynamic interpretation**

Explain what each metric establishes: official F1 measures dynamic/change recognition; ghost rate diagnoses stale geometry; background F@5cm tests revealed-space recovery; recovery frames tests delay. Do not infer geometric removal from ownership release or any dynamic success from Replica.

- [ ] **Step 5: Replace the static ablation with a causal dynamic ablation**

Use rows `Positive-only base`, `+ Signed visibility`, `+ Reversible ownership`, and `+ Dense background recovery`. Use columns static mIoU, change F1, ghost rate, background F@5cm, and recovery frames. State the predicted failure addressed by each row, while leaving all unavailable cells explicitly marked. Add a separate geometry-reclaim row only if that mechanism is implemented.

- [ ] **Step 6: Add compact efficiency evidence**

Use a compact main-paper table with total seconds/frame, processed Hz, maintenance seconds/frame, peak GPU memory, peak RAM, and map size. Separate frontend, backend, maintenance, finalization, and evaluation I/O in prose. Do not mix paper-reported hardware values into ranked same-hardware columns.

- [ ] **Step 7: Replace static qualitative placeholders**

Specify a dynamic sequence figure with pre-change state, intervention, first revisit, and recovered map. Compare a frozen map, positive-only OVIV2, and full OVIV2; highlight stale geometry and revealed background.

- [ ] **Step 8: Verify main-paper scope**

Run:

```bash
rg -n 'OV-CurrentMap|3RScan|NOT_FOUND|Presence AUROC|risk-coverage|ID switches|Reactivation' docs/paper/oviv2_aaai/main.tex
```

Expected: no matches in the main manuscript.

### Task 5: Rewrite Discussion, Reproducibility, And Conclusion

**Files:**
- Modify: `docs/paper/oviv2_aaai/main.tex:238-261`

- [ ] **Step 1: Rewrite Discussion and Limitations**

Lead with what the main benchmark is designed to test. Bound the claim around incomplete TESSE-CD cells. Discuss proposal misses, false absence under partial visibility, delayed observation of vacated space, pose assumptions, large frontends, and weak static instance partitions. Remove the old statement that the paper is currently supported mainly by static results.

- [ ] **Step 2: Generalize Reproducibility**

Retain immutable manifests, hashes, deterministic ordering, and frozen settings. Add TESSE-CD sequence/split identifiers, lifecycle-threshold provenance, intervention checkpoints, per-sequence reporting, and evaluator-command requirements.

- [ ] **Step 3: Rewrite Conclusion**

Conclude with the distinction between accumulated history and the current map, the signed-evidence/reversible-ownership principle, and the dynamic evidence categories. Do not mention Replica scores or supplementary protocols.

- [ ] **Step 4: Update the reserved appendix sentence**

Reserve per-sequence TESSE-CD results, evaluator definitions, additional dynamic visualizations, failure cases, sensitivity analysis, licenses, and reproduction commands. Do not introduce the ignored supplementary tasks.

### Task 6: Run Source-Level Verification

**Files:**
- Verify: `docs/paper/oviv2_aaai/main.tex`

- [ ] **Step 1: Check scope and legacy naming**

Run:

```bash
rg -n 'OVIOVO|OV-CurrentMap|3RScan|NOT_FOUND|Presence AUROC|risk-coverage|ID switches|Reactivation' docs/paper/oviv2_aaai/main.tex
```

Expected: no matches.

- [ ] **Step 2: Check static-result placement**

Run:

```bash
rg -n '0\.337|0\.576|0\.331|0\.317|0\.882|Replica|ScanNet200' docs/paper/oviv2_aaai/main.tex
```

Expected: verified values are confined to static calibration, implementation/reproducibility context, or limitations; none appears in the Abstract, Introduction contributions, or Conclusion.

- [ ] **Step 3: Check claim-evidence terms**

Run:

```bash
rg -n 'TESSE-CD|current mIoU|ghost rate|background F@5cm|recovery frames|Signed visibility|Reversible ownership' docs/paper/oviv2_aaai/main.tex
```

Expected: dynamic benchmark terms appear in the Abstract/Introduction promise, Experiments evidence, and Conclusion boundary as appropriate.

- [ ] **Step 4: Check LaTeX source hygiene**

Run:

```bash
git diff --check -- docs/paper/oviv2_aaai/main.tex
```

Expected: exit status 0.

- [ ] **Step 5: Compile when an engine is available**

Run:

```bash
cd docs/paper/oviv2_aaai
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: PDF build succeeds. If `latexmk` and `pdflatex` are unavailable, report source-level verification only and do not claim compilation success.
