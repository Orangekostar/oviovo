# Minimal instance repair implementation plan

**Goal:** Implement and execute the exact nine-arm, 26-scene study, paired cold timing, three tables, requirement review and verified publication.

**Architecture:** A narrow binder consumes the published parent and actual v2 lineage. A pose-selected, label-independent full-mesh observer supplies immutable source-row masks to local structural proposals and incumbent rereading. GT remains confined to diagnostics and partition-specific evaluation. All numerical rules come unchanged from the archived protocol.

**Execution:** Primary agent implements algorithms, adapters, experiment decisions and integration directly, using the executing-plans and test-driven-development workflows. The user's explicit autonomous instruction supplies authorization; no additional design confirmation is needed. Mechanical delegation is optional and restricted by AGENTS.md.

**Tech stack:** NumPy/SciPy/Open3D 0.19 CPU controller; inherited FC Python/PyTorch 2.4 FP32 worker; pinned ordinary AnyUp and released evaluator; existing PDF renderer.

**Spec:** `docs/paper/static_ovmap/minimal_instance_repair_v1/spec/CODEX_FINAL_EXECUTION_EN.md`, `IMPLEMENTATION_CONTRACTS.md`, `EVALUATION_AND_SELECTION.md`, `PROTOCOL_SPEC.json`.

## Constraints

- Base `247e1e9782d43e882589bd9ab0015d513c200e49`; branch `research/ovimap-minimal-instance-repair-v1`; no parent or deployment edits.
- Exactly IR00–IR08, Replica8 and CF18: 234 scene rows, 18 ordered full pools; reuse 52 verified baseline rows and four pools.
- Fixed xyz/faces/TSDF/projection; structural partitions may change; original incumbent rows and raw-zero rows never move.
- No new maps, segmentation, N/Q inference, parameter fits, backbones, GT activation or threshold changes.
- Observer: representative-based 0.20 m/15° pose bins, maximum 32 frames, alternating banks; full mesh nearest-hit barycentric source vertex; original depth tolerance and integer rays.
- Whole residual units, exact support distances, shared capped proposal library, disjoint noncascading edits and stable fresh IDs.
- IR06/07 share FULL candidate and equal cosine aggregation; IR07 additionally tests distinct CORE/intersection observations. IR08 uses original-incumbent anchored changes.
- New COMBO builder; evaluator mask roots use actual owner-array digest and all positive owners. Keep all five metrics and loaded overlap floats.
- Models required by each measured arm only; four arms × eight scenes × two reverse-order repeats, all measured through final payload.
- One GPU worker, four raycast threads, at most two observers and three evaluators with BLAS ≤4; ≥20 GiB storage preflight.
- Successful scientific observer/FC/AnyUp frame work each ≤832; extra pilot FC ≤2, AnyUp QK ≤3. Count failures, retries and cold work separately.
- One canonical result store; exactly three main MD/CSV/JSON/LaTeX tables, one rendered/inspected preview, four reports, one final requirement review; compact artifacts <50 MiB.

## Tasks

1. **Bind and observe.** `binding.py` exposes `bind(...)`, `load_binding(root)` and `load_scene(binding, scene)`; `support_units.py` constructs inherited K/full P units and physical areas; `observations.py` exposes pose bank selection, source-hit mapping, unit votes and `observe_scene(...)`. Tests exercise nonchaining pose grouping, floor subsampling, maximal-barycentric ties, full-scene occlusion, neutral/unknown votes and physical support area. Run `--phase bind`, then observe every scene; missing files retain explicit dependency blocks.
2. **Propose and diagnose.** `proposals.py` computes AABB/KD-tree exact neighbors and the common positive-consensus library; `verification.py` locks the four structural slates. Tests exercise clique completeness, positive-only generation, held-view abstention, shared priority, host conflict and fresh IDs. Generate all 26 GT-free libraries before `diagnostics.py` opens targets and reports R/K/P/library opportunity and relaxed matching.
3. **Recognize and reread.** `recognition.py` selects shared union views and original low-margin incumbents, binds exact masks and executes one inherited FC/AnyUp worker; `reread.py` deduplicates same-frame masks and implements both exact class rules. Pilot office1/scene0011_00 validates real operators and identities; reuse unchanged prior ordinary AnyUp parity. Commit complete implementation/spec freeze before full scientific recognition; cache leaves by exact consumed identities and preserve successful work.
4. **Construct and evaluate.** `outputs.py` applies disjoint whole-unit operations once, leaves unedited G1 rows exact, recomputes ranks, and saves locked COMBO payloads. Tests verify KEEP identity, host preservation, union ID collision avoidance, one class per owner and semantic-only fixed masks. `evaluation.py` builds per-partition expanded registries using original target/source-manifest bindings; test same owner with changed support gets a different mask root. Evaluate missing scientific rows with three CPU workers and pool the exact ordered cohorts.
5. **Select and time.** `selection.py` implements all ten nondecrease inequalities and CF D2 APall/AP50 targets with epsilon and prescribed lexicographic tie-break; tests reject a single decreased metric and strict-AP equality. `timing.py` replays cache-cold G1 plus each arm's required work, unloads unrelated models, measures five exclusive stages and both absolute/incremental allocator peaks, then verifies payload parity outside the timer. Freeze arm/order plan only after metric selection and run all 64 calls.
6. **Report and render.** `reporting.py` creates one canonical store with all five pooled metrics, per-scene/rank/diagnostic/cost provenance, five mechanism pairs, and measured timing rows (unmeasured null). Generate exactly three tables in four formats and four named reports; render and inspect one preview PDF using existing tooling. No GT coordinates or licensed scan imagery in Git.
7. **Review and publish.** Inspect each explicit spec item against actual files, production-boundary tests, both pilots, scientific coverage, accounting caps, timing parity and rendered output. Write one requirement review, commit and normal-push the prescribed branch, compare full HEAD to ls-remote, and write external `publication/final.json` only on exact equality. Final response supplies four report links, SHA, coverage, target flags, mechanism outcomes, actual computation and any precise blocks.

## Verification entry points

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest tests/static_ovmap/test_minimal_instance_repair.py -q
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_minimal_instance_repair.py --spec configs/static_ovmap/minimal_instance_repair_v1.json --parent-root /mnt/shared/ww/ovimap-evidence-exploration-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001 --phase all --resume
git ls-remote origin refs/heads/research/ovimap-minimal-instance-repair-v1
```

Status is recorded in external phase receipts, not inferred from this plan. Completion requires the full final command and requirement-by-requirement evidence.
