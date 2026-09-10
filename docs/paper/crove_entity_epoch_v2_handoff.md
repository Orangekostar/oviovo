# CROVE Entity-Episode Dynamics V2 Handoff

## Delivery identity

- Branch: `research/crove-entity-epoch-dynamics-v2`
- Evidence base: `d5c0688bc662f8e65455cb9c62909de87c941b43`
- Evaluated implementation commit: `98b35fbd9f4a9c5a61322f03683c00e1699e0b74`
- Frozen-confirmation and deterministic-render commit: `863c95394c9bd40c5dfa7623c20bf8d78caa5ebd`
- Final compact evidence commit: `56883ed3d4662b9469d9da4fc36caf61d1ade9cf`
- Configuration: `configs/evaluation/crove_entity_epoch_v2.json`
- Compact package: `configs/evaluation/results/crove_entity_epoch_v2/`

The real DEV run used the implementation later committed as `98b35fb`; changes between execution and that commit were formatting, portable public paths, structured missing-confirmation handling, and figure export. No prediction, state-transition, relation, selection, or headline evaluator rule changed.

## Status ledger

| Component | Status | Evidence |
| --- | --- | --- |
| CODE | PASS | D0-D6 runner, entity-episode kernel, G1, ReScene, memory, composition, attribution |
| RESULTS | PASS_DEV_ONE_PAIR | Eight hypothesis rows; selected `H1_B3 / D1_B3` |
| MODEL | NOT_APPLICABLE_REUSED | One exact pair-bound forward; zero training |
| MAP | LOCAL_ONLY_POLICY | Canonical PLY/NPZ and per-variant sidecars remain outside Git |
| CONFIRM | RAW_MISSING | Office database, change labels, and RGB-D export manifest absent |

## Reproduction

```bash
python scripts/evaluation/run_crove_entity_epoch.py \
  --config configs/evaluation/crove_entity_epoch_v2.json \
  --phase prepare --split dev

python scripts/evaluation/run_crove_entity_epoch.py \
  --config configs/evaluation/crove_entity_epoch_v2.json \
  --phase run --split dev

python scripts/evaluation/run_crove_entity_epoch.py \
  --config configs/evaluation/crove_entity_epoch_v2.json \
  --phase summarize --split dev

python scripts/evaluation/render_crove_entity_epoch_results.py \
  --config configs/evaluation/crove_entity_epoch_v2.json

python scripts/evaluation/run_crove_entity_epoch.py \
  --config configs/evaluation/crove_entity_epoch_v2.json \
  --phase run --split confirm \
  --selection configs/evaluation/results/crove_entity_epoch_v2/selected_config.json
```

## Local artifacts

- Canonical OVI current-map directory: `$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/apartment_two_visit_full_v1/current_map/`, approximately 4.4 GB.
- Canonical surface NPZ: 271,722,745 bytes, SHA-256 `7d3ad160bdee68fb9054826c1e949b153983d2f744cb412c63f6a4bfde927367`.
- Canonical RGB/instance/semantic/state PLYs: 1,098,664,155 bytes each; their hashes are bound by `current_surface_manifest.json`.
- Entity-episode run root: `$HOME/oviovo_baseline_runs/20260910_crove_entity_epoch_v2/`, approximately 1.3 GB.
- Selected B3 state NPZ: 115,869,640 bytes, SHA-256 `04919d7f00e40db11ae6054e63b4334aa41cd2df3e969a91c31c6972fadfc72e`.
- D2 inherited-update state NPZ: 117,183,103 bytes, SHA-256 `2ff6c5104737f4859d7c53abdd4acd9b9e0c41c120269146c1a7ff20a005e011`.
- A pre-metric-fix interrupted run is retained only under the local `hypotheses.invalid-pre-incremental-deletion-metric/` directory and is excluded from all compact scans and indexes.

## Evidence boundaries

- Compact artifacts contain no machine-specific absolute paths; 17 files are SHA-256 bound by `compact_artifact_index.json`.
- Final verification: 60 focused entity-episode tests and 215 affected V1/T1 regression tests pass; Ruff, compilation, and `git diff --check` pass.
- The repository-wide privacy scanner still reports legacy absolute paths that predate this branch; a diff-scoped scan reports none in this delivery.
- The normalized deletion/retention columns use prior-valid, currently supported t0 rows as the denominator. The compact CSV retains the raw legacy receipt values in separate `raw_*` columns.
- `H7_MEMORY_BANK` is not a real multiview result because only one final OVI support observation exists.
- Office has not been evaluated. Materialize the frozen Office assets, then rerun the recorded confirmation command without changing thresholds.
- This package does not update aggregate T1-T4 tables and does not provide official Obj/Dyn/Chg values.
