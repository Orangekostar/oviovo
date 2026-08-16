# CROVE A5 Role-Separated Identity Implementation Plan

## Task 1: Restore role-separated spatial gates

Files:

- `tests/oviv2/test_temporal_association.py`
- `src/oviv2/temporal_association.py`

Steps:

1. Replace the active-wide-gate regression test with a failing test requiring
   ACTIVE/UNCERTAIN targets to remain inside the local association radius even
   when their appearance passes dormant re-ID qualification.
2. Keep a separate test proving a qualified DORMANT target can use the wider
   re-ID radius.
3. Remove active-role use of `maximum_reid_distance_m`.
4. Run the complete temporal association test file.

## Task 2: Verify runtime admission and motion controls

Files:

- `tests/oviv2/test_temporal_runtime.py`
- no production change unless an existing contract fails

Steps:

1. Verify `confirm_hits=2` prevents one-frame proposals from allocating a
   persistent temporal identity.
2. Verify the existing dynamic-state implementation honors the selected
   three-frame, 0.15 m, 0.8-confidence settings.
3. Add tests only where the contract is not already covered.

## Task 3: Materialize one A5 diagnostic

Artifacts:

- A4 base configuration plus the four A5 settings from the design document.
- Separate output root labeled nonformal development evidence.

Steps:

1. Record base config hash, code commit, exact overrides, schedule hash, and
   input bindings before launch.
2. Run Apartment with the causal schedule and no future-frame access.
3. Export common-v2 and official Khronos-format metrics.
4. Compare identity count, singleton tracks, dynamic entities, geometry epochs,
   object/dynamic/change F1, current mIoU, ghost, background F@5cm, and recovery.

## Task 4: Promotion or stop

1. Stop if object F1 or current mIoU falls below A0.
2. Stop if fragmentation falls but dynamic/change evidence does not improve.
3. If Apartment passes, freeze the exact A5 configuration and run Office once.
4. Only formal immutable runs may update paper tables.

## Verification Commands

```bash
python -m pytest -q tests/oviv2/test_temporal_association.py
python -m pytest -q tests/oviv2/test_temporal_runtime.py
python scripts/evaluation/verify_oviv2_dual_readout_development_gates.py \
  --verify-source-manifest \
  configs/evaluation/manifests/oviv2_t1_transitive_sources_v1.json
git diff --check
```
