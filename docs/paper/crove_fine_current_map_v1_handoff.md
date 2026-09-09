# CROVE Fine Current Map V1 Handoff

## Delivery identity

- Branch: `research/crove-fine-current-map-v1`
- Evidence base: `4870b928960783e8d66e9e5e6b2c751d27889400`
- Evaluated implementation commit: `635e13b`
- Configuration: `configs/evaluation/crove_fine_current_map_v1.json`
- Compact package: `configs/evaluation/results/crove_fine_current_map_v1/`
- Model upload: `NOT_APPLICABLE_NO_NEW_TRAINING`

The delivery commit differs from the evaluated implementation only by the
result package, documentation, and an LF-only CSV writer fix with its regression
test. No prediction, selection, metric or export algorithm changed.

## Selected result

- Static Replica room0: native OVI 1 cm geometry with CROVE S2 reliable
  surface/owner fallback. mIoU 0.4479, mAcc 0.5111, f-mIoU 0.6736, AP25
  0.5626, AP50 0.4707 and F@5cm 0.9353.
- Dynamic TESSE-CD Apartment: B3 remains selected. Current mIoU 0.1359,
  ghost 0.0000, background F@5cm 0.3664 and surface F@5cm 0.4313.
- All fine dynamic recovery candidates are rejected because ghost is
  0.9226--0.9531 against the fixed maximum of 0.02.

## Reproduction

Run focused unit tests:

```bash
python -m pytest -q \
  tests/oviv2/test_current_surface.py \
  tests/oviv2/test_fine_current_composer.py \
  tests/oviv2/test_fine_dynamic_policy.py \
  tests/oviv2/test_fine_surface_io.py \
  tests/oviv2/test_fine_surface_validity.py \
  tests/oviv2/test_surface_semantics.py \
  tests/evaluation/test_render_crove_fine_current_views.py \
  tests/evaluation/test_run_crove_fine_current_map.py \
  tests/evaluation/test_summarize_crove_fine_current_map.py
```

Run the static and dynamic experiments when the configured local assets are
mounted:

```bash
python scripts/evaluation/run_crove_fine_current_map.py \
  --config configs/evaluation/crove_fine_current_map_v1.json \
  --case replica-static \
  --output "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/replica_room0_static_full_v1"

python scripts/evaluation/run_crove_fine_current_map.py \
  --config configs/evaluation/crove_fine_current_map_v1.json \
  --case apartment-two-visit \
  --output "$HOME/oviovo_baseline_runs/20260909_crove_fine_current_map_v1/dev/apartment_two_visit_full_v1"
```

The four large maps are under each run's `current_map/` directory. Do not add
them to Git. Use the committed fixed-view images and receipts for inspection.

## Verification state

- Compact package: 16 indexed artifacts, all hashes verified.
- New focused tests: 43 passed.
- Existing affected suites: 65 passed.
- Four static and four dynamic PLYs share exact geometry within each run.
- Compact files contain no machine-specific absolute path.

## Open evidence gaps

- D4 is not complete. No matched static confirmation OVI surface exists in the
  configured run, and the known dynamic native-run root has Apartment t0/t1
  only, not Office.
- The user-reported bad PLY was not identified; the legacy room0 files are an
  evidence-bound representative only.
- The new results do not fill aggregate T1--T4 scene averages or the TESSE-CD
  object/dynamic/change F1 cells.
- Semantic S2 improves mean accuracy but reduces f-mIoU relative to S0/S1, so
  the tradeoff must be reported rather than hidden.

Next action: materialize the frozen Office t0/t1 OVI native surfaces, bind them
to the same configuration, and run confirmation without retuning on Office.
