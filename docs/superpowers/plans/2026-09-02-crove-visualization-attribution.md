# CROVE P1/P2 Implementation Plan

> Scope: visualization-only and instrumentation-only. Do not modify T1
> protected sources or read Office outputs.

## Task 1: P1 labeled-PLY visualization exporter

Files:

- Create `src/evaluation/crove_map_visualization.py`.
- Create `scripts/evaluation/export_crove_map_visualization.py`.
- Create `tests/evaluation/test_crove_map_visualization.py`.

Steps:

1. Write failing tests for the four modes, missing provenance/sidecars,
   deterministic palettes, uniform instance colors, and preservation of XYZ,
   IDs, confidence fields, faces, and arbitrary extra elements.
2. Implement strict input validation, stable palettes, atomic publication, and
   a hash-bound manifest.
3. Run the focused test, Ruff, `py_compile`, and `git diff --check`.
4. Export all supported views for a bound Apartment labeled mesh if one exists;
   otherwise record the capability limitation without fabricating RGB.

## Task 2: P2 runtime observation proxy

Files:

- Create `src/evaluation/crove_runtime_attribution.py`.
- Create `tests/evaluation/test_crove_runtime_attribution.py`.

Steps:

1. Write failing tests proving association and motion funnels are complete and
   wrapped execution is state/result equivalent to direct execution.
2. Implement a fail-closed context manager around exact temporal runtime
   symbols, with single-call forwarding and unconditional restoration.
3. Serialize deterministic per-frame causal records without ground truth.
4. Run focused tests and the protected temporal regression suite.

## Task 3: P2 Apartment diagnostic launcher

Files:

- Create `scripts/evaluation/run_crove_tesse_attribution.py`.
- Create `tests/evaluation/test_run_crove_tesse_attribution.py`.

Steps:

1. Write failing hermetic tests around runner dependency wrapping and output
   identity.
2. Launch the existing v2 Apartment development runner through the proxy and
   publish the diagnostic JSONL beside, never inside, the formal run root.
3. Run `--help` before the real diagnostic execution.

## Task 4: P2 overlay and ghost authority analyzer

Files:

- Create `scripts/evaluation/attribute_crove_tesse_failures.py`.
- Create `tests/evaluation/test_attribute_crove_tesse_failures.py`.

Steps:

1. Write failing synthetic tests for every authority bucket, background tie
   ownership, 5 cm matching, exact partitions, and tamper detection.
2. Implement hash-bound loading of the composed/source/anchor runs and target
   package, then emit deterministic JSON and Markdown.
3. Run `--help`, analyze the existing Apartment artifacts, and cross-check the
   aggregate ghost count against the official evaluator.

## Task 5: P2 root-cause gate

Files:

- Create `docs/superpowers/reports/P2_ROOT_CAUSE_REPORT.md`.
- Update the recovery Decision Ledger.

Steps:

1. Merge causal runtime and offline authority evidence.
2. Rank R1-R4 with measured counts, coverage, and representative IDs.
3. Mark P3/P4/P5 GO or NO-GO individually.
4. Re-run the T1 source verifier and protected regression suite; commit the
   phase only after both pass.
