# OVIV2 Temporal Staging Recovery Proxy Design

Date: 2026-08-16

Status: approved for implementation by the user's continuation instruction

## Objective

Recover evaluator-readable temporal artifacts from an otherwise complete OVIV2
staging tree whose runner failed after checkpoint capture but before publishing
`source_index.json` and `run_manifest.json`. The recovered output is a
development diagnostic proxy only. It must never be accepted as formal run or
paper evidence.

## Inputs And Invariants

The command accepts one preserved staging root and one nonexistent destination.
The source root remains read-only. It must contain the frozen schedule,
`capture_status.json`, trajectories, frame coverage, lifecycle transitions, and
one neutral-current checkpoint for every official schedule entry.

Every consumed path must be a regular non-symlink file below the source root.
JSON is parsed with duplicate-key rejection. Scene, method, scheduled frames,
captured frames, timestamps, freeze boundaries, and all existing source records
must agree exactly. Missing, duplicate, extra, aliased, or inconsistent official
checkpoints fail closed.

## Alternatives

1. Reconstruct a formal run manifest and runtime diagnostics. Rejected because
   the missing runner phase cannot be recreated without inventing mechanism
   evidence.
2. Mutate the preserved staging tree by adding a legacy source index. Rejected
   because this would alter the best available witness of the failed execution.
3. Build a separate minimal proxy tree. Selected because it preserves the source,
   copies only evaluator-required bytes, and remains compatible with the existing
   strict temporal exporter.

## Output

The tool builds a sibling staging directory and atomically publishes:

```text
<output>/
  inputs/schedule.json
  capture_status.json
  trajectories.jsonl
  temporal_frame_coverage.jsonl
  lifecycle_transitions.jsonl
  checkpoints/<official-frame>-<timestamp>/
    checkpoint_status.json
    neutral_current/snapshot.npz
    neutral_current/entities.json
  source_index.json
  recovery_receipt.json
```

`source_index.json` uses only the exporter's legacy base fields. It deliberately
contains no `runtime_diagnostics`, `run_manifest`, `run_execution`, or
`frozen_run_identity`. Its method remains the captured CROVE/OVIV2 method label so
the exporter validates snapshot identity normally.

`recovery_receipt.json` has status
`NONFORMAL_RECOVERY_PROXY_NOT_SUBMISSION_EVIDENCE`, records the immutable source
root, exact source file witnesses, source-index witness, and recovery tool
identity. The receipt is outside the source-index schema and cannot upgrade the
proxy into formal evidence.

## Data Flow And Publication

The tool loads the official schedule and capture status, resolves each official
checkpoint by exact frame and timestamp, verifies all file identities, and copies
verified bytes into the staging tree. It then computes fresh destination-relative
records for the copied files, writes the base source index and recovery receipt,
revalidates both source and destination witnesses, fsyncs the staged tree, and
publishes with a no-clobber rename. Any failure before publication removes only
the new staging directory. Existing destinations are never modified.

The existing `export_tesse_temporal_artifact.py` is the downstream semantic
validator. A successful recovery does not bypass any snapshot, entity, lifecycle,
coverage, or checkpoint consistency check enforced by that exporter.

## Tests

Tests first establish a minimal valid preserved staging fixture and require a
complete proxy accepted by the existing exporter. Focused failures cover source
mutation, symlinks, duplicate or missing official checkpoints, timestamp and
freeze-boundary mismatch, capture/schedule mismatch, destination no-clobber, and
the absence of all formal-evidence fields. Verification runs the recovery tests,
the temporal exporter tests, `py_compile`, and `git diff --check`.
