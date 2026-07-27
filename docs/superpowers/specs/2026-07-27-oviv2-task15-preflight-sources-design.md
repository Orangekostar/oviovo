# OVIV2 Task15 Preflight Sources Design

## Scope

Task15 must produce source-backed Apartment launch evidence before the A0-A4
search starts. The implementation closes two blockers:

1. The development exact-profile transaction currently requires a final freeze
   manifest, while the final freeze requires the T1 evidence produced by that
   transaction.
2. `build_oviv2_tesse_search_preflight.py` consumes a candidate-source index,
   but no tracked producer creates that index, future-leakage evidence, and the
   per-candidate temporal-occlusion results from real exact-run artifacts.

Office execution, metric selection, T4 collection, and final freeze behavior
are outside this change.

## Rejected Approaches

- A provisional or fabricated `FROZEN` manifest is rejected because it would
  assert T1/T4 and selection evidence that does not yet exist.
- Reusing an older freeze is rejected because its code, config, evidence, and
  output-root bindings cannot match the current transaction.
- Writing derived evidence into exact-run roots is rejected because it changes
  the roots whose inventory and digest establish T1 equality.

## Development Transaction

The exact transaction is explicitly an unfrozen Apartment development mode.
Its spec inventory remains the fixed sequence
`reference,a0,a1,a0,a2,a0,a3,a0,a4`, but each spec contains exactly:

```json
{
  "profile": "a0",
  "config": "/absolute/materialized-profile-config.json",
  "output_root": "/absolute/independent-run-root",
  "source_manifest": "/absolute/oviv2_t1_transitive_sources_v1.json"
}
```

The transaction launches the Apartment runners without `--freeze-manifest`
or `--run-slot`. `run_oviv2_t1_reference.py` gains an explicit development
entry point that delegates to the existing v1 runner with both freeze values
set to `None`. The ordinary frozen runner entry point remains unchanged.

Every execution receipt records an exact execution-context binding:

```json
{
  "mode": "apartment_development_unfrozen",
  "config_sha256": "<sha256>",
  "algorithm_hash": "<sha256>",
  "profile_sha256": "<sha256>"
}
```

Common input fingerprints are derived from the bytes at the config-declared
input-manifest, schedule, and target paths. The verifier cross-checks those
records against `run_manifest.source_bindings.input_manifest`, the run
manifest schedule/target records, and the config's existing target SHA before
accepting them. It rejects Office configs, freeze flags, noncanonical argv,
changed config/source files, PID reuse, aliased roots, or any disagreement in
common inputs. Final freeze validation continues consuming and independently
revalidating this exact T1 transaction.

## Preflight Source Bundle

Add `scripts/evaluation/build_oviv2_tesse_search_preflight_sources.py`. Its
public API consumes:

- the verified development-gate evidence;
- the search manifest and Apartment base config;
- the TESSE-CD target manifest and dataset root;
- a new output bundle path.

It selects only the main candidate executions at fixed transaction positions
`a0=1`, `a1=2`, `a2=4`, `a3=6`, and `a4=8`. It reopens every observation,
run manifest, source index, checkpoint index, checkpoint status, compact
metadata, coverage record, and cache/source-frame binding before publication.
The selected profile must agree with the materialized search declaration and
the exact execution record.

The bundle is independent of the exact-run roots:

```text
<bundle>/
  a0/ ... a4/
    run_manifest.json
    source_index.json
    trajectories.jsonl
    lifecycle_transitions.jsonl
    temporal_frame_coverage.jsonl
    runtime_diagnostics.json
    temporal_occlusion_result.json
    future_leakage.json
  candidate_sources.json
  preflight.json
  publication_receipt.json
```

Source files are copied byte-for-byte by verified descriptor into the same
relative layout under each candidate staging directory. The copied run
manifest and source index therefore retain their original nested
`{path,sha256,byte_count}` records without a derived rewrite. Every record the
preflight consumer needs must resolve inside that candidate directory and
match the copied bytes. No file is created under an exact-run root.

`candidate_sources.json` has exact top-level keys
`{schema_version,manifest_id,candidates}`. Each candidate has exact keys
`{candidate_id,run_manifest,temporal_occlusion_result,future_leakage_evidence}`
and preserves A0-A4 canonical order. The producer invokes the existing
`build_preflight()` on the staged candidate-source index, so the final
`preflight.json` is validated by the same code used by the search runner.

Every `preflight.json` `source_evidence.path` is a canonical relative path
from the preflight file's parent. Absolute paths, `.`, and any `..` component
are rejected. The search consumer resolves the record from its already-opened
preflight witness directory and then applies the existing hash, inode, source
root, and pre-launch revalidation. Relative records remain valid when the
whole staging bundle is renamed to its final no-clobber destination.

## Future-Leakage Audit

An empty `records` list is published only after all causal checks pass. For
each processed frame and scheduled checkpoint, the audit verifies:

- exact processed-frame coverage `0..processed_frame_count-1`, agreement with
  `covered_frame_count`, `first_frame_index`, and `last_frame_index`, unique
  increasing frame indices, and dataset timestamps;
- cache entries identify the same source and dataset frame and never a later frame;
- checkpoint `consumed_through_frame` equals its frame and the exclusive bound
  equals `frame + 1`;
- compact metadata frame, revision, and timestamp equal the checkpoint status;
- source index, run manifest, checkpoint index, target manifest, and schedule
  records match by path, digest, and byte count;
- every protected runtime/export source still matches the development evidence
  code commit and source-manifest binding.

Any violation is emitted as a deterministic structured record and causes the
producer to fail before publication. It is never converted into a passing
preflight.

`scheduled_frame_indices` remains the checkpoint schedule and is validated
against the checkpoint index; it is never used as the expected per-frame
coverage inventory.

## Occlusion And Anchor Gate

For each candidate, the producer invokes the real temporal-occlusion evaluator
against the copied checkpoint index and the verified TESSE-CD dataset. The
result must contain exactly 66 eligible Apartment anchors and at least 53
unique mappings. Zero-overlap and ambiguous anchors remain separate counts.
The producer does not synthesize or patch the gate values.

## Publication And Failure Semantics

The output is staged under a random sibling directory and published once with
atomic no-clobber rename. Every input is revalidated immediately before the
rename. Success fsyncs files and the parent directory.

After any staging object exists, an error raises a module-level
`PreflightPublicationUncertain` containing preserved name, logical path,
parent/artifact device and inode, complete mode, and ownership classification.
No mutable-name cleanup is attempted. If creation provably did not happen,
the original exception is propagated. Existing output always raises
`FileExistsError` and is never overwritten.

## Verification

Tests must cover the unfrozen Apartment argv/schema, Office and freeze-flag
rejection, exact T1 equality, fixed execution positions, source rebinding,
future-frame/cache/checkpoint tampering, 52/66 failure and 53/66 success,
source mutation during copy, symlinks, duplicate keys, no-clobber, concurrent
replacement, fd closure, uncertain preservation, and end-to-end acceptance by
the existing preflight search consumer. Formal verification uses Python 3.10
with plugin autoload and bytecode writes disabled.
