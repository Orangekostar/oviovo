# TESSE-CD T2 Publication Hardening Design

## Scope

Harden the existing TESSE-CD T2 finalizer and temporal artifact exporter without changing metric definitions or protocol schedules. The change covers frozen run identity pairing, strict F1 types, no-clobber JSON publication, and transactional temporal directory publication.

## Frozen Run Identity

Every official metrics payload and run status payload used by scene evidence or a full result must contain:

```json
{
  "run_identity": {
    "run_id": "canonical-run-id",
    "config_sha256": "lowercase-64-hex"
  }
}
```

`run_id` must match `[A-Za-z0-9][A-Za-z0-9._-]*`. `config_sha256` must be exactly 64 lowercase hexadecimal characters. A metrics/status pair must have exactly equal normalized identities.

Scene evidence validates only the paired identity because it has no separate provenance input. A full result additionally requires both scene identities to use `provenance.run_id`; each scene `config_sha256` must equal the observed SHA256 of at least one provenance config. The finalized output preserves each scene identity.

## Metric Types

Finite F1 values accept only exact Python `int` and `float` values. Booleans, strings, mappings, lists, and other coercible objects are rejected before conversion. Accepted values must remain finite and within `[0, 1]`.

## JSON Publication

Scene evidence and full-result CLI output use one sibling temporary file writer. The writer serializes, flushes, and fsyncs the temporary file, then uses a hard link to publish with atomic no-replace semantics. It fsyncs the parent directory and always removes the temporary name. Existing targets raise `FileExistsError` and are never modified; pre-publication failures leave no target.

## Temporal Directory Publication

The temporal exporter validates sources first, then creates a sibling staging directory. All checkpoint copies, trajectory generation, manifest generation, file fsyncs, and directory fsyncs happen under staging. Before publication, the final output path must not exist.

Publication reserves the final path with exclusive `mkdir`, then renames staging over that empty reservation and fsyncs the parent directory. Build or copy failures before reservation remove staging and leave no output, allowing retry. If an error occurs after reservation succeeds, the exporter raises a `publication-uncertain` error and does not delete by the final path, avoiding deletion races. A pre-existing output is never modified.

## Tests

Tests prove metrics/status run mismatches and missing identities fail, valid identities pass and remain in output, provenance bindings are enforced, and non-numeric F1 values fail. Finalizer tests cover scene/full no-clobber and injected publication failure. Temporal tests cover copy failure cleanup and retry, pre-existing output preservation, and successful deterministic publication.
