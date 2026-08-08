# OVIV2 Dual-Readout A4 Recovery Design

## Scope

Recover one externally killed A4 development run without mutating the failed A0-A4 search, rerunning successful candidates, changing the predeclared search matrix, or weakening any packaging and selection gate. The recovered evidence remains development-only Apartment evidence; Office remains bound and unexecuted until a winner is frozen.

## Immutable Inputs

The recovery accepts exactly three source files:

- the production A0-A4 search manifest;
- one original search status containing A0-A3 Apartment PASS records followed by one A4 Apartment FAIL record;
- one retry search status containing exactly one A4 Apartment PASS record.

Both search roots remain read-only. The original A4 failure must be process exit `-9`, have empty stdout and stderr files, have no run identity or input hashes, and have no semantic failure reason. This identifies the observed external SIGKILL rather than admitting arbitrary failed experiments.

The retry is produced by the existing `run_search(..., candidate_ids=("a4",))` API in a fresh output root. It materializes the same manifest-declared A4 config and uses one GPU. Existing output roots are never resumed or overwritten.

## Equivalence Gates

Before recovery, the merger snapshots and validates all source JSON and evidence files as regular, non-symlink files. It requires:

- both statuses bind the same manifest bytes, Apartment base config, and Office config;
- Office is marked unexecuted in both statuses;
- candidate IDs and status shapes are exact, with no unscheduled candidates;
- original and retry A4 config bytes and canonical hashes are identical;
- A0-A3 and retry A4 configs match their manifest declarations;
- all five candidates share one non-temporal config hash and one input-binding-values hash;
- every PASS record binds an existing run manifest with matching algorithm hash, input hash, source bindings, and code commit;
- all five successful runs share the same code commit and source bindings;
- config, stdout, and stderr file records match freshly read bytes.

Any mismatch fails closed before creating an output root.

## Composite Output

Recovery creates a new output root containing:

```text
<output>/
  search_status.json
  candidates/a0/apartment/
  candidates/a1/apartment/
  candidates/a2/apartment/
  candidates/a3/apartment/
  candidates/a4/apartment/
```

The composite `search_status.json` is PASS and contains the original A0-A3 records plus the retry A4 record. Their paths continue to point to immutable source artifacts; heavyweight run directories are not copied. A `recovery` field records exact byte/hash/path witnesses for both source statuses, the replaced candidate, the accepted original failure signature, and the strategy identifier `immutable_single_candidate_retry_v1`.

Candidate result packaging writes new `result.json` files under the composite candidate directories. The existing tuner can then enforce its canonical `candidates/<id>/apartment/result.json` layout without changing metric definitions.

## Publication

The merger builds the complete tree in a sibling staging directory. Immediately before publication it revalidates every source witness, exclusively reserves the destination directory, renames staging over that empty reservation, and fsyncs the parent. Existing destinations are never modified. Pre-publication failures remove staging and leave no output.

## Tests

Tests cover the exact successful recovery, all failure-signature fields, A4-only retry enforcement, config/Office/code/source binding mismatches, source mutation during publication, output no-clobber behavior, and cleanup after injected publication failure. The recovered status must pass the existing result packager's search-status checks without special cases.
