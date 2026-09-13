# Complete query capture for subsequent native runs

The native mapper retains all executed queries in `main`'s `inst_dict`, then writes only
the area-ranked top ten per object to its normal semantic cache. Its current build does
not write the commented-out temporary feature files. A normal rebuild alone therefore
does not recover the missing observation bank.

`run_ovimap_native.py --capture-query-history` now wraps the exact mapper source with one
callback appended to the normal end of `main`. The original mapper file is unchanged;
its SHA is checked before execution. Sources with explicit returns/yields are rejected
instead of guessing where to capture. The callback runs after the native cache is saved.
It changes no query, view-selection, VLM, geometry or native retention operation.

The separate `query_history/full_query_cache.pkl` contains every executed query's original
frame, feature, pose, bbox and area, in per-owner execution order, plus native owner colors.
`history_receipt.json` checks every retained field against the exact original area argsort
and records native-to-full indices, source/cache hashes and capture time. The scene
manifest includes both additional files. A mismatch fails the run rather than silently
labelling an inconsistent bank complete.

B0 still consumes the original retained cache. Taking last8 of the new full cache would
be a different baseline, because execution order differs from the native area-sorted
retained order. Full-history S1 readout must preserve the recorded identity bridge and
report its larger candidate pool separately from the completed retained-top10 experiment.

Validation: 27 query-capture/native-runner tests pass. A real Room0 two-frame mapping smoke
using the existing predicted masks and geometry exits 0 and captures all 29 queries from
18 owners. Capture takes 0.095923 s; the entire smoke takes 46.400 s under shared-machine
load. The hook itself adds zero image queries. The smoke does perform its own native
two-frame encoding and is excluded from all benchmark/query-budget comparisons.
This short smoke has no discarded queries; the deterministic test separately covers
12 executed versus 10 retained queries and rejects a corrupted retained feature.

Small evidence is in `artifacts/static_ovmap/query_history_capture_smoke`. Large outputs
are under `/mnt/shared/ww/ovimap-static-20260913/history_capture_smoke`.
Office2 is the first full scene launched with capture enabled. The already running
Room0/office0/office1 jobs are unmodified and remain retained-cache runs. Capturing new
runs does not retroactively recover the historical Room0 172 discarded query identities.
Full-scene history readout and validation remain pending.

## Paired full-history readout

`run_static_ovmap_readout.py` supports opt-in `S1a_FULL`, `RANDOM8_FULL`, `QUALITY8_FULL`
and `ALL_VIEWS_FULL` alongside the unchanged retained-cache conditions. Supply
`--full-query-cache`, `--history-receipt` and `--full-enrichment`, while `--native-cache`
and `--native-binding` still describe the original retained top-ten baseline.
The receipt must bind both actual cache hashes and the native mapper source hash.
Enrich the full cache using the existing enrichment script before running these conditions.

The reader validates every original observation against its full-history index, keeps
original IDs for B0, and names full-pool observations `owner:full:index`. The mapping is
saved in `input_binding.json`. No full-pool condition changes B0's order or eligibility.
K remains eight except for the explicitly different-budget `ALL_VIEWS_FULL` control.
These are larger-candidate-pool experiments, not silently substituted S1/C1 results.

The real two-frame captured bank passes paired versus standalone B0 equality for all
18 owners (11 eligible). All four full-pool conditions execute with the measured quality
of 29 observations and add zero image calls. This verifies the interface, not a full-scene
gain. Evidence and exact argv are under `artifacts/static_ovmap/full_history_readout_smoke`.
16 targeted history/readout/enrichment tests pass, including an invalid bridge rejection.

Fresh text encoding binds the rebuilt image-encoder source to a separate feature-space
identifier: `sha256:f26a6f0eafda897bcfb5ac2b2319f8747a02425c7e39017bec9454317d3f1713`.
The existing text-cache script ran on CPU in 20.636 s including model load. This avoids
assigning the historical source identifier to a different implementation. No historical
and rebuilt feature vectors are mixed. The capture smoke's original source hashes refer
to commit d3a9898; the later readout-only identity bridge does not alter its executed hook.
