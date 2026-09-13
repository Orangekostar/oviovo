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
