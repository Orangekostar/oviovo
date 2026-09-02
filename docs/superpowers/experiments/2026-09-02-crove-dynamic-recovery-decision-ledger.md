# CROVE Dynamic Recovery Decision Ledger

## Frozen Contract

- Base commit: `c55396b24d7706016a569c951914c79c9f641408`.
- Development scene: Apartment.
- Held-out scene: Office; no run before final configuration freeze.
- Runtime inputs: current and past observations only.
- T1 path and protected sources: unchanged.

## Decisions

| Phase | Evidence | Result | Decision | Commit | Reason |
| --- | --- | --- | --- | --- | --- |
| P0 provenance and map quality | `CODE_EVIDENCE` + `MEASURED_EVIDENCE` | Native PLY producer and runtime binary are hash-bound; 63 palette instances match 63 anchors; five moved objects fall from 1,937,312 to 3,502 points; 11 new objects total 8,734 points | **GO** for P1/P2 | `1f7b675d3602c195ba807475e37f677df6158980` | Provenance is exact and the dense-to-sparse authority transition is quantified without benchmark modification |

## Artifact Binding

- P0 JSON SHA-256:
  `aabacf1f232cf394579e3822ff96fbe7351892af7b7105637b86f99c86c7a45c`.
- P0 Markdown SHA-256:
  `49d2ab77a1ee4313ed6e9c38d226ed49718753082473606d9728dcefd157cc74`.
- Repeated runs were byte-identical.

## Active Gate

P1 visualization-contract work and P2 causal failure attribution are
authorized. P3 association changes, P4 dense readout, P5 ghost handling, and P6
ReScene integration remain blocked until P2 produces ranked, source-bound
failure contributors. Office remains blocked until Apartment configuration
selection is complete and frozen.
