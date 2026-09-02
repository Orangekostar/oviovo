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
| P1 visualization contract | `CODE_EVIDENCE` + tests | `rgb`, `instance`, `semantic`, and `dynamic` are explicit recoloring-only modes; output geometry and IDs remain invariant | **GO** | `d724822ef3b77f924381985d22862eb0b3ead535` | Visualization is separated from formal MapSnapshot/evaluator state |
| P2 Ghost authority attribution | `MEASURED_EVIDENCE` | 2,152,474/2,152,474 official ghost matches are partitioned; unbound anchors contribute 74.002752%; bound dynamic-false and epoch-not-advanced states contribute another 24.639% | **GO** for P4A and one P5 visibility pilot | `812928e`, `87bfcd3` | Ghost and dense-to-sparse branches have exact or direct measured support |
| P2 Dyn mechanism attribution | `CODE_EVIDENCE` + bounded diagnostics | Official Dyn has 19,556 missed and 22,431 hallucinated counts, but the evaluator does not expose runtime entity correspondence | **NO-GO** for P3 threshold/weight tuning | `812928e`, `87bfcd3` | Re-ID/ICP mechanism counts cannot be relabeled as official Dyn failures |
| P4A dense moved-anchor shadow | `MEASURED_EVIDENCE` + tests | Five target objects recover 1,937,312 dense points from 3,502 compact points; all state fields match and maximum rigid residual is `8.81e-7 m`; `ovimap:2` has zero 5 cm compact agreement | **GO** for a separately named Apartment evaluation candidate; **NO-GO** for formal promotion before metrics | `903a934`, `a6a835e`, `336d1f8` | The density discontinuity is removed, but one large translated template has a measured geometry mismatch that must be evaluated rather than assumed beneficial |
| P4A official Apartment gate | `MEASURED_EVIDENCE` | Obj F1 0.353859, Chg F1 0.086534, Current mIoU 0.148174, Ghost 0.443852; Obj F1 delta -0.018903 against the registered floor | **REJECTED_RETAIN_A6**; use compact geometry for P5 | `34a35ab` + bound external artifacts | Dense translation improves three metrics but fails the preregistered Object F1 floor |
| P5 causal unbound-anchor visibility | `CODE_EVIDENCE` + `MEASURED_EVIDENCE` | Ten unbound anchors become dormant from causal depth evidence; Obj F1 0.367952, Chg F1 0.088088, Current mIoU 0.149563, Ghost 0.441677; Obj F1 delta -0.004810 | **REJECTED_RETAIN_A6**; Office remains blocked | `a040319`, `21dde04`, `2a8f610`, `fd19df6` + bound external artifacts | The stale-anchor mechanism is validated, but the single preregistered candidate misses the Object F1 floor |

## Artifact Binding

- P0 JSON SHA-256:
  `aabacf1f232cf394579e3822ff96fbe7351892af7b7105637b86f99c86c7a45c`.
- P0 Markdown SHA-256:
  `49d2ab77a1ee4313ed6e9c38d226ed49718753082473606d9728dcefd157cc74`.
- P2 attribution JSON SHA-256:
  `2e3d47cbdb2bc627a886e843d843f95d6a7bdd806db93d66a9c6a8da1382b7bc`.
- P2 source runtime diagnostics SHA-256:
  `4e3eef84499bc5ad991d553fc514a071b0000b9cb62ce812e34de0165c9efdc5`.
- Repeated runs were byte-identical.
- P4A compact-control manifest SHA-256:
  `10bb0edcd2d89175b606e25ba465b931795ab7b68347914357695b27d22e5e82`.
- P4A dense-shadow manifest SHA-256:
  `71d03749ca162711fdada6af7ed33dccd452fa6ef51d7e51a7a9dc6d7dd397a5`.
- P4A final audit JSON SHA-256:
  `6bae5c9aae684af4e6e0ddd0885d312832f7c8df3244d08176933c63e74acb3f`.
- P4A official metrics SHA-256:
  `2a7f251973d63444ccf833ad956eb25c112a62fd4eb7d1ff8699327ea8dac657`.
- P4A gate decision SHA-256:
  `59e961e8852f5687efd6719ba6d2fb1a0613eb4baf02c28be7bf6c4d02766c47`.
- P5 composition manifest SHA-256:
  `3cfa389183a930ac5fb10aa34093cf63e64e1c8206f1f4e9ccffef1490369fc3`.
- P5 visibility diagnostics SHA-256:
  `7ee9a287a3484147bf8daeb02105a2829046bb357dee4aec1fcfb66ac090ddf3`.
- P5 official metrics SHA-256:
  `901fe7587f2b224819d3ed4838a4bfe225ed3c361e51914e4acd18093bda7a21`.
- P5 gate decision SHA-256:
  `b3b4db3ea564e5c4aec821c37d3362cae0fbee677bea0238df226c29123d6f12`.

## Active Gate

P4A and P5 are rejected by the preregistered Object F1 floor, and A6 remains
the formal official-metric candidate. P3 association changes, P4B new-object
dense readout, and P6 ReScene integration remain blocked by the attribution
evidence. No new configuration is frozen, so Office remains held out and was
not run.
