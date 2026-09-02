# CROVE Map-Quality Provenance Audit

## Result

P0 is **GO**. The Apartment artifact chain is complete and hash-bound, the
native PLY is proven to use OVI-MAP instance-palette colors rather than RGB
texture, and the dense-anchor to sparse-temporal geometry transition is
quantified. P0 does not evaluate or change benchmark scores.

## Reproducibility

- Frozen base: `c55396b24d7706016a569c951914c79c9f641408`.
- Audit implementation: `1f7b675d3602c195ba807475e37f677df6158980`.
- Audit script SHA-256:
  `c9f06ee4425b5348587a4f0c8207c168c02c34361a5b7c2304858d59b462aaaa`.
- Test SHA-256:
  `659d22abe78b9131d5c6b8549d921193ddc5caeb2b0942be74e63501b5390373`.
- Deterministic JSON:
  `/home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p0/map_quality_audit.formatted.json`,
  SHA-256
  `aabacf1f232cf394579e3822ff96fbe7351892af7b7105637b86f99c86c7a45c`.
- Deterministic Markdown:
  `/home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p0/map_quality_audit.formatted.md`,
  SHA-256
  `49d2ab77a1ee4313ed6e9c38d226ed49718753082473606d9728dcefd157cc74`.
- Independent repeated outputs have the same two hashes.
- Focused validation: `27 passed`; Ruff check and format check passed.

## Bound Inputs

| Artifact | SHA-256 |
| --- | --- |
| Native mapping manifest | `8d8f27d9b418527d8130766b1c21785e2794b6e7c9789c9e5528ac4af8ec4386` |
| OVI-MAP instance PLY | `8459a59cba3f59ece14a9f3462dbe0796d00c712214e558c487bde9dbbcffb57` |
| Instance color log | `2d9615434225934a144b7e87834b9c255d3cf51e144f2b326bccec642d768b41` |
| Anchor manifest | `b6a3bdb18216d2e2b0a43d76f00a57a769fddb845a8c6e310379fe6a455d84af` |
| Anchor entities | `d17dd78a101d8482bc7c4f2223d90297214072e4e511d603c85b5293e046799c` |
| Composed run manifest | `cb6f5c8858739ff861364a0e5d32735f5001849ca4e083aba430d316f6fe9983` |
| Final current entities | `83378c646b2f00de83ec22235cffe1b645e6d9145028e20b9a9d9ece5314ba16` |
| Final checkpoint diagnostics | `2cc471a962c6061b93991886d2777675a16ce7b8c8c45ef9d3d609551fc4599c` |
| `consistent_gsm` extension | `edd9be8e10e5ab88d8086bb1fc133c758dc667e3ba8dc09650aa59fc82a106ca` |
| OVI-MAP source-hashes file | `b149958d8dcbbfbffc75271ea5455adfa19d95e80c5f241e2415d1d15cdf5963` |

## Producer Chain

`CODE_EVIDENCE`: OVI-MAP commit
`58a804e2d7c82ba05a489eb071aba3367301fed8` executes the bound
`panoptic_mapping_.py`, which requests only the instance mesh from
`GlobalSegmentMap_py::generateMesh`. The bound `consistent_gsm` extension runs
the instance mesh integrator. Its source path selects
`MeshLabelIntegrator::kInstance`, obtains colors through
`instance_color_map_`, and writes `instance_mesh_263.ply` through
`outputMeshLayerAsPly`.

The audit also records current compatibility-source hashes independently from
the bound runtime binary. It does not claim that unbound source text alone
proves the executed binary.

## PLY Classification

`MEASURED_EVIDENCE`:

| Property | Value |
| --- | ---: |
| Format | ASCII PLY 1.0 |
| Vertices | 8,194,551 |
| Faces | 2,731,517 |
| Unique RGB triples | 65 |
| Registered instance colors | 63 |
| Registered instance vertices | 8,194,548 |
| Unregistered vertices | 3 |

All 63 authoritative instance colors occur in the PLY and map one-to-one to
the 63 anchor entities. The three unregistered vertices are reported as
background-or-unregistered; no stronger semantic interpretation is made.

## Geometry Authority

`MEASURED_EVIDENCE`:

| Population | Entities | Points |
| --- | ---: | ---: |
| Causal OVI-MAP anchor | 63 | 8,194,548 |
| Final unchanged anchor authority | 50 | 5,394,094 |
| Moved objects at anchor source | 5 | 1,937,312 |
| Moved objects at current temporal source | 5 | 3,502 |
| New temporal objects | 11 | 8,734 |

The moved population retains `0.0018076592722287375` of its anchor point count,
or about `0.181%`. New temporal objects have 30 to 2,629 points and a median of
294 points.

| Anchor ID | Temporal ID | Anchor points | Current points | Retained ratio |
| --- | ---: | ---: | ---: | ---: |
| `ovimap:131` | 23 | 2,912 | 818 | 0.2809065934 |
| `ovimap:179` | 24 | 7,253 | 154 | 0.0212325934 |
| `ovimap:2` | 79 | 1,681,524 | 177 | 0.0001052617 |
| `ovimap:6` | 11 | 241,957 | 2,234 | 0.0092330455 |
| `ovimap:85` | 6 | 3,666 | 119 | 0.0324604474 |

## Interpretation And Gate

`MEASURED_EVIDENCE`: confirmed moved bindings replace dense anchor geometry
with sparse CROVE temporal geometry. This is a source-authority change, not a
visualization-only color effect.

`HYPOTHESIS`: this transition is the primary cause of degraded moved/new object
surfaces. P1 may now isolate visualization semantics and P2 may instrument
Dyn/Ghost failure funnels. P3-P6 remain blocked until P2 identifies the dominant
failure contributor; P4A remains the pre-approved readout candidate if dense
moved geometry is selected for intervention.
