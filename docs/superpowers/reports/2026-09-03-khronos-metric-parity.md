# Khronos Metric Aggregation Parity Audit

Date: 2026-09-03

Status: `PARITY_FOR_TESTED_POST_RELEASE_AGGREGATION`

## Question

Does `summarize_khronos_official_metrics()` change the rows, duplicate policy,
slice semantics, or final F1 values produced by the frozen post-release Khronos
plotting implementation?

This audit tests aggregation only. It does not establish that the post-release
evaluator is the implementation used for RSS 2024 Table I.

## Frozen Inputs

| Source | Frozen identity |
| --- | --- |
| Khronos repository | `MIT-SPARK/Khronos` at `63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e` |
| Upstream aggregation source | `khronos_eval/plotting/utils.py`, SHA-256 `a7c1bed4b97f8d9e27361296d00cf741ed67d18ae4d3f8e21e259fc4213d0763` |
| Numeric contract | exact row identity, `rtol=0`, `atol=1e-12` |
| A6 source manifest | SHA-256 `eab78809fab5f0f611269425f49ce9e477f4cb8314e9dcbb1d07ab4ffd7f46e5` |
| A6 bridge status | SHA-256 `bd6463102678480de8378c610d6adbf551543f2bff103d15080f5795b5e0ccfe` |
| A6 historical metrics | SHA-256 `1a7120e356c50c804baa13b3f5137fa079e16759da7992562c94535252e93f54` |
| P5 historical metrics | SHA-256 `901fe7587f2b224819d3ed4838a4bfe225ed3c361e51914e4acd18093bda7a21` |

A6 was recovered read-only from an authorized compute node by its registered
metric values and then rebound to its source, bridge, CSV, and timestamp hashes.
No external path is part of the public artifact contract.

## Method

`src/evaluation/khronos_metric_parity.py` snapshots all three official CSVs and
`map_timestamps.txt`, rejects malformed or changing sources, rejects conflicting
duplicate `(Name, Query)` rows, and validates causal query times. It then loads
the exact frozen `utils.py`, calls upstream parsing and augmentation functions,
and compares:

1. normalized row grids and duplicate collapse;
2. `4D`, `Robot`, `Query`, and `Online` slice values or matching unavailability;
3. Object, Dynamic, Change, and Background F1 aggregation; and
4. input and upstream source hashes before and after execution.

The executable command was:

```bash
python scripts/evaluation/audit_khronos_metric_parity.py "$RESULTS" \
  --upstream-checkout "$KHRONOS_CHECKOUT" \
  --upstream-commit 63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e \
  --output "$AUDIT_JSON"
```

## Results

### A6

| Metric | Local | Upstream | Absolute delta |
| --- | ---: | ---: | ---: |
| Object F1 | 0.372762155393872 | 0.372762155393872 | 0 |
| Dynamic F1 | N/A | N/A | 0 |
| Change F1 | 0.0608534568735205 | 0.060853456873520485 | 1.39e-17 |
| Background F1@0.2 | 0.6651238900562044 | 0.6651238900562044 | 0 |

Result: `PARITY`. Audit JSON SHA-256:
`4ecc68bc54f0fff56becacb4d0900fc8cbe19bc2656cd1f213ddce0282b8a6d1`.

### P5

| Metric | Local | Upstream | Absolute delta |
| --- | ---: | ---: | ---: |
| Object F1 | 0.3679520027114184 | 0.3679520027114184 | 0 |
| Dynamic F1 | 0.06922505723328032 | 0.06922505723328032 | 0 |
| Change F1 | 0.0880875002614133 | 0.08808750026141329 | 1.39e-17 |
| Background F1@0.2 | 0.6501797105835381 | 0.6501797105835381 | 0 |

Result: `PARITY`. Audit JSON SHA-256:
`f9d4e9e578ea69ee50df85576f0a9a5e6f533fd5ad57844688846e0257bca8e4`.

### Artifact Shape

Both artifacts contain 946 raw object rows but only 43 unique `(Name, Query)`
states. For map `m`, the same diagonal query is repeated `m+1` times; upstream
dictionary assignment and the local strict parser both collapse identical
duplicates to one state. Therefore:

| Slice | Object / Dynamic / Change | Background |
| --- | --- | --- |
| `4D` | parity over 43 unique states | parity over 946 weighted values |
| `Online` | parity over 43 diagonal states | unavailable on both paths |
| `Robot` | unavailable on both paths | unavailable on both paths |
| `Query` | unavailable on both paths | unavailable on both paths |

The object CSVs do not contain the off-diagonal keys required by `Robot` and
`Query`. Background expansion uses map indices as query keys while slice lookup
uses timestamps, so non-4D background slices are unavailable in the frozen
upstream code. Matching unavailability is recorded explicitly, not converted to
zero or silently dropped.

## Decision

`MEASURED_EVIDENCE`: local aggregation is not the cause of the A6 or P5 Object,
Dynamic, Change, or Background values. The available post-release aggregation
matches to far below the frozen `1e-12` tolerance.

`CODE_EVIDENCE`: the evaluated artifact shape is diagonal-equivalent rather
than the full historical robot-time/query-time grid. Consequently, B2 rules out
a local summarizer arithmetic bug but does not establish RSS Table I protocol
equivalence. B3 must keep the paper-protocol identity gate open and must not
label these artifacts `PAPER_PROTOCOL_EXACT`.

## Artifact Bindings

| Artifact | A6 SHA-256 | P5 SHA-256 |
| --- | --- | --- |
| `static_objects.csv` | `8e97cc1fbf7748fd5ebafb00b4c744ae5a7f19a3107e1ff5d49f17cb2632b562` | `490be778730d259220f1c4e7b3b7709d8bc0b32cb769cf07d813d4c283299be7` |
| `dynamic_objects.csv` | `f7e6499ca2ee23009fde34db309664543b848311ee3c366fa0176157f5726982` | `61ffb22a2d95f702517abaa4621f0eadd512e50c1cc7c9a7952058c457ac19ff` |
| `background_mesh.csv` | `103bbddfcf548b20533f35de3d8aee4349b98668473aa9872c6d658049baeabb` | `2d9d08a84d9070911477a12687433645c90c4567e37801298d5c11a37be7d97b` |
| `map_timestamps.txt` | `9a77062a447c8c2d9b59c362813e955de965868f7b0f5ffbf091a5f513a68bdb` | `9a77062a447c8c2d9b59c362813e955de965868f7b0f5ffbf091a5f513a68bdb` |
