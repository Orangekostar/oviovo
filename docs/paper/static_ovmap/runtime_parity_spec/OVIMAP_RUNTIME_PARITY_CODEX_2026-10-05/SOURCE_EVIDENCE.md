# Source evidence and unresolved binding boundary

## Scope

Reviewed commit: `77335848b776d99ffb4ca772e33fa391d9fd8e5e`.
The branch query still resolves to that commit. Direct fetch of
`AREA_FALLBACK_V2_RESULTS.md` returned Not Found. Conversation/Library search found
prior execution packages but no current v2 report/producer. The actual v2 source
and final latency series therefore require server-side binding. We did not run
Open3D, a CUDA model, a 26-scene benchmark or a GitHub write.

Do not interpret the known old source blob hashes as hashes of the latest local
v2. Git blob SHA is source-control identity, not an independently computed SHA256.
The JSON companion lists inspected anchors and expected mapping responsibilities.

## Source → decision

| Evidence | Function / path | Observation | Execution decision |
|---|---|---|---|
| E01 | `src/static_ovmap/cvpr_compact/projected_views.py` / FullSceneProjector.project_frame, _frustum_aabb, camera_rays | Full image cast precedes frustum diagnostics; per-owner full-image scans; top3 packing before discard; RGB and manifest decoded per request; producer-bound request IDs | R1_IO_EXACT, R2_ROI_EXACT, identity_mapping |
| E02 | `src/static_ovmap/cvpr_compact/region_worker.py` / FCSession.encode, classify_regions, archived_u2_plan | Already groups image encoding; still calls each request loader; input CPU roundtrip; per-region transfer; per-image empty_cache; old empty-mask path is not v2 | R1_IO_EXACT, R3_RUNTIME_EXACT, actual_v2_binding |
| E03 | `src/static_ovmap/cvpr_compact/recovery_run.py` / load_recovery_inputs, recover_fc, export_conditions | Inputs already own xyz/faces/raw; build_projected_views reloads surface; full callable includes views, encoding and prediction writes | resident_input_reuse, same_timing_boundary |
| E04 | `src/static_ovmap/cvpr_compact/timing.py` / run_timings, reserve_measurement, compare_recovery_parity | Model/common input resident before synchronized timer; legacy leaves single-use; parity after timer | new_ledger, balanced_repeated_cold_timing, semantic_key_parity |
| E05 | `src/static_ovmap/cvpr_compact/runtime.py` / require_frozen_execution, execute_leaf | Old freeze pins committed producer hashes; old task scope cannot be impersonated by a modified new worktree | new_parent_consumption_adapter, preserve_old_freeze |
| E06 | `src/static_ovmap/a7_evidence_upgrade/region_adapter.py` / image_tensor, signed_mask, region_vector | Fixed 800/1333 preprocessing, padding, FP32 weights and normal signed-mask pooling; backbone name 320 is not actual input size | no_resize_or_precision_change, bind_actual_fallback |
| E07 | `src/static_ovmap/recovery_wave2/binding.py` / ConsumptionIndex.identity, ConsumptionIndex.write_memo | File verification memo already exists and includes filesystem identity; avoid claiming every check rehashes files | profile_actual_duplicate_overhead |
| E08 | `src/static_ovmap/cvpr_compact/outputs.py` / construct_output, build_method_outputs, fc_only_labels | Fixed incumbent support; A2/A3/A5 share G1 recovery; A5 unknowns and current-class ranking matter | output_parity, S7_reuse |
| E09 | `docs/paper/static_ovmap/COMPACT_TABLES_RESULTS.md` /  | Published old phase is 168/172 and 10/14; it cannot replace locally measured complete v2 | do_not_bind_v1_as_v2 |

## Exact source links

- E01: [src/static_ovmap/cvpr_compact/projected_views.py](https://github.com/Orangekostar/oviovo/blob/77335848b776d99ffb4ca772e33fa391d9fd8e5e/src/static_ovmap/cvpr_compact/projected_views.py); inspected Git blob `2f833d8721d2806f344b8dcbfc2ab9d2ba487bc7`.
- E02: [src/static_ovmap/cvpr_compact/region_worker.py](https://github.com/Orangekostar/oviovo/blob/77335848b776d99ffb4ca772e33fa391d9fd8e5e/src/static_ovmap/cvpr_compact/region_worker.py); inspected Git blob `9114b1a83bc775b63281072b9fcb3e27280f799f`.
- E03: [src/static_ovmap/cvpr_compact/recovery_run.py](https://github.com/Orangekostar/oviovo/blob/77335848b776d99ffb4ca772e33fa391d9fd8e5e/src/static_ovmap/cvpr_compact/recovery_run.py); inspected Git blob `a67230f91072498899166cfc8880d8f383ee4775`.
- E04: [src/static_ovmap/cvpr_compact/timing.py](https://github.com/Orangekostar/oviovo/blob/77335848b776d99ffb4ca772e33fa391d9fd8e5e/src/static_ovmap/cvpr_compact/timing.py); inspected Git blob `cc5d435d210e35d8bfba98ba4a00c975ca1aa9a8`.
- E05: [src/static_ovmap/cvpr_compact/runtime.py](https://github.com/Orangekostar/oviovo/blob/77335848b776d99ffb4ca772e33fa391d9fd8e5e/src/static_ovmap/cvpr_compact/runtime.py); inspected Git blob `6cbcdb26b8e37ac42c1f470189d8f8c7dadf1c55`.
- E06: [src/static_ovmap/a7_evidence_upgrade/region_adapter.py](https://github.com/Orangekostar/oviovo/blob/77335848b776d99ffb4ca772e33fa391d9fd8e5e/src/static_ovmap/a7_evidence_upgrade/region_adapter.py); inspected Git blob `70e101ef4a90f0b6df3e4132572cfa619b5dec87`.
- E07: [src/static_ovmap/recovery_wave2/binding.py](https://github.com/Orangekostar/oviovo/blob/77335848b776d99ffb4ca772e33fa391d9fd8e5e/src/static_ovmap/recovery_wave2/binding.py); inspected Git blob `4d3b5e9cb5f488f14643f2f69c86af763bda1231`.
- E08: [src/static_ovmap/cvpr_compact/outputs.py](https://github.com/Orangekostar/oviovo/blob/77335848b776d99ffb4ca772e33fa391d9fd8e5e/src/static_ovmap/cvpr_compact/outputs.py); inspected Git blob `8b7bd06d012f4abb6706533130c73ec8c3057b25`.
- E09: [docs/paper/static_ovmap/COMPACT_TABLES_RESULTS.md](https://github.com/Orangekostar/oviovo/blob/77335848b776d99ffb4ca772e33fa391d9fd8e5e/docs/paper/static_ovmap/COMPACT_TABLES_RESULTS.md); inspected Git blob `d9e328020cbba837307bb2b595d7d3f6fed40386`.

## OVI-MAP Table 7 / 8 and implementation documentation

- W1: [Supplementary §§7–8, Tables 7–8](https://arxiv.org/html/2603.26541v1). Table7 model variants/metrics/parameters; Table8 component time, threads and skipped frames. Use qualitative table purposes, not transplanted numeric rows. Status: HTML_READ; PDF fetch failed, no PDF visual-verification claim.
- W2: [cast_rays](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html). Ray direction length affects t_hit units; preserve parent camera-z convention. Do not upgrade runtime dependency to the docs release automatically. Status: OFFICIAL_DOCUMENTATION_READ.
- W3: [empty_cache](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.empty_cache.html). Releases unused allocator cache, not the same as persistent image/semantic-feature caching. Status: OFFICIAL_DOCUMENTATION_READ.

The source paper's Table 7 is not a mandate to repeat five backbones; our new task
uses the already executed A2/A5/A3 readout comparison and limits its claim. Table 8
motivates a recovery-only stage breakdown, not mixing another system's hardware,
frame skipping or online execution into our offline seconds-per-scene table.

## Proposed, not already measured

Conservative ROI pruning, grouped/lazy mask work, loader reuse, new balanced cold
repetitions and any speed reduction are proposed work. The source supports that
redundant operations exist; it does not establish how much of 45.58s they account
for or how much acceleration will result. The included small reference tests cover
specified invariants only, not the actual parent code, numerical backend or data.
