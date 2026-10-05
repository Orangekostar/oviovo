# Requirement Review

This checklist follows CODEX_FINAL_EXECUTION_EN.md, IMPLEMENTATION_CONTRACTS.md and TIMING_AND_TABLES.md. A pending row is not completion evidence. The primary agent reviews actual source, files, commands, measurement receipts and rendered PDFs before publication.

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Mandate: actual execution, original CT_A3_ER science, deployment N0_UNCHANGED | Original operators remain separate; live pilot ledger; final series still required | IN_PROGRESS |
| Source resolution from actual measured v2, not rounded report numbers | reference_binding.json seals exact producers, v2 experiment, sixteen historical receipts, full 26 contexts and model/text identities | PASS |
| Separate worktree/branch and immutable measured-source snapshot | Dedicated source commit 7fad2c70; source_snapshot.json preserves original HEAD, index and copied producer hashes | PASS |
| Preserve source/index, old reservations/results and external jobs | Read-only source consumers; original index hash retained; final state check pending | PENDING_FINAL_CHECK |
| Resolve literal fallback and genuine U2 semantics | Pinned area_fallback.region_vector: empty hard-support only; area occupancy and original frozen head; archived U2 keeps original signed pooling | PASS |
| Narrow source findings tied to changes | source_to_task.md maps actual functions, differences, invariants and evidence | PASS |
| Required CLI/config and phases | run_ovimap_runtime_parity.py --help exposes bind/profile/screen/verify/freeze/benchmark/tables/publish/all, resume and optional arguments; real phase execution ongoing | IN_PROGRESS |
| New adapter/ledger, no old freeze monkeypatch or old measurement replay | runner.py ColdContext; workflow.py fresh phase/round/scene/variant identities, live PID/start-tick checks | PASS |
| R0 actual reference, R1/R2/R3 cumulative flags | Original FullSceneProjector and original preprocessing/pooling/head/classification/export; explicit four variants | PASS_IMPLEMENTATION |
| C1/C6: cold/verification separation and no parent recovery feature handles | ColdContext rejects reference vectors; VerificationContext alone accesses them; U2 inherited Native observation policy disclosed | PASS |
| C2: honest producer IDs and scientific request-key bijections | views.scientific_key includes geometry, camera, RGB/depth, full mask, both bbox conventions, area/rank, model and ordered vocabulary | PASS_IMPLEMENTATION |
| C3: complete occlusion mesh, integer FP32 rays, four threads and 65536 cap | Production fixture tests cover exact ray bytes, noncandidate/mixed-owner occluders, image edges and depth boundaries | PASS_TESTS |
| C3: conservative ROI, true inverse rotation, outward guard and uncertainty fallback | PreparedCamera.query_union; at least two-pixel guards and full-frame uncertainty/near-plane/camera-inside fallback | PASS_TESTS |
| C4: grouped compact-owner stats and lazy exact Top-3/ties | Large sparse owner IDs and equal-leading-term hash tie tests | PASS_TESTS |
| C5: within-call image reuse, unchanged full-image preprocessing | Regression catches legacy inclusive bbox; real room0 repeated-request read check verifies identical RGB/mask/bbox with one decode | PASS_TESTS |
| C5: original FP32 arithmetic and no precision/batching/kernel tuning | Session invokes pinned original operators, batch one; tensor hash round trip retained; hardware flags recorded | PASS_IMPLEMENTATION |
| C5: deferred computed-vector copies and no live scene tensor retention | R3 per-frame stack/transfer; finally releases dense/image/vector values; no per-image empty_cache | PASS_IMPLEMENTATION |
| Pilot: fixed two scenes, two repeats, four variants, at most sixteen calls | pilot/summary.json: 16/16 COMPLETE, every parity PASS | PASS |
| Pilot: profiling inside R0 slots, exclusive host spans, no extra profiler calls | Same Stages instrumentation on every variant; first R0 calls are profile evidence | PASS_IMPLEMENTATION |
| Selection: equal scene means, two repeats, lower index within 3%, no accuracy selection | pilot_selection.json: R0 41.92894, R1 41.50404, R2 14.69252, R3 15.07190 seconds; selected R2_ROI_EXACT | PASS |
| Verification: full 26 per-frame areas/bboxes/admissibility, ordered Top-3 and mask bytes | verification/summary.json: ALL_26_PASS, identity 508e960b9c7f857a34e9b44d530fd295a23a61e7791a7fc7e0bc26d6d7937acd; every scene receipt verifies all candidate statistics and selected mask bytes | PASS |
| Verification: CF18 cached A2/A3/A5 exports, exact full owners/semantics/ranks, zero GPU | All 18 CF18 receipts regenerate A2/A3/A5 from parent vectors and original text; full prediction keys, owners, semantics and rank strings are exact; new CF18 image encodings 0 | PASS |
| Verification: CPU passes at most 52; reference pass only for missing diagnostics | All original frame diagnostics exist; one selected pass per scene, 26 total, zero reference passes | PASS |
| Freeze: commit candidate, binding, hardware, sources, order and timer before final | freeze_implementation; final worker requires exact committed JSON and source/hardware match | PENDING |
| Final: exactly 64 candidate or 48 retained-reference calls, two reversed rounds | call_plan production test and frozen-plan gate; final series pending | PENDING |
| Timing: model/common inputs resident, recovery views/BVH/RGB/features rebuilt per call | load_common outside timer; build_views and encoding/export inside recover; cache hits explicitly zero | PASS_IMPLEMENTATION |
| Timing: one shape-only warm-up per model process and separate actual GPU costs | warmup receipts record zero scene pixels, one image/one pooling; final process pending | IN_PROGRESS |
| Timing: synchronized total, six exclusive host stages, non-additive CUDA events | Instrumentation test passes; each actual record includes times and allocated/reserved peaks | PASS_IMPLEMENTATION |
| Timing: every valid observation retained; no minimum, outlier removal or slow/parity retry | Complete two-repeat aggregation test rejects missing cells; ledger preserves reservations and failure costs | PASS_IMPLEMENTATION |
| Cost: actual rays, views, packing, RGB, encoder/pooling/fallback, transfers, cache and bytes | Per-call counters and separate warm-up receipts; complete totals pending | IN_PROGRESS |
| Parity: exact discrete outcomes/labels/source order and FP32 atol=rtol=1e-5 | Per-call supervisor; near-tie argmax flip test; first live R0-R3 room0 calls pass | IN_PROGRESS |
| Accuracy: import 172 rows/14 pools once, original identities retained | imported_metrics.json and binding-input verifications | PASS_IMPORT; REUSE_PENDING_ALL_PARITY |
| Speed: complete new G1 reference/selected ratio, >=5% engineering target, nominee unchanged | reporting.aggregate_times; final matched series pending; historical times never used as denominator | PENDING |
| Tables 1/2 unchanged; Table3 contemporary U2/G1/G3 and None zero by definition | Typed parent cells and current final aggregate writer; actual render pending | PENDING |
| S7: A2/A5/A3 both cohorts, shared recovered instances, not pure backbone ablation | reporting.py reuses Table2 cell objects; Native-painted A5 and unavailable-to-unknown behavior disclosed | PENDING_RENDER |
| S8: full exclusive wall stages + total and separate non-additive GPU diagnostics | reporting.py builds receipt-linked three-column supplementary table | PENDING_RENDER |
| Every numeric cell links to parent+parity or all sixteen final measurement IDs | cell_provenance builder; aggregate test checks sixteen constituents | PENDING_FINAL_ARTIFACT |
| Per-scene/repetition means, round means, variation, counters and memory published | runtime_summary/per_call_timing/per_scene_timing/costs builders | PENDING_FINAL_ARTIFACT |
| Three main tables, legible PDF with no clipping | LaTeX compile gate rejects overfull boxes; primary visual inspection pending | PENDING |
| Four named reports, compact artifacts <=25 MiB, no weights/scans/large archive copies | Writers use compact summaries only; final bytes/deliverables audit pending | PENDING |
| Final primary review checks prompt, scoped tests, source and scientific/runtime evidence | This checklist plus validation/completion_audit.json; focused tests already passed, final review pending | IN_PROGRESS |
| Ordinary GitHub push, exact full SHA match, external PUSH_VERIFIED receipt | publish implementation; actual push pending | PENDING |
| Final answer: four report links, SHA/binding, Table3/S8, speed/parity/calls/GPU costs, CF18 limits | Final measured artifacts and verified remote required | PENDING |

Legacy U2 cached request receipts did not record original hard-support counts. This is explicit missing historical metadata, not an invented zero: U2 verifies its full target mask, effective input tensor, original signed-mask operator, success set, FP32 feature, argmax and full export/ranks. Projected v2 requests do contain hard-support counts and compare them exactly.
