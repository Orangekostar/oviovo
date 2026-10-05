"""Complete two-repeat timing summaries and receipt-backed paper artifacts."""

from collections import Counter
import copy
import csv
import io
from pathlib import Path
import shutil
import subprocess

import numpy as np

from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.cvpr_compact.tables import format_cell, render_tables
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import read

from .binding import REPO, file_identity
from .instrumentation import STAGES


def aggregate_times(records, scenes, label):
    rows = [row for row in records if row["label"]==label]
    if (len(scenes)!=8 or len(rows)!=16 or len({(row["scene"],row["round"]) for row in rows})!=16
            or {(row["scene"],row["round"]) for row in rows}!={(scene,r) for scene in scenes for r in (1,2)}
            or any(row["status"]!="COMPLETE" or row["parity"]["status"]!="PASS"
                or not np.isfinite(row["seconds"]) or row["seconds"]<=0 for row in rows)):
        raise ValueError("aggregate requires exactly two complete finite passing repeats on all eight scenes")
    per_scene = {scene:sum(row["seconds"] for row in rows if row["scene"]==scene)/2 for scene in scenes}
    rounds = {str(r):sum(row["seconds"] for row in rows if row["round"]==r)/8 for r in (1,2)}
    stage_names = set().union(*(row["stage_seconds"] for row in rows))
    service_names = set().union(*(row["CUDA_service_seconds_non_additive"] for row in rows))
    counters = Counter()
    for row in rows:
        counters.update(row["work_counters"])
    differences = [abs(next(row["seconds"] for row in rows if row["scene"]==scene and row["round"]==1)
        -next(row["seconds"] for row in rows if row["scene"]==scene and row["round"]==2)) for scene in scenes]
    return seal({"label":label,"mean_seconds":sum(per_scene.values())/8,"scene_means":per_scene,
        "round_means":rounds,"within_scene_repeat_difference_range_seconds":[min(differences),max(differences)],
        "measurement_ids":[row["identity"] for scene in scenes for r in (1,2)
            for row in rows if row["scene"]==scene and row["round"]==r],
        "stage_seconds":{name:sum(row["stage_seconds"].get(name,0.) for row in rows)/16 for name in sorted(stage_names)},
        "CUDA_service_seconds_non_additive":{name:sum(row["CUDA_service_seconds_non_additive"].get(name,0.) for row in rows)/16
            for name in sorted(service_names)},"work_counters_total":dict(counters),
        "work_counters_mean":{name:value/16 for name,value in counters.items()},
        "peak_cuda_allocated_bytes_max":max(row["peak_cuda_allocated_bytes"] for row in rows),
        "peak_cuda_reserved_bytes_max":max(row["peak_cuda_reserved_bytes"] for row in rows)})


def csv_text(rows):
    if not rows:
        raise ValueError("cannot publish an empty measured CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def supplementary_table(caption, label, columns, rows):
    lines = [r"\begin{table*}[!ht]",r"\centering",r"\caption{"+caption+"}",r"\label{"+label+"}",
        r"\fontsize{9}{11}\selectfont",r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l"+"r"*(len(columns)-1)+r"@{}}",r"\toprule",
        " & ".join(columns)+r" \\",r"\midrule"]
    lines += [" & ".join(row)+r" \\" for row in rows]
    return "\n".join([*lines,r"\bottomrule",r"\end{tabular*}",r"\end{table*}",""])


def build_report(reference, spec, root):
    root = Path(root)
    pilot,final,verification,freeze = (read(root/name) for name in ("pilot/summary.json","final/summary.json",
        "verification/summary.json","implementation_freeze.json"))
    for document in (pilot,final,verification,freeze):
        _verified_identity(document)
    if verification["status"]!="ALL_26_PASS" or final["status"]!="COMPLETE":
        raise ValueError("complete tables require full scientific parity and the entire frozen timing series")
    selected = freeze["selected"]
    scenes = spec["cohorts"]["replica8"]
    labels = list(dict.fromkeys(row["label"] for row in final["measurements"]))
    aggregates = {label:aggregate_times(final["measurements"],scenes,label) for label in labels}
    selected_g1 = "G1_REFERENCE" if selected=="R0_REFERENCE" else "G1_SELECTED"
    selected_g3 = "G3_REFERENCE" if selected=="R0_REFERENCE" else "G3_SELECTED"
    ref_time,fast_time = (aggregates[label]["mean_seconds"] for label in ("G1_REFERENCE",selected_g1))
    speedup,reduction = ref_time/fast_time,1-fast_time/ref_time
    material = selected!="R0_REFERENCE" and reduction>=spec["reporting"]["faster_threshold_fraction"]
    report = seal({"status":"COMPLETE","source_binding":"BOUND_ACTUAL_MEASURED_V2",
        "implementation":"IMPLEMENTED_AND_EXECUTED","parity":"ALL_26_CPU_AND_ALL_COLD_CALLS_PASS",
        "timing_coverage":f"{final['calls']}/{final['planned_calls']}",
        "metric_reuse":"PREDICTION_IDENTICAL_PARENT_METRIC","selected":selected,
        "recommended_execution":selected if material else "R0_REFERENCE","speedup":speedup,"reduction_fraction":reduction,
        "speed_result":"MATERIAL_REDUCTION" if material else "NO_DEMONSTRATED_MATERIAL_REDUCTION",
        "deployment":"N0_UNCHANGED","reference_binding_identity":reference["identity"],
        "freeze_identity":freeze["identity"],"verification_identity":verification["identity"],
        "pilot_calls":pilot["calls"],"final_calls":final["calls"],"normal_cold_calls":pilot["calls"]+final["calls"],
        "aggregates":aggregates,"publication":"AWAITING_PUSH_VERIFICATION"})
    destination = root/"tables"
    destination.mkdir(parents=True,exist_ok=True)
    atomic_write_json(destination/"runtime_summary.json",report)
    imported = read(root/"imported_metrics.json")
    imported.update(reuse_status="PREDICTION_IDENTICAL_PARENT_METRIC",output_parity_identity=verification["identity"])
    imported = seal(imported)
    atomic_write_json(destination/"imported_metrics.json",imported)
    tables = copy.deepcopy(read(Path(reference["v2_results_root"])/"tables/tables_main.json"))
    tables.update(protocol="OVIMAP_RUNTIME_PARITY_V1",timing_identity=final["identity"],parity_identity=verification["identity"])
    for table_rows in tables["tables"].values():
        for row in table_rows:
            if row["block"]=="AUTHOR_REPORTED_NOT_RERUN":
                continue
            for cell in row["cells"]:
                cell.update(reuse_status="PREDICTION_IDENTICAL_PARENT_METRIC",output_parity_identity=verification["identity"])
    arm_labels = {"U2":"U2_CONTROL","G1":selected_g1,"G3":selected_g3}
    for row in tables["tables"]["table3"]:
        for cell in row["cells"]:
            if cell["metric"]!="cold_feature_incremental_seconds_mean":
                continue
            if row["arm"]=="NONE":
                cell.update(value_seconds=0.,zero_basis="ZERO_INCREMENTAL_RECOVERY_BY_DEFINITION",measurement_receipt_identities=[])
            else:
                aggregate = aggregates[arm_labels[row["arm"]]]
                cell.update(value_seconds=aggregate["mean_seconds"],zero_basis=None,source_kind="MEASURED",
                    source_id=aggregate["identity"],receipt_identity=final["identity"],receipt_path=str(root/"final/summary.json"),
                    measurement_receipt_identities=aggregate["measurement_ids"],completed_coverage=8,
                    repeats_per_scene=2,implementation=selected if row["arm"]!="U2" else "R0_REFERENCE",
                    unavailable_reason=None)
    tables = seal(tables)
    atomic_write_json(destination/"tables_main.json",tables)
    sources = render_tables(tables)
    caption = ("Recovery quality and incremental latency on Replica-8. n/N is candidate owners receiving semantics, not true-positive recall. "
        "TP/FP are definite added matcher entries at IoU 0.50; ambiguous ties retain the dagger. "
        "Every nonzero latency averages eight scenes and two cold repeats with resident model/common map; mapping/evaluation are excluded. "
        "U2 uses its archived observation policy; G1/G3 perform fresh full-sequence visibility search. "
        "Evaluated G1/G3 implementation: "+selected.replace("_",r"\_")+".")
    lines = sources["table3_recovery.tex"].splitlines()
    lines[next(i for i,line in enumerate(lines) if line.startswith(r"\caption{"))] = r"\caption{"+caption+"}"
    sources["table3_recovery.tex"] = "\n".join(lines)+"\n"
    provenance = [cell for rows in tables["tables"].values() for row in rows for cell in row["cells"]]
    s7_rows = []
    readouts = {"CT_A2_R":"A2: Native","CT_A5_FC_ONLY":"A5: FC-only","CT_A3_ER":"A3: D2"}
    by_id = {row["row_id"]:row for row in tables["tables"]["table2"]}
    for method,name in readouts.items():
        cells = by_id[method]["cells"]
        metrics = [next(cell for cell in cells if cell.get("cohort")==cohort and cell["metric"]==metric)
            for cohort in ("replica8","scannet_cf18") for metric in ("apall","ap50","miou")]
        s7_rows.append([name,*[format_cell(cell) for cell in metrics]])
        provenance += [{**cell,"table_id":"S7","shared_cell_source_id":cell["source_id"]} for cell in metrics]
    sources["supplementary_s7_readout.tex"] = supplementary_table(
        "Readout comparison with shared recovered instances. All rows share G1-FC recovery and differ in incumbent readout. "
        "A5 retains Native-painted geometry and emits unknown where incumbent FC is unavailable. This is not a pure backbone ablation.",
        "tab:readout-shared",["Readout + shared G1","Replica AP","AP50","mIoU","CF18 AP","AP50","mIoU"],s7_rows)
    runtime_labels = ["U2_CONTROL","G1_REFERENCE",selected_g1]
    headers = ["Stage (s)","U2 control","G1 reference","G1 selected" if selected!="R0_REFERENCE" else "G1 retained"]
    s8_rows = []
    for stage in [*STAGES,"complete_incremental_total"]:
        cells = []
        for label in runtime_labels:
            aggregate = aggregates[label]
            value = aggregate["mean_seconds"] if stage=="complete_incremental_total" else aggregate["stage_seconds"][stage]
            cells.append(f"{value:.3f}")
            provenance.append({"table_id":"S8","row_id":stage,"column_id":label,"value_seconds":value,
                "display_unit":"seconds_per_scene","measurement_receipt_identities":aggregate["measurement_ids"],
                "source_id":aggregate["identity"]})
        s8_rows.append([stage.replace("_"," "),*cells])
    s8_rows.append([r"\multicolumn{4}{l}{GPU event diagnostics (non-additive)}"])
    for service in ("encoder","region_pool_head"):
        cells = []
        for label in runtime_labels:
            aggregate = aggregates[label]
            value = aggregate["CUDA_service_seconds_non_additive"][service]
            cells.append(f"{value:.3f}")
            provenance.append({"table_id":"S8","row_id":"GPU_"+service,"column_id":label,"value_seconds":value,
                "display_unit":"nonadditive_GPU_event_seconds","measurement_receipt_identities":aggregate["measurement_ids"],
                "source_id":aggregate["identity"]})
        s8_rows.append([service.replace("_"," "),*cells])
    sources["supplementary_s8_runtime.tex"] = supplementary_table(
        "Recovery runtime with the same measured return boundary. Host stages are exclusive and sum to total wall time. "
        "GPU events are non-additive elapsed diagnostics, not extra wall time. U2 has an archived observation policy; G1 performs fresh visibility search.",
        "tab:runtime",headers,s8_rows)
    atomic_write_json(destination/"cell_provenance.json",seal({"tables_identity":tables["identity"],"cells":provenance}))
    for filename,source in sources.items():
        (destination/filename).write_text(source)
    preamble = r"\documentclass[10pt]{article}"+"\n"+r"\usepackage[margin=15mm]{geometry}"+"\n"+r"\usepackage{booktabs}"+"\n"+r"\begin{document}"+"\n"
    for name,files in (("table_layout_preview",["table1_main","table2_ablation","table3_recovery"]),
                      ("supplementary_preview",["supplementary_s7_readout","supplementary_s8_runtime"]),
                      ("table3_preview",["table3_recovery"])):
        (destination/(name+".tex")).write_text(preamble+"\n".join(r"\input{"+file+".tex}" for file in files)+"\n"+r"\end{document}"+"\n")
        result = subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error",name+".tex"],cwd=destination,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError("table PDF build failed: "+result.stdout[-4000:])
        if "Overfull \\hbox" in result.stdout or "Overfull \\vbox" in result.stdout:
            raise RuntimeError("table PDF has clipping/overfull boxes; inspect the TeX before publication")
    records = [*pilot["measurements"],*final["measurements"]]
    csv_rows = [{"phase":row["phase"],"scene":row["scene"],"round":row["round"],"label":row["label"],
        "implementation":row["implementation"],"seconds":row["seconds"],"measurement_id":row["identity"],
        "parity":row["parity"]["status"],**row["stage_seconds"],
        **{"GPU_"+name:value for name,value in row["CUDA_service_seconds_non_additive"].items()},
        **row["work_counters"],"peak_cuda_allocated_bytes":row["peak_cuda_allocated_bytes"],
        "peak_cuda_reserved_bytes":row["peak_cuda_reserved_bytes"]} for row in records]
    (destination/"per_call_timing.csv").write_text(csv_text(csv_rows))
    scene_rows = [{"label":label,"scene":scene,"mean_seconds":aggregate["scene_means"][scene],
        "aggregate_identity":aggregate["identity"]} for label,aggregate in aggregates.items() for scene in scenes]
    (destination/"per_scene_timing.csv").write_text(csv_text(scene_rows))
    counters = Counter()
    for row in records:
        counters.update(row["work_counters"])
    warmups = [read(path) for phase in ("pilot","final") for path in sorted((root/phase/"warmup").glob("*.json"))]
    costs = seal({"normal_cold_calls":len(records),"cold_scene_counters":dict(counters),"warmups":warmups,
        "warmup_image_encodings":sum(row["physical_image_encodings"] for row in warmups),
        "warmup_region_poolings":sum(row["physical_region_poolings"] for row in warmups),
        "new_CF18_GPU_inference":0,"failed_attempts":[str(p) for p in root.glob("**/measurement.json") if read(p)["status"]=="FAILED_MEASUREMENT"]})
    atomic_write_json(destination/"costs.json",costs)
    write_documents(reference,root,report,tables,s8_rows,costs)
    compact = REPO/"artifacts/static_ovmap/runtime_parity_v1"
    compact.mkdir(parents=True,exist_ok=True)
    for filename in ("reference_binding.json","implementation_freeze.json","pilot_selection.json"):
        shutil.copy2(root/filename,compact/filename)
    for phase in ("pilot","final","verification"):
        shutil.copy2(root/phase/"summary.json",compact/(phase+"_summary.json"))
    for path in destination.iterdir():
        if path.suffix in (".json",".csv",".tex",".pdf"):
            shutil.copy2(path,compact/path.name)
    size = sum(path.stat().st_size for path in compact.rglob("*") if path.is_file())
    if size>25*1024**2:
        raise ValueError("new compact evidence exceeds the 25 MiB publication limit")
    return report


def write_documents(reference, root, report, tables, s8_rows, costs):
    docs = REPO/"docs/paper/static_ovmap"
    docs.mkdir(parents=True,exist_ok=True)
    caption = "Cache-cold incremental recovery with model/common geometry/NQF resident; OS page cache uncontrolled."
    table3 = "| Arm | n/N | TP50 | FP50 | AP (%) | mIoU (%) | Seconds/scene |\n| --- | --- | --- | --- | --- | --- | --- |\n"
    for row in tables["tables"]["table3"]:
        cells = {cell["metric"]:format_cell(cell).replace("$","") for cell in row["cells"]}
        table3 += "| "+" | ".join([row["arm"],*[cells[name] for name in ("recovered_n_over_N","added_tp50_entries","added_fp50_entries","apall","miou","cold_feature_incremental_seconds_mean")]])+" |\n"
    s8 = "| Stage (seconds) | U2 control | G1 reference | G1 selected/retained |\n| --- | --- | --- | --- |\n"
    s8 += "".join("| "+" | ".join(row)+" |\n" for row in s8_rows if len(row)==4)
    hardware = read(root/"final/summary.json")["hardware"]
    body = ("# Runtime Parity Results\n\n"+caption+"\n\n"
        +f"Source binding: `{reference['identity']}`. Actual v2: `{reference['v2_experiment_identity']}`. Reference commit: `{reference['reference_commit']}`.\n\n"
        +f"Frozen nominee: `{report['selected']}`; recommended future execution: `{report['recommended_execution']}`. Deployment: `N0_UNCHANGED`.\n\n"
        +f"New paired G1 speedup: {report['speedup']:.4f}x; mean reduction: {report['reduction_fraction']*100:.3f}%. Status: `{report['speed_result']}`. This is not a significance test.\n\n"
        +table3+"\n## Supplementary S8\n\n"+s8+"\nGPU event rows are non-additive annotations. Host stages include all required output writes.\n\n"
        +f"Coverage: {report['pilot_calls']} pilot + {report['final_calls']} final cold calls; all 26 CPU view/cached-export checks passed. Accuracy inherits 172 rows/14 pools as `PREDICTION_IDENTICAL_PARENT_METRIC`.\n\n"
        +f"Cold-scene image encodings: {costs['cold_scene_counters'].get('FC_image_inputs',0)}; normal pool/head calls: {costs['cold_scene_counters'].get('normal_pool_head_calls',0)}; fallback pool/head calls: {costs['cold_scene_counters'].get('fallback_pool_head_calls',0)}. Shape-only warm-ups: {costs['warmup_image_encodings']} image inputs and {costs['warmup_region_poolings']} pool/head calls, counted separately. CF18 new GPU inference: 0.\n\n"
        +f"GPU: {hardware['GPU']}; CPU: {hardware['CPU']}; PyTorch {hardware['torch']}; CUDA {hardware['CUDA']}; Open3D {hardware['open3d']}. Raycast threads=4, encoder batch=1; original precision/TF32/determinism unchanged.\n\n"
        +"U2 recomputes FC features for its genuine archived Native observation policy; G1/G3 reconstruct views/BVH from the full sequence in every call. Existing grouping, model residency and inference_mode are inherited. The common new evidence writer replaces legacy duplicate receipts for both R0 and candidates; historical timings are not the speedup denominator.\n\n"
        +"Two repeats use all valid observations, including slow runs. Per-call CSV, per-scene means, round means, repetition differences, memory peaks, ray/IO/transfer/output counters and all constituent measurement IDs are in the compact artifacts.\n\n"
        +"Unchanged accuracy is expected. Runtime optimization does not repair CF18's recovery tradeoff: A3 vs D2 remains approximately AP -0.08pp, mIoU +0.02pp. All scenes are exposed; CF18 comprises 18 captures from seven physical families. No online, end-to-end FPS, unseen generalization or global-SOTA claim follows.\n")
    (docs/"RUNTIME_PARITY_RESULTS.md").write_text(body)
    (docs/"RUNTIME_PARITY_SELECTION.md").write_text("# Runtime Parity Selection\n\n"
        +f"Pilot selection identity: `{read(root/'pilot_selection.json')['identity']}`. Two fixed scenes (room0, office1), two repeats per R0-R3 variant; equal scene means, lower variant index within 3% of fastest. Accuracy is not a selection criterion.\n\n"
        +f"Frozen nominee: `{report['selected']}`. Final result: `{report['speed_result']}`, reduction {report['reduction_fraction']*100:.3f}%. Recommended execution: `{report['recommended_execution']}`. Evaluated Table3 rows retain the frozen nominee even if the final gain is not material.\n\n"
        +"No per-scene implementation selection, lucky-minimum replacement or additional model search occurred. Deployment remains N0_UNCHANGED.\n")
    command = ("/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_runtime_parity.py "
        "--spec configs/static_ovmap/runtime_parity_v1.json --source-worktree "+reference["source_worktree"]
        +" --v2-results-root "+reference["v2_results_root"]+" --output-root "+str(root)+" --phase all --resume")
    (docs/"RUNTIME_PARITY_HANDOFF.md").write_text("# Runtime Parity Handoff\n\n"
        +f"Attempt: `{root}`. Reference binding: `{reference['identity']}`. Implementation freeze: `{report['freeze_identity']}`.\n\n"
        +"```bash\n"+command+"\n```\n\n"
        +"Phases: bind, profile, screen, verify, freeze, benchmark, tables, publish, all. Profile/screen consume the same bounded pilot series; completed leaves are reused by source/variant/scene/round. Live reservations are observed, never killed. Internal or parity failures are not automatically replayed.\n\n"
        +"Common cold inputs contain no parent recovery features/views/results. Parent vectors are available only to VerificationContext after timing or in CPU cached export checks. Dense tensors, scans and weights remain outside Git. Original source files, its index and old timing reservations are preserved.\n\n"
        +"The publication receipt is external at publication/final.json and is written only after ordinary push and exact local/remote SHA equality. Rendering QA and the requirement audit are linked in artifacts/static_ovmap/runtime_parity_v1.\n")
    (docs/"RUNTIME_PARITY_CLAIMS.md").write_text("# Runtime Parity Claims\n\n"
        +f"Supported: actual measured v2 binding; exact verified outputs over 26 geometries; complete same-boundary cold latency over eight Replica scenes with two repeats; G1 reduction {report['reduction_fraction']*100:.3f}%; full counters and memory; inherited accuracy with explicit parity links.\n\n"
        +"Unsupported: new accuracy gain, strict backbone-only ablation, Native-free A5, G3 contemporary speedup without a newly measured G3 reference, full-ScanNet generalization, significance from two repeats, online mapping or end-to-end FPS.\n\n"
        +"S7 compares incumbent readouts with shared recovered instances. A5 retains Native-painted geometry and emits unknown without incumbent FC. N/Q share weights; no new backbone or fit is introduced. Runtime optimization cannot resolve the CF18 geometric recovery failure. Deployment remains N0_UNCHANGED.\n")
