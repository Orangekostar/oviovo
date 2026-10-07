"""One final store produces all displayed cells, companion data, and reports."""

from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import subprocess

from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .diagnostic_details import paired_analysis
from .verifier import METRICS


def compressed_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n").encode()
    with path.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as handle:
        handle.write(payload)


def table(directory, name, columns, rows):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory/(name+".json"), {"columns": columns, "rows": rows})
    with (directory/(name+".csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    def display(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)
    lines = ["| "+" | ".join(columns)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
    lines += ["| "+" | ".join(display(row[col]) for col in columns)+" |" for row in rows]
    text = "\n".join(lines)+"\n"
    (directory/(name+".md")).write_text(text)
    return text


def resource_context():
    """Existing matched measurements; author times remain separate context."""
    old_repo = Path("/mnt/shared/ww/ovimap-runtime-parity-v1/worktree")
    table_path = old_repo/"artifacts/static_ovmap/table78_reproduction_v1/table_data.json"
    parity_path = Path("/mnt/shared/ww/ovimap-runtime-parity-v1/attempt_001/tables/runtime_summary.json")
    profile = Path("/mnt/shared/ww/ovimap-table78-reproduction-v1/attempt_002/table78")
    index = ConsumptionIndex()
    data, parity = read(table_path), read(parity_path)
    index.identity(table_path)
    index.identity(parity_path)
    matched = []
    for method in ("G1_SELECTED", "G3_SELECTED"):
        measurements = []
        for path in sorted(profile.glob("round_*/*/"+method+"/measurement.json")):
            index.identity(path)
            value = read(path)
            _verified_identity(value)
            if value["parity"]["status"] != "PASS":
                raise ValueError("resource comparison requires prediction parity")
            measurements.append(value)
        if len(measurements) != 16:
            raise ValueError("resource comparison requires both full Replica8 repeats")
        matched.append({"Method": method, "Calls": len(measurements),
            "Seconds_scene": sum(x["seconds"] for x in measurements)/len(measurements),
            "Allocated_GiB": max(x["peak_cuda_allocated_bytes"] for x in measurements)/2**30,
            "Reserved_GiB": max(x["peak_cuda_reserved_bytes"] for x in measurements)/2**30,
            "FC_images_scene": data["table8"][method]["mean_counts_per_scene"]["FC_image_inputs"]})
    ours = {(x["method"], x["component"]): x for x in data["table8_skipped30"]["our_rows"]}
    modules = []
    for component, ms, _, _ in data["published_reference"]["table8"]:
        component = "Semantic extraction" if component == "Feature extraction" else component
        a, b = ours["G1", component], ours["G3", component]
        modules.append({"Module": component, "OVIMAP_ms": ms, "G1_ms": a["amortized_ms_per_processed_unit"],
            "G3_ms": b["amortized_ms_per_processed_unit"], "n": a["skipped_frames_n"]})
    return {"source_records": index.entries(), "source": data["published_reference"]["url"],
        "source_kind": "OVI-MAP author report, not same-machine rerun", "OVIMAP_peak_GPU_memory": None,
        "OVIMAP_hardware": "RTX3090 + Intel Core i7-12700K", "our_hardware": data["hardware"],
        "module_rows": data["table8_skipped30"], "module_comparison_rows": modules,
        "module_boundaries": data["shared_frontend"]["module_boundaries"],
        "matched_G1_G3_rows": matched, "matched_before_after_G1": {key: parity[key] for key in ("speedup", "reduction_fraction")},
        "before_after_rows": {key: {k: value[k] for k in ("mean_seconds", "peak_cuda_allocated_bytes_max", "peak_cuda_reserved_bytes_max")}
            for key, value in parity["aggregates"].items() if key in ("G1_REFERENCE", "G1_SELECTED")},
        "parameter_count_is_not_peak_memory": True, "no_cross_hardware_speedup_claim": True,
        "new_inference_for_historical_comparison": 0}


def build_tables(binding, root):
    root = Path(root)
    repo = Path(__file__).resolve().parents[3]
    artifact = repo/"artifacts/static_ovmap/evidence_exploration_v1"
    science, selection, timing = (read(root/name) for name in ("result_store.json", "selection.json", "timing/summary.json"))
    for value in (science, selection, timing):
        _verified_identity(value)
    if science["scene_method_coverage"] != 234 or science["full_cohort_pool_coverage"] != 18:
        raise ValueError("tables require the complete fixed study, not a selected subset")
    if timing["status"] != "PAIRED_COLD_TIMING_COMPLETE":
        raise ValueError("completed selected paired timing is required")
    paired = paired_analysis(binding, root)
    decisions = {scene: {method["id"]: read(root/"predictions"/scene/method["id"]/"decisions.json")
        for method in binding["specification"]["methods"]} for scene in binding["scenes"]}
    bundle = seal({"status": "RESULTS_AND_TIMING_COMPLETE", "specification": binding["specification"],
        "source_binding_identity": binding["identity"], "science": science, "selection": selection,
        "timing": timing, "paired_diagnostics": paired, "candidate_decisions": decisions,
        "baseline_diagnostic_summary": read(root/"diagnostics/detail_summary.json"),
        "scientific_acquisition": read(root/"encoded/summary.json"),
        "selected_frame_geometry": read(root/"prepared/summary.json"),
        "real_pilots": {"neutral_full_frame": read(root/"pilots/neutral_full_frame.json"),
                        "acquisition": read(root/"pilots/acquisition.json")},
        "implementation_freeze": read(root/"implementation_freeze.json"), "assets": read(root/"assets.json"),
        "timing_hardware_manifest": read(root/"timing/hardware_manifest.json"),
        "resource_comparison": resource_context(),
        "baseline_counterfactual_pools": read(root/"diagnostics/summary.json")["counterfactual_pools"],
        "failures_preserved": [read(path) for path in sorted((root/"history").rglob("*.json")) if path.name in ("patch.json", "receipt.json") or path.name == "output_metadata_fix_001.json"],
        "storage_relocation": {"logical_root": str(root), "physical_root": str(root.resolve()),
            "receipt": "/home/ww/ovimap-evidence-exploration-v1-storage/storage_relocation.json"},
        "deployment": "N0_UNCHANGED", "all_cohorts_exposed": True, "independent_confirmation": False})
    compressed_json(artifact/"study_result_store.json.gz", bundle)
    atomic_write_json(root/"study_result_store_identity.json", {"identity": bundle["identity"], "artifact": str(artifact/"study_result_store.json.gz")})
    poolmap = {(p["cohort"], p["method"]): p for p in science["pooled_metrics"]}
    main, cells = [], []
    for cohort in binding["cohorts"]:
        baseline = poolmap[cohort, "EV01_G1_V2"]
        for method in binding["specification"]["methods"]:
            name = method["id"]
            row = poolmap[cohort, name]
            main.append({"Cohort": cohort, "Method": name, **{m: row["metrics"][m]*100 for m in ("apall", "ap50", "miou")},
                **{"Δ"+m+"_pp": (row["metrics"][m]-baseline["metrics"][m])*100 for m in ("apall", "ap50", "miou")},
                "TARGET_MET": selection["target_flags"][name]["target_met"]})
            cells.append({"table": "table1_main", "cohort": cohort, "method": name, "pool_identity": row["identity"],
                "delta_reference": baseline["identity"], "target_flag_selection": selection["identity"], "store": bundle["identity"]})
    first = table(artifact, "table1_main", ["Cohort", "Method", "apall", "ap50", "miou", "Δapall_pp", "Δap50_pp", "Δmiou_pp", "TARGET_MET"], main)
    second_rows = []
    baseline_counts = {(x["cohort"], x["candidate"]): x["paired_score_and_GT_counts"] for x in paired["baseline_comparisons"]}
    for row in paired["comparisons"]:
        counts = row["paired_score_and_GT_counts"]
        against_B1 = baseline_counts[row["cohort"], row["candidate"]]
        second_rows.append({"Cohort": row["cohort"], "Comparison": row["candidate"]+" / "+row["reference"],
            "ΔAPall_pp": row["metric_deltas_fraction"]["apall"]*100, "ΔAP50_pp": row["metric_deltas_fraction"]["ap50"]*100,
            "ΔmIoU_pp": row["metric_deltas_fraction"]["miou"]*100,
            "Accepted": row["candidate_decisions"]["accepted"], "Deferred": row["candidate_decisions"]["deferred"],
            "Relabelled": row["candidate_decisions"]["relabelled"],
            "Lost_reference_TP50": counts["lost_definite_TP_score_entries50"], "Removed_reference_FP50": counts["removed_definite_FP_score_entries50"],
            "Lost_B1_TP50": against_B1["lost_definite_TP_score_entries50"], "Removed_B1_FP50": against_B1["removed_definite_FP_score_entries50"]})
        cells.append({"table": "table2_mechanisms", "cohort": row["cohort"], "candidate": row["candidate"], "reference": row["reference"],
            "candidate_pool": poolmap[row["cohort"], row["candidate"]]["identity"],
            "reference_pool": poolmap[row["cohort"], row["reference"]]["identity"], "diagnostic_identity": paired["identity"], "store": bundle["identity"]})
    second = table(artifact, "table2_mechanisms", list(second_rows[0]), second_rows)
    third_rows = []
    for row in timing["summary"]:
        third_rows.append({"Method": row["method"], "Calls": row["calls"], "Seconds_scene": row["mean_seconds_per_scene"],
            "Allocated_GiB": row["peak_cuda_allocated_bytes"]/2**30, "Reserved_GiB": row["peak_cuda_reserved_bytes"]/2**30,
            "FC_inputs_total": row["counts"].get("FC_image_inputs", 0), "AnyUp_QK_total": row["counts"].get("AnyUp_QK_computations", 0),
            "Added_projection_total": row["counts"].get("additional_full_scene_projection_calls", 0)})
        cells.append({"table": "table3_timing", "method": row["method"], "timing_identity": timing["identity"],
            "call_identities": [x["identity"] for x in timing["calls"] if x["method"] == row["method"]], "store": bundle["identity"]})
    third = table(artifact, "table3_timing", list(third_rows[0]), third_rows)
    atomic_write_json(artifact/"cell_provenance.json", {"single_result_store_identity": bundle["identity"], "cells": cells})
    compressed_json(artifact/"scene_and_pooled_all_metrics_classes.json.gz", {"scene_rows": science["scene_metrics"], "pooled_rows": science["pooled_metrics"]})
    compressed_json(artifact/"candidate_decisions.json.gz", decisions)
    compressed_json(artifact/"paired_diagnostics.json.gz", paired)
    atomic_write_json(artifact/"timing_records.json", timing)
    atomic_write_json(artifact/"selection.json", selection)
    atomic_write_json(artifact/"implementation_freeze.json", bundle["implementation_freeze"])
    atomic_write_json(artifact/"upstream_weight_identity.json", {"AnyUp": bundle["assets"],
        "FC_physical_model_identity": binding["reference"]["contexts"][binding["cohorts"]["replica8"][0]]["model_identity"],
        "FC_runtime": binding["reference"]["fc"], "weights_published": False,
        "AnyUp_attribution": "Thomas Wimmer et al.; wimmerth/anyup@351807a9; CC-BY-4.0. Our wrapper adds owner/depth readout after original averaged-head attention."})
    compressed_json(artifact/"source_binding.json.gz", binding)
    compressed_json(artifact/"baseline_diagnostics.json.gz", {"original": read(root/"diagnostics/summary.json"),
        "detail_supplements": {scene: read(root/"diagnostics"/scene/"detail_supplement.json") for scene in binding["scenes"]}})
    dependency = {"large_files_excluded_from_Git": True, "shared_root": str(root), "physical_root": str(root.resolve()),
        "reconstruct": "/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_evidence_exploration.py --phase all --resume",
        "prepared_receipts": {scene: read(root/"prepared"/scene/"receipt.json") for scene in binding["scenes"]},
        "encoded_receipts": [read(root/"encoded"/x["scene"]/str(x["frame_id"])/"receipt.json") for x in bundle["scientific_acquisition"]["frames"]]}
    compressed_json(artifact/"large_artifact_dependencies.json.gz", dependency)
    write_reports(binding, bundle, artifact, first, second, third)
    write_resource_report(bundle, artifact)
    return bundle


def write_resource_report(store, artifact):
    context = store["resource_comparison"]
    first = table(artifact, "resource_G1_G3", list(context["matched_G1_G3_rows"][0]), context["matched_G1_G3_rows"])
    atomic_write_json(artifact/"resource_comparison.json", {"single_result_store_identity": store["identity"], **context})
    g1, g3 = context["matched_G1_G3_rows"]
    report = "# 显存与可比耗时\n\n同机、同场景、同计时边界的 G1/G3：各 8 场景 × 2 次正式冷调用，包含必要文件输出。\n\n"+first
    report += (f"\nG3 比 G1 慢 {(g3['Seconds_scene']/g1['Seconds_scene']-1)*100:.2f}%；峰值 allocated 显存相同。"
        "reserved 是缓存分配器预留峰值，受分配历史影响，不能将该差值解释为模型节省显存。两列只覆盖 FC 恢复进程，未包含公共 Native/SigLIP/CropFormer 系统显存。\n\n")
    before = context["before_after_rows"]["G1_REFERENCE"]
    after = context["before_after_rows"]["G1_SELECTED"]
    report += (f"此前独立配对优化实验：G1 {before['mean_seconds']:.3f} → {after['mean_seconds']:.3f} 秒/场景，"
        f"加速 {context['matched_before_after_G1']['speedup']:.2f} 倍，耗时减少 {context['matched_before_after_G1']['reduction_fraction']*100:.2f}%；"
        f"峰值 allocated 均为 {before['peak_cuda_allocated_bytes_max']/2**30:.2f} GiB。该实验与上表的插桩实验分开，不能混用其分母。\n\n")
    report += "OVI-MAP 原表与我们的模块对照，单位 ms/处理帧；n 为每 n 个物理输入帧处理一次：\n\n"
    modules = context["module_comparison_rows"]
    report += table(artifact, "resource_module_reference", list(modules[0]), modules)
    depth_association_ms = sum(x["G1_ms"] for x in modules if x["Module"] in ("Depth segmentation", "2D--3D association"))
    report += (f"\nOVI-MAP：RTX3090 + i7-12700K；我们：A40 + Xeon Silver 4314。"
        f"[原论文表 8]({context['source']})为作者报告；其表 7/8 未报告峰值显存，参数量不能代替峰值显存。"
        "跨硬件且关联包装的计时范围不同，上表只用于定位差距，不形成公平的端到端加速比。视角与语义行是离线摊销成本，"
        "语义更新 n=10 仍是预算方案；RGB/深度/关联是实际 n=30 离线序列计时。\n\n"
        f"当前主要差距在深度分割与 2D–3D 关联。深度+关联同一执行器每个处理帧合计 {depth_association_ms:.2f} ms，"
        "在 n=30、30 FPS 输入下只有 1000 ms 预算；完整系统不满足该均值预算。没有实测连通的在线流水线。\n\n"
        "新证据探索的同边界四方法冷计时及显存：\n\n")
    report += (artifact/"table3_timing.md").read_text()
    report += "\n该四方法表包含驻留 FC+AnyUp、重算 G1/额外几何/FC/AnyUp/预测 payload；收据和预测一致性校验在计时外。与前面的包含文件导出旧 G1/G3 表保持独立。\n"
    repo = Path(__file__).resolve().parents[3]
    (repo/"docs/paper/static_ovmap/RESOURCE_COMPARISON.md").write_text(report)


def write_reports(binding, store, artifact, first, second, third):
    repo = Path(__file__).resolve().parents[3]
    docs = repo/"docs/paper/static_ovmap"
    science, selection, timing = store["science"], store["selection"], store["timing"]
    acquisition = store["scientific_acquisition"]
    target = bool(selection["passing_methods"])
    intro = (f"固定九方法 × 26 场景已完成，覆盖 {science['scene_method_coverage']}/234 场景行及 {science['full_cohort_pool_coverage']}/18 官方完整池化结果。"
        f"目标门槛：{'达到' if target else '未达到'}；统一研究推荐 `{selection['recommended_method']}`，部署 `N0_UNCHANGED`。\n\n"
        "Replica-8 与 ScanNet-CF18 均为已曝光队列；CF18 包含 18 次采集、7 个物理场景家族。结果属于回顾性研究选择，不是独立验证。\n\n")
    results = "# 固定证据探索实验结果\n\n"+intro+"指标单位为百分数；差值为百分点，参考 B1。AP 使用全队列官方池化。显示保留两位小数，选模使用未舍入数值。\n\n"+first
    results += "\n方法机制的配对比较如下。TP/FP 列是 IoU=.5 的确定性评分条目变化；唯一 GT 匹配变化、歧义条目和全部类级差值见 companion JSON。\n\n"+second
    results += "\n新配对冷计时：模型与公共几何/NQF 驻留，G1 搜索、额外几何、FC、AnyUp 与输出均在边界内；8 场景各测两轮，第二轮反序。系统页缓存及共享主机后台 CPU 负载未控制，初期另有只读配对诊断进程；未验证在线 30 FPS。\n\n"+third
    results += (f"\n科学获取新增 FC 编码 {acquisition['counts'].get('FC_image_inputs',0)} 次、AnyUp Q/K {acquisition['counts']['AnyUp_QK_computations']} 次。"
        "官方完整帧一致性验证额外 1 次 AnyUp Q/K，计时阶段的新推理另见 table3 与 timing_records.json。"
        "基线固定旧 rank 的两次诊断池化不属于官方性能。\n\n"
        "显存为 PyTorch allocator allocated/reserved 峰值，包含驻留模型；完整建图系统显存未测。模型加载时间、五阶段耗时、非可加 CUDA events、失败成本和全部重复保留在机器数据中。\n")
    prepared = store["selected_frame_geometry"]
    results += (f"\n成功的共享几何准备完成 {prepared['selected_frames']} 次完整选中帧投射；首次字段修复前另有 1 次 CPU 投射，隔离保留。"
        "共享盘 ENOSPC 中断的预测/评分叶没有新增 GPU 推理，但失败墙钟未在当时记录，记为 null，不补零。"
        "新冷计时的输出边界止于完整预测 payload 构造；预测哈希、一致性检查及收据写入在计时外，不能与旧版包含文件导出的计时直接作加速比。\n")
    (docs/"EVIDENCE_EXPLORATION_RESULTS.md").write_text(results)
    selection_text = "# 研究选模\n\n"+intro
    selection_text += "规则使用未舍入 fraction 与 1e-10 数值容差：两队列各五项指标均不低于 B1，CF18 APall 严格高于 D2，且 AP50 不低于 D2；0.10 pp 为另列的工程幅度标记。\n\n"
    for method, flags in selection["target_flags"].items():
        selection_text += f"- `{method}`：Replica 五项 {flags['replica_nondecrease_all5']}；CF18 五项 {flags['cf_nondecrease_all5']}；CF AP 条件 {flags['cf_AP_conditions']}；TARGET_MET {flags['target_met']}；MATERIAL_TARGET_MET {flags['material_target_met']}。\n"
    selection_text += "\n通过者按 CF APall、CF AP50、Replica APall、CF mIoU、固定方法索引决胜；无通过者保留 B1。没有逐场景或逐队列切换，也未追加阈值搜索。\n\n"
    selection_text += f"冷计时探索对象 `{selection['timing_candidate']}`，固定比较集：{', '.join(selection['timing_methods'])}。未新计时的精度对比不作速度结论。\n"
    (docs/"EVIDENCE_EXPLORATION_SELECTION.md").write_text(selection_text)
    claims = "# 证据与结论边界\n\n"+intro
    for row in store["paired_diagnostics"]["comparisons"]:
        delta = row["metric_deltas_fraction"]
        claims += f"- {row['cohort']}，`{row['candidate']}` 对 `{row['reference']}`：APall {delta['apall']*100:+.4f} pp、AP50 {delta['ap50']*100:+.4f} pp、mIoU {delta['miou']*100:+.4f} pp。\n"
    claims += "\n只有完整指标和匹配诊断支持的方法差异可作为研究结论。接受数量不是真实召回；owner/depth 约束不能保证恢复粗特征中已丢失的信息，竞争区域可来自同类邻居。几何不足与类别错误为非互斥诊断。\n\n"
    for cohort, counts in store["baseline_diagnostic_summary"]["by_cohort"].items():
        claims += (f"{cohort}：{counts['available']}/{counts['candidates']} 个候选有 B1 表征，其中 "
            f"{counts['geometry_insufficient_available_at50']} 个在 IoU=.5 下几何不足、{counts['target_small_available']} 个目标投影少于 100 点；"
            f"released scorer 实际忽略 {counts['actually_ignored_at50']} 个。三项非互斥，不相加。\n\n")
    claims += ("粗对比相对单纯 margin 未增加确定 TP50，CF18 APall 还下降约 0.026 pp，不能支持其额外有效性。"
        "普通 AnyUp 相对 bilinear 改善两队列的五项指标，但 Replica AP25 低于 B1，未达到固定目标。"
        "owner/depth 约束相对普通 AnyUp 使两队列 APall/AP50 下降；CF18 去深度后的五项池化指标完全相同，未显示深度的增益。"
        "组合在 CF18 相对其两个组件有所改善，但仍低于 D2 的 APall，且 Replica mIoU/mAcc 低于 B1。\n\n"
        "所有直接机制比较的唯一 GT50 匹配增减均为 0；AP 改变同时涉及类别、FP 与官方当前类别面积 rank，"
        "不能将其描述成新增真实召回，也不能仅用 TP50 解释 APall 的九阈值池化差异。\n\n")
    claims += "FC 与 AnyUp 均为冻结预训练模型；继承 Q/温度包含既有拟合，因此不宣称全系统无训练。没有新建图、前端、Native N/Q 推理、训练或拟合，也不宣称泛化、显著性、在线实时性能或 OVI-MAP 全方法复现。\n"
    (docs/"EVIDENCE_EXPLORATION_CLAIMS.md").write_text(claims)
    handoff = "# 交接与复现\n\n"+intro
    handoff += f"实现冻结提交 `{store['implementation_freeze']['commit']}`。唯一最终结果库为 `artifacts/static_ovmap/evidence_exploration_v1/study_result_store.json.gz`，身份 `{store['identity']}`。所有表格单元格由该库生成，来源见 cell_provenance.json。\n\n"
    handoff += "```bash\n/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_evidence_exploration.py --phase all --resume\n```\n\n"
    handoff += f"共享逻辑输出根 `{store['storage_relocation']['logical_root']}`；物理位置 `{store['storage_relocation']['physical_root']}`。共享盘耗尽后仅迁移本任务输出，逐文件 SHA256 校验，保留原路径符号链接和部分失败文件；原输入身份与成功结果继续复用。\n\n"
    handoff += "ENOSPC 失败叶的墙钟未记录，故为 null；成功评分与模型调用成本保留。首次准备字段修复前的额外 1 次 CPU 投射另计。后续诊断补充只处理锁定预测，新增 GPU/官方评分为 0。\n\n"
    handoff += "权重、完整 owner/depth 图、FC dense cache、ScanNet RGB/mesh 不进入 Git；精确依赖与重构入口见 large_artifact_dependencies.json.gz。AnyUp 固定 351807a9 与 paper 检查点 9d035c0f…，严格载入；源码上游未修改，新增流式 wrapper 保留 CC-BY-4.0 署名。\n\n"
    handoff += "模型/文本/类别顺序、G1/候选、D2 incumbent、官方 rank 定义均固定。GPU 2 单进程；评分 3 个 CPU 进程、BLAS ≤4、raycast 4。真实 office1/scene0011_00 pilots、focused tests、冷计时后预测一致性和逐项 requirement review 为验收证据。最终 GitHub full-SHA 发布收据位于提交之外的 output_root/publication/final.json。\n"
    (docs/"EVIDENCE_EXPLORATION_HANDOFF.md").write_text(handoff)


def publish(binding, root):
    # Requirement review is performed and stored before this final external write.
    root = Path(root)
    repo = Path(__file__).resolve().parents[3]
    from .review import review_study
    review = review_study(binding, root)
    if review["required_execution_verified"] is not True:
        raise ValueError("publication requires the completed requirement-by-requirement review")
    paths = ["src/static_ovmap/evidence_exploration", "scripts/evaluation/run_ovimap_evidence_exploration.py",
        "tests/evaluation/test_evidence_exploration.py", "docs/paper/static_ovmap/RESOURCE_COMPARISON.md",
        "docs/superpowers/plans/2026-10-07-evidence-exploration.md", "artifacts/static_ovmap/evidence_exploration_v1",
        *["docs/paper/static_ovmap/"+name for name in binding["specification"]["publication"]["reports"]]]
    subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True)
    subprocess.run(["git", "-c", "core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol", "diff", "--cached", "--check"], cwd=repo, check=True)
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=repo, text=True).strip()
    if staged:
        subprocess.run(["git", "commit", "-m", "Publish complete fixed evidence exploration results and paired timing"], cwd=repo, check=True)
    branch = binding["specification"]["branch"]
    subprocess.run(["git", "push", "origin", "HEAD:refs/heads/"+branch], cwd=repo, check=True)
    local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/"+branch], cwd=repo, text=True).split()[0]
    if local != remote:
        raise ValueError("remote branch full SHA differs from local HEAD")
    receipt = seal({"status": "PUSH_VERIFIED", "local_sha": local, "remote_sha": remote, "branch": branch,
        "requirement_review_identity": review["identity"], "receipt_outside_attested_commit": True})
    atomic_write_json(root/"publication/final.json", receipt)
    print("PUSH_VERIFIED", local, flush=True)
    return receipt
