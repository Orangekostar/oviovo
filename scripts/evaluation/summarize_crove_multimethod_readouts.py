"""Collect all scored DEV readouts without selecting or opening new GT."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_crove_adapter_room0 import path
from scripts.evaluation.run_ovimap_native import _atomic_json, _sha256

# Explicit method contracts also keep diagnostics and input-status JSON out.
METHODS = {
    "MV_TOPK_PATCH_ONLY": ("M1+M2", "graph_mv_topk", "MV_TOPK_MEAN"),
    "MV_TOPK_GRAPH_GEOM": ("M1+M2", "graph_mv_topk", "MV_TOPK_PATCH_ONLY"),
    "B_SEM_OVI_NATIVE": ("BASELINE", "native_cached_batch", None),
    "MV_NATIVE_CACHED": ("BASELINE", "native_cached_batch", None),
    "B_SEM_CROVE_S0": ("BASELINE", "semantic_controls", None),
    "B_SEM_CROVE_S2": ("BASELINE", "semantic_controls", "B_SEM_CROVE_S0"),
    "B_SEM_CROVE_S0_DENSE_REPLAY": ("BASELINE", "local_semantic_controls", None),
    "B_SEM_CROVE_S2_DENSE_REPLAY": (
        "BASELINE",
        "local_semantic_controls",
        "B_SEM_CROVE_S0_DENSE_REPLAY",
    ),
    "MV_SINGLE": ("M1", "native_cached_batch", "MV_NATIVE_CACHED"),
    "MV_TOPK_MEAN": ("M1", "native_cached_batch", "MV_NATIVE_CACHED"),
    "MV_DIVERSE_MEAN": ("M1", "native_diverse_quality", "MV_TOPK_MEAN"),
    "MV_QUALITY": ("M1", "native_diverse_quality", "MV_DIVERSE_MEAN"),
    "MV_CURRENT_AWARE": ("M1", "current_aware", "MV_REENCODE_QUALITY"),
    "B_SEM_OVI_REENCODE": ("BASELINE", "reencoded_regions", None),
    "MV_REENCODE_DIVERSE": ("M1", "reencoded_regions", "B_SEM_OVI_REENCODE"),
    "MV_REENCODE_QUALITY": ("M1", "reencoded_regions", "MV_REENCODE_DIVERSE"),
    "MV_BG_OWNER_QUALITY": ("M1", "local_background", "B_SEM_OVI_REENCODE"),
    "MV_LOCAL_BG_QUALITY": ("M1", "local_background", "MV_BG_OWNER_QUALITY"),
    "S2_PATCH_ONLY": ("M2", "graph_s2", "B_SEM_CROVE_S2"),
    "S2_GRAPH_GEOM": ("M2", "graph_s2", "S2_PATCH_ONLY"),
    "S2_GRAPH_BOUNDARY": ("M2", "graph_s2_boundary", "S2_GRAPH_GEOM"),
    "S2_DENSE_REPLAY_PATCH_ONLY": (
        "M2",
        "graph_s2_dense_replay",
        "B_SEM_CROVE_S2_DENSE_REPLAY",
    ),
    "S2_DENSE_REPLAY_GRAPH_GEOM": (
        "M2",
        "graph_s2_dense_replay",
        "S2_DENSE_REPLAY_PATCH_ONLY",
    ),
    "S2_DENSE_REPLAY_GRAPH_BOUNDARY": (
        "M2",
        "graph_s2_dense_replay_boundary",
        "S2_DENSE_REPLAY_GRAPH_GEOM",
    ),
    "MV_QUALITY_PATCH_ONLY": ("M1+M2", "graph_mv_quality", "MV_QUALITY"),
    "MV_QUALITY_GRAPH_GEOM": ("M1+M2", "graph_mv_quality", "MV_QUALITY_PATCH_ONLY"),
    "MV_QUALITY_GRAPH_BOUNDARY": (
        "M1+M2",
        "graph_mv_quality_boundary",
        "MV_QUALITY_GRAPH_GEOM",
    ),
    "INST_PAIRWISE": ("M3", "consensus", None),
    "INST_CONSENSUS_OWNER": ("M3", "consensus", "INST_PAIRWISE"),
    "INST_CONSENSUS_RESEM": ("M3+M1", "reencoded_regions", "INST_CONSENSUS_OWNER"),
    "ADAPTER_CLIP_MEAN": ("M4", "adapter_projected_top4", None),
    "ADAPTER_CLIP_LEARNED": ("M4", "adapter_projected_top4", "ADAPTER_CLIP_MEAN"),
    "ADAPTER_LEARNED_PATCH_ONLY": (
        "M4+M2",
        "graph_adapter_learned",
        "ADAPTER_CLIP_LEARNED",
    ),
    "ADAPTER_LEARNED_GRAPH_GEOM": (
        "M4+M2",
        "graph_adapter_learned",
        "ADAPTER_LEARNED_PATCH_ONLY",
    ),
}


def per_class_delta(metrics, baseline):
    current = metrics.get("semantic", {}).get("per_class", {})
    reference = baseline.get("semantic", {}).get("per_class", {})
    if not current or not reference:
        return None
    return {
        key: {
            "iou": current.get(key, {}).get("iou"),
            "baseline_iou": reference.get(key, {}).get("iou"),
            "delta": current[key]["iou"] - reference[key]["iou"]
            if key in current and key in reference
            else None,
            "GT_support": current.get(key, {}).get("support"),
        }
        for key in sorted(set(current) | set(reference), key=int)
    }


def main():
    config = json.loads(
        (ROOT / "configs/evaluation/crove_multimethod_readout_v1.json").read_text()
    )
    compact, run = path(config["compact_output_root"]), path(config["run_root"])
    geometry_audit = json.loads(
        (compact / "apartment_readout_geometry_audit.json").read_text()
    )
    rows = []
    for case, state in (("room0", None), ("apartment", "B3"), ("apartment", "H2")):
        prefix = case if state is None else f"{case}_{state}"
        records = {}
        for filename in sorted(compact.glob(f"{prefix}_*.json")):
            data = json.loads(filename.read_text())
            if data.get("status") != "FULL_MAP_EVALUATED" or "metrics" not in data:
                continue
            method = data["method"]
            if method not in METHODS or method in records:
                raise ValueError(f"unknown or duplicate scored method: {filename}")
            records[method] = (filename, data)
        native_name = "B_SEM_OVI_NATIVE" if state is None else "MV_NATIVE_CACHED"
        s2_name = "B_SEM_CROVE_S2" if state is None else "B_SEM_CROVE_S2_DENSE_REPLAY"
        metric_key = "miou" if state is None else "current_miou"
        native_metrics = records[native_name][1]["metrics"]
        s2_metrics = records[s2_name][1]["metrics"]
        room = run / "dev" / case
        with np.load(
            room
            / "native_cached_batch"
            / f"{'' if state is None else state + '_'}{native_name}.npz"
        ) as data:
            source, owners = data["source_indices"], data["owner_ids"]
        for method, (filename, data) in records.items():
            family, directory, direct = METHODS[method]
            prediction_method = method
            if state is None and method == "MV_CURRENT_AWARE":
                directory, prediction_method, direct = (
                    "native_diverse_quality",
                    "MV_QUALITY",
                    "MV_QUALITY",
                )
            if method == "MV_BG_OWNER_QUALITY":
                direct = native_name
            prediction = (
                room
                / directory
                / f"{'' if state is None else state + '_'}{prediction_method}.npz"
            )
            with np.load(prediction) as values:
                if not np.array_equal(values["source_indices"], source):
                    raise ValueError(f"source geometry changed: {prediction}")
                changed_owners = int(np.count_nonzero(values["owner_ids"] != owners))
                if not method.startswith("INST_") and changed_owners:
                    raise ValueError(f"semantic readout changed owners: {prediction}")
                ids = values["semantic_ids"]
                if len(ids) != len(source) or np.any(ids < 0):
                    raise ValueError(f"invalid source semantic readout: {prediction}")
                unknown = int(np.count_nonzero(ids == 0))
            metrics = data["metrics"]
            score = metrics[metric_key]
            if not np.isfinite(score):
                raise ValueError(f"nonfinite primary metric: {filename}")
            instance = metrics.get("instance", {}).get("class_agnostic", {})
            if direct is not None and direct not in records:
                raise ValueError(f"missing direct control {direct} for {filename}")
            audit = geometry_audit["states"][state] if state else None
            ghost_counts = (
                audit["readouts"][method].get("role_conditioned_ghost", {})
                if audit
                else {}
            )
            row = {
                "family": family,
                "exact_implementation": method,
                "case": case,
                "state_variant": state,
                "split": "dev",
                "actual_status": data["status"],
                "result_file": filename.name,
                "result_sha256": _sha256(filename),
                "prediction": str(prediction.relative_to(run)),
                "source_rows": len(source),
                "geometry_fixed": True,
                "owner_changed": bool(changed_owners),
                "changed_owner_rows": changed_owners,
                "unknown_rows": unknown,
                "unknown_rate": unknown / len(source),
                "static_miou": score if state is None else None,
                "f_miou": metrics.get("f_miou"),
                "CA_AP25": instance.get("ap25"),
                "CA_AP50": instance.get("ap50"),
                "CA_AR50": instance.get("recall50"),
                "predicted_instance_count": instance.get("predicted_instance_count"),
                "dynamic_current_miou": score if state else None,
                "Ghost": metrics.get("ghost"),
                "Ghost_numerator": ghost_counts.get("numerator"),
                "Ghost_denominator": ghost_counts.get("denominator"),
                "BG_F5": metrics.get("background_f1_at_5cm"),
                "Surface_F5": metrics.get("surface_f1_at_5cm", metrics.get("f5")),
                "Surface_precision": metrics.get("surface_precision_at_5cm"),
                "all_current_free_conflict_count": audit["full_current_geometry"][
                    "confirmed_free_conflict_rows"
                ]
                if audit
                else None,
                "all_current_free_conflict_denominator": audit["full_current_geometry"][
                    "source_rows"
                ]
                if audit
                else None,
                "all_current_conflicting_5cm_voxels": audit["full_current_geometry"][
                    "conflicting_5cm_voxels"
                ]
                if audit
                else None,
                "delta_vs_native_OVI": score - native_metrics[metric_key],
                "S2_reference": s2_name,
                "delta_vs_S2": score - s2_metrics[metric_key],
                "direct_control": direct,
                "delta_vs_direct_control": score
                - records[direct][1]["metrics"][metric_key]
                if direct
                else None,
                "per_class_delta_vs_native": per_class_delta(metrics, native_metrics),
                "per_class_delta_vs_S2": per_class_delta(metrics, s2_metrics),
                "feature_coverage": data.get("feature_coverage"),
                "runtime_stages": {
                    k: v
                    for k, v in data.items()
                    if k.endswith("seconds") or k in ("shared_runtime", "timing_scope")
                },
                "peak_memory_bytes": data.get("peak_gpu_bytes"),
                "new_encoder_forwards": data.get(
                    "new_encoder_forwards", data.get("extra_dense_encoder_forwards")
                ),
                "training_updates": data.get("training_updates"),
                "passes_original_per_case_gates": data.get(
                    "passes_original_per_case_gates"
                ),
                "selection_policy_id": config["selection_policy"]["id"],
                "selection_status": "NOT_FROZEN",
            }
            rows.append(row)
        print(prefix, len(records), "scored readouts collected", flush=True)
    missing = [
        f"{case}_{method}"
        for case in ("room0", "apartment_B3", "apartment_H2")
        for method in ("MV_BG_OWNER_QUALITY", "MV_LOCAL_BG_QUALITY")
        if not (compact / f"{case}_{method}.json").exists()
    ]
    report = {
        "status": "INTERMEDIATE_ALL_SCORED_DEV_READOUTS",
        "selection_frozen": False,
        "GT_opened_by_collector": False,
        "unknown_definition": "source semantic ID equals zero; source-row fraction, not voxel-weighted",
        "missing_value_rule": "null means not recorded or not applicable; no inferred zero runtime or dynamic instance AP",
        "dynamic_S2_scope": "actual same-visit legacy dense replay baseline; not a full-vocabulary VLM posterior",
        "missing_local_background_results": missing,
        "rows": rows,
    }
    _atomic_json(compact / "all_method_results.json", report)
    export_tables(compact, report)


def export_tables(compact, report):
    rows = report["rows"]
    missing = report["missing_local_background_results"]
    flat_columns = [
        key
        for key in rows[0]
        if key
        not in ("runtime_stages", "per_class_delta_vs_native", "per_class_delta_vs_S2")
    ]
    target = compact / "all_method_results.csv"
    temporary = target.with_suffix(".partial")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=flat_columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    lines = [
        "# CROVE 四类方法结果（DEV 中间汇总）",
        "",
        "本表收录全部已评分全图结果，不执行选型。确认评分与最终配置冻结尚未完成。",
        "source 行与 owner 逐文件核对；unknown 为源行中标签 0 的比例。",
        "动态 AP 无对应 GT，保持 N/A；代价仅保留原结果明确记录的阶段，不填估计值。",
        "完整逐类增量、原始结果哈希及耗时字段见 [JSON](all_method_results.json)，平面字段见 [CSV](all_method_results.csv)。",
        "",
        "## 四类结果解释",
        "",
        "- M1：原缓存、匹配重编码和局部背景分别保留直接对照；局部背景矩阵未齐前不冻结家族赢家。",
        "- M2：patch-only、纯几何图与边界图分开报告；相对直接对照的增量用于区分池化与传播收益，不能解释为几何新增。",
        "- M3：owner-only 的主任务是静态 CA-AP50；语义重估另列，动态无实例 GT，不能据动态 mIoU 反选聚类参数。",
        "- M4：普通池化与官方训练后 Adapter 使用相同编码器和投影输入；图组合单列。checkpoint 已训练不等于本轮发生训练更新。",
        "",
        "## 全部已评分行",
        "",
        "| 家族 | case/state | 方法 | mIoU | Δ native | Δ直接对照 | unknown | CA-AP50 | 动态门槛 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    def number(value):
        return "N/A" if value is None else f"{value:.6f}"

    for row in rows:
        state = row["state_variant"]
        case_label = row["case"] + (f"/{state}" if state else "")
        score = row["dynamic_current_miou"] if state else row["static_miou"]
        gate = row["passes_original_per_case_gates"]
        gate_label = "N/A" if gate is None else ("PASS" if gate else "FAIL")
        lines.append(
            f"| {row['family']} | {case_label} | {row['exact_implementation']} | "
            f"{number(score)} | {number(row['delta_vs_native_OVI'])} | "
            f"{number(row['delta_vs_direct_control'])} | {number(row['unknown_rate'])} | "
            f"{number(row['CA_AP50'])} | {gate_label} |"
        )
    lines += [
        "",
        f"局部背景待评分行数：{len(missing)}。最终逐家族采用结论、计算收益与下一轮优先级在完整 DEV 矩阵后更新。",
        "",
    ]
    target = compact / "family_results.md"
    temporary = target.with_suffix(".partial")
    temporary.write_text("\n".join(lines))
    temporary.replace(target)
    print(
        len(rows),
        "full readouts;",
        len(missing),
        "local background results pending",
        flush=True,
    )


if __name__ == "__main__":
    main()
