"""Complete-case per-scene frozen transfer measurements, never imputed results."""

import csv
import io
from pathlib import Path

from src.static_ovmap.composition_study.execution import verified_receipt
from src.static_ovmap.composition_study.io import SourceIndex, read_json, write_once
from src.static_ovmap.composition_study.reporting import (
    METRICS,
    compare_methods,
    metric_summary,
)
from src.static_ovmap.composition_study.selection import TOLERANCE

from .protocol import SOURCE_ROOT


def complete_metrics(summary, scenes):
    return summary["status"] == "COMPLETE" and all(
        summary["denominators"][key]["defined"] == len(scenes) for key in METRICS
    )


def compare_all_metrics(rows, role, candidate, reference, scenes):
    comparison = compare_methods(rows, role, candidate, reference, scenes)
    left, right = (metric_summary(rows, role, method, scenes) for method in (candidate, reference))
    if not all(complete_metrics(value, scenes) for value in (left, right)):
        return {"status": "INCONCLUSIVE_INCOMPLETE_ROWS", "candidate": candidate,
                "reference": reference, "role": role}
    by_key = {(row["scene_id"], row["method_id"]): row for row in rows if row["role"] == role}
    deltas = {scene: {key: by_key[scene, candidate]["metrics"][key] - by_key[scene, reference]["metrics"][key]
                      for key in METRICS} for scene in scenes}
    mean = {key: left["means"][key] - right["means"][key] for key in METRICS}
    nonnegative = all(value >= -TOLERANCE for row in deltas.values() for value in row.values())
    mean_gain = all(value >= -TOLERANCE for value in mean.values()) and any(value > TOLERANCE for value in mean.values())
    status = ("MEAN_GAIN_NO_OBSERVED_SCENE_LOSS" if nonnegative else "MEAN_GAIN_WITH_SCENE_TRADEOFF") if mean_gain else "NO_MEAN_GAIN"
    return {**comparison, "status": status, "status_metrics": list(METRICS), "mean_deltas": mean,
            "scene_deltas": deltas, "every_scene_nonnegative": nonnegative,
            "every_scene_strictly_positive": all(value > TOLERANCE for row in deltas.values() for value in row.values()),
            "worst_scene_deltas": {key: min(row[key] for row in deltas.values()) for key in METRICS},
            "positive_scene_counts": {key: sum(row[key] > TOLERANCE for row in deltas.values()) for key in METRICS}}


def summarize(rows, scenes, methods):
    keys = [(r["scene_id"], r["method_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate scene/method result")
    expected = {(scene, method) for scene in scenes for method in methods}
    if not set(keys) <= expected:
        raise ValueError("unlisted scene/method result")
    missing = sorted(expected - set(keys))
    means = {method: metric_summary(rows, "replica_transfer", method, scenes) for method in methods}
    comparisons = {}
    for candidate in methods:
        for reference in ("N0", "Q_GAIN", "S_SIGLIP2_AREA"):
            if reference == candidate or reference not in methods:
                continue
            comparison = compare_all_metrics(rows, "replica_transfer", candidate, reference, scenes)
            comparisons[candidate + "_vs_" + reference] = comparison
    complete = not missing and all(complete_metrics(value, scenes) for value in means.values())
    return {"status": "COMPLETE" if complete else "INCOMPLETE", "expected_rows": len(expected),
            "completed_rows": len(rows), "missing": missing, "means": means, "comparisons": comparisons}


def report(config):
    root = Path(config["attempt_root"])
    rows, costs, audits = [], {}, {}
    index = SourceIndex()
    for scene in config["scenes"]:
        local = root / "scenes" / scene
        for method in config["methods"]:
            path = local / "rows/replica_transfer" / scene / (method + ".json")
            if not path.is_file():
                continue
            row = read_json(path)
            verified_receipt(row["evaluation_receipt"], identity=row["evaluation_identity"])
            index.identity(path)
            rows.append(row)
        paths = {
            "frontend": root / "native" / scene / "frontend_job/receipt.json",
            "mapping": root / "native" / scene / "mapping_job/receipt.json",
            "static_siglip2": local / "study/scenes" / scene / "semantic_models/siglip2/receipt.json",
            "Q_GAIN": local / "query" / scene / "Q_GAIN/receipt.json",
            "CP_M4_GAIN_S2": local / "query" / scene / "CP_M4_GAIN_S2/receipt.json",
        }
        costs[scene] = {}
        for name, path in paths.items():
            if path.is_file():
                receipt = verified_receipt(path)
                index.identity(path)
                costs[scene][name] = {"receipt": str(path), **{key: receipt[key] for key in (
                    "elapsed_seconds", "request_seconds", "load_seconds", "physical", "physical_attempts",
                    "physical_crop_inputs", "logical", "required_logical") if key in receipt}}
        query = paths["Q_GAIN"].with_name("decisions.json")
        reader = paths["CP_M4_GAIN_S2"].with_name("decisions.json")
        if query.is_file() and reader.is_file():
            q, s = read_json(query), read_json(reader)
            # Reader executes the same actual paid inputs, while its own lineage
            # and feature retention are left untouched and recorded separately.
            same = q["paid_requests"] == s["paid_requests"]
            if not same:
                raise ValueError(f"forced M4 paid trajectory differs: {scene}")
            audits[scene] = {"same_paid_trajectory": same, "query_trace": str(query), "reader_trace": str(reader),
                             "query_retained": q["retained_features"], "reader_retained": s["retained_features"]}
    summary = summarize(rows, config["scenes"], config["methods"])
    if summary["status"] != "COMPLETE":
        return summary
    result = {**summary, "dataset": "Replica", "transfer_identity": config["identity"],
              "no_replica_fitting": True, "historical_replica_exposure": True,
              "semantic_classes": 51, "released_ap_classes": 48,
              "aggregation": "equal scene mean; paired within-dataset deltas",
              "rows": rows, "physical_jobs_counted_once": costs, "trajectory_audits": audits,
              "source_manifest": index.manifest()}
    result["cpu_text_jobs_counted_once"] = {
        name: verified_receipt(root / "text" / name / "receipt.json")
        for name in ("native", "siglip2")
    }
    source_config = read_json(SOURCE_ROOT / "resolved_config.json")
    result["scannet_reference_by_original_role"] = {}
    for role in ("compose_cal", "regression_only", "confirmation"):
        source_rows = [read_json(p) for p in sorted((SOURCE_ROOT / "rows" / role).glob("*/*.json"))]
        source_scenes = source_config["spec"]["data"][role]
        result["scannet_reference_by_original_role"][role] = {
            "scenes": source_scenes,
            "means": {m: metric_summary(source_rows, role, m, source_scenes) for m in config["methods"]},
            "paired_changes": {
                name: compare_all_metrics(source_rows, role, value["candidate"], value["reference"], source_scenes)
                for name, value in summary["comparisons"].items()
            },
            "source_rows": [str(p) for p in sorted((SOURCE_ROOT / "rows" / role).glob("*/*.json"))],
        }
    output = root / "report"
    write_once(output / "results.json", result)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["scene", "method", *METRICS, "changed_owners", "logical_requests"])
    for row in rows:
        writer.writerow([row["scene_id"], row["method_id"], *[row["metrics"].get(k) for k in METRICS], row["changed_owners"], row["logical_requests"]])
    (output / "per_scene.csv").write_text(stream.getvalue())
    lines = ["# Replica 冻结迁移验证", "", "8 个场景，每场景 200 帧；模型、Q 权重及温度从 ScanNet 冻结迁移，无 Replica 拟合。",
             "Replica 存在历史实验暴露，不能称为未接触留出集。语义评价 51 类，发布版实例 AP 48 类。", "",
             "| 方法 | uAP | AP50 | AP25 | mIoU | mAcc |", "|---|---:|---:|---:|---:|---:|"]
    for method, values in summary["means"].items():
        cells = ["未定义" if values["means"][key] is None else f'{100 * values["means"][key]:.3f}' for key in METRICS]
        lines.append("| " + " | ".join([method, *cells]) + " |")
    lines += ["", "表中指标为百分数，按场景等权平均。", "", "| 对比 | ΔuAP（百分点） | ΔmIoU（百分点） | uAP 提升场景数 | mIoU 提升场景数 |", "|---|---:|---:|---:|---:|"]
    for name, value in summary["comparisons"].items():
        if value["candidate"] not in ("CP_M2_EQUAL_RAW", "CP_M2_EQUAL_CAL", "CP_M4_GAIN_S2"):
            continue
        lines.append(f'| {name} | {100 * value["mean_deltas"]["uap"]:+.3f} | {100 * value["mean_deltas"]["miou"]:+.3f} | {value["positive_scene_counts"]["uap"]}/8 | {value["positive_scene_counts"]["miou"]}/8 |')
    lines += ["", "| 对比 N0 | ΔAP25（百分点） | ΔAP50（百分点） | ΔmAcc（百分点） | AP25 / AP50 / mAcc 提升场景数 |", "|---|---:|---:|---:|---|"]
    for method in config["methods"]:
        if method == "N0":
            continue
        value = summary["comparisons"][method + "_vs_N0"]
        cells = [f'{100 * value["mean_deltas"][key]:+.3f}' for key in ("ap25", "ap50", "macc")]
        counts = " / ".join(f'{value["positive_scene_counts"][key]}/8' for key in ("ap25", "ap50", "macc"))
        lines.append("| " + " | ".join([method, *cells, counts]) + " |")
    lines += ["", "ScanNet 参考结果按原 CAL、回归、确认角色分别保存在 results.json；未运行的方法标为缺失，不补零。跨数据集仅比较各自内部的变化方向，不直接比较绝对分数。",
              "", "逐场景结果、最差场景差值、完整轨迹和成本见 results.json 与 per_scene.csv。公共捕获、静态 S2、Q_GAIN 和 M4 的实际执行费用分别计一次；方法逻辑费用不能相加当作总实际费用。", ""]
    (output / "results.md").write_text("\n".join(lines))
    return {"status": "COMPLETE", "completed_rows": len(rows), "report": str(output / "results.md"), "means": summary["means"]}
