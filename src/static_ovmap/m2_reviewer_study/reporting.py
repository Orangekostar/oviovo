"""Evidence-derived tables A–F; missing stages cannot become final reports."""

import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.module_validation.contracts import canonical_digest

from .audit import validate_matrix
from .binding import InputIndex
from .cost_ledger import build_cost_ledger
from .diagnostics import bootstrap_delta, probability_metrics
from .evaluation import write_gzip

METRICS = ("uap", "ap25", "ap50", "miou", "macc")
RANKS = ("FROZEN_N0", "OFFICIAL_CURRENT_CLASS")


def assert_report_ready(states):
    for stage in ("core", "query", "diagnostics", "robustness", "fresh"):
        allowed = {"COMPLETE"} if stage != "fresh" else {"FRESH_CONFIRMATION_COMPLETE", "FRESH_BLOCKED_INSUFFICIENT_LOCAL_UNEXPOSED_FAMILIES"}
        if states.get(stage) not in allowed:
            raise ValueError(f"report prerequisite incomplete: {stage}: {states.get(stage)}")


def collect_report(binding, *, allow_partial=False):
    root, index = Path(binding["output_root"]), InputIndex()
    for name in ("reporting.py", "claims.py", "cost_ledger.py", "audit.py"):
        index.identity(Path(__file__).with_name(name))
    missing = []

    def read(path, stage):
        path = Path(path)
        if not path.is_file():
            missing.append({"stage": stage, "path": str(path)})
            return None
        index.identity(path)
        return read_json(path)

    spec = read_json(binding["spec"])
    replica = spec["datasets"]["official_replica_pool_order"]
    scene_rows, pool_rows = [], []
    for scene in binding["scenes"]:
        for method in binding["methods"]:
            for rank in RANKS:
                row = read(root / "rows" / scene / method / (rank + ".json"), "core")
                if row and row["status"] == "COMPLETE":
                    scene_rows.append(row)
    for dataset in ("ScanNet", "Replica"):
        for method in binding["methods"]:
            for rank in RANKS:
                row = read(root / "pooled" / dataset / method / (rank + ".json"), "core")
                if row and row.get("status") == "COMPLETE":
                    pool_rows.append({**row, "dataset": dataset, "rank_mode": rank, "method": method})
    query = {d: read(root / "query_controls" / (d + "_summary.json"), "query") for d in ("ScanNet", "Replica")}
    fresh = read(root / "fresh/status.json", "fresh")
    gate = read(root / "query_controls/curve_gate.json", "query")
    diagnostics, objects, mechanisms, attribution = {}, {}, {}, {}
    for scene in binding["scenes"]:
        diagnostics[scene] = read(root / "diagnostics" / scene / "summary.json", "diagnostics")
        mechanisms[scene] = read(root / "diagnostics" / scene / "mechanisms.json", "diagnostics")
        attribution[scene] = read(root / "diagnostics" / scene / "attribution_AP25_AP50.json", "diagnostics")
        path = root / "diagnostics" / scene / "objects.json.gz"
        if path.is_file():
            index.identity(path)
            with gzip.open(path, "rt") as handle:
                objects[scene] = json.load(handle)
        else:
            missing.append({"stage": "diagnostics", "path": str(path)})
    robust = {s: read(root / "robustness/results" / s / "summary.json", "robustness") for s in replica}
    costs, cost_path = build_cost_ledger(binding)
    query_expected = 300 + (192 if gate and gate["triggered"] else 0)
    query_complete = all(v and v["status"] == "COMPLETE" for v in query.values()) and sum(v["scene_rows"] for v in query.values() if v) == query_expected
    states = {"core": "COMPLETE" if len(scene_rows) == 260 and len(pool_rows) == 52 else "INCOMPLETE",
              "query": "COMPLETE" if query_complete and costs["completed_B200_jobs"] == 50 else "INCOMPLETE",
              "diagnostics": "COMPLETE" if len(objects) == 10
                  and all(d and d["status"] == "OBJECT_AND_PROBABILITY_COMPLETE" and len(objects.get(s, [])) == d["owners"] for s, d in diagnostics.items())
                  and all(m and m["status"] == "COMPLETE" for m in mechanisms.values()) and all(attribution.values()) else "INCOMPLETE",
              "robustness": "COMPLETE" if all(v and v["status"] == "COMPLETE" for v in robust.values()) else "INCOMPLETE",
              "fresh": fresh["status"] if fresh else "INCOMPLETE"}
    if not allow_partial:
        assert_report_ready(states)
    matrix_audit = {}
    if states["core"] == "COMPLETE":
        matrix_audit["core"] = validate_matrix(scene_rows, pool_rows,
            datasets={"ScanNet": spec["datasets"]["calibration"], "Replica": replica},
            methods=binding["methods"], ranks=RANKS)
    if states["query"] == "COMPLETE":
        for dataset, summary in query.items():
            methods = sorted({row["method"] for row in summary["pooled"]})
            expected_methods = [f"{p}_B200" if mode == "STANDALONE" else f"RV_B_{p}_B200_{mode}"
                                for p in ("Q_GAIN", "Q_COMBINE", "RV_Q_RANDOM_s17", "RV_Q_RANDOM_s23", "RV_Q_RANDOM_s41")
                                for mode in ("STANDALONE", "RAW", "CAL")]
            if dataset == "Replica" and gate["triggered"]:
                expected_methods += [f"{p}_B{budget}" if mode == "STANDALONE" else f"RV_B_{p}_B{budget}_{mode}"
                                     for budget in (100, 400) for p in ("Q_GAIN", "Q_COMBINE")
                                     for mode in ("STANDALONE", "RAW", "CAL")]
            if methods != sorted(expected_methods):
                raise ValueError("query matrix methods disagree with frozen gate")
            scenes = spec["datasets"]["calibration"] if dataset == "ScanNet" else replica
            measured = [read(root / "rows" / s / m / (r + ".json"), "query")
                        for s in scenes for m in methods for r in RANKS]
            matrix_audit["query_" + dataset] = validate_matrix(measured, summary["pooled"],
                datasets={dataset: scenes}, methods=methods, ranks=RANKS)
    macro = []
    for dataset, scenes in (("ScanNet", spec["datasets"]["calibration"]), ("Replica", replica)):
        for method in binding["methods"]:
            for rank in RANKS:
                selected = [r for r in scene_rows if r["scene"] in scenes and r["method"] == method and r["rank_mode"] == rank]
                if len(selected) != len(scenes):
                    continue
                macro.append({"dataset": dataset, "method": method, "rank_mode": rank, "aggregation": "SCENE_MACRO",
                              "scenes": len(scenes), "metrics": {k: float(np.mean([r["metrics"][k] for r in selected])) for k in (*METRICS, "apall")}})
    probability, pairs = [], []
    for scene, value in diagnostics.items():
        if not value:
            continue
        for method, populations in value["probability"].items():
            for population, metrics in populations.items():
                probability.append({"dataset": value["dataset"], "scene": scene, "method": method, "population": population,
                                    **{k: v for k, v in metrics.items() if k not in ("bins", "owner_ids")}})
    if all(s in objects for s in replica):
        config = read_json(binding["scenes"][replica[0]]["config"])
        ids = config["models"]["native"]["valid_ids"]
        for method in objects[replica[0]][0]["methods"]:
            for population in ("paired", "union"):
                rows = [r for s in replica for r in objects[s] if r["identifiable"]
                        and (population == "union" or r["paired_population"])
                        and r["methods"][method]["probabilities"] is not None]
                quality = probability_metrics([r["methods"][method]["probabilities"] for r in rows],
                    [ids.index(r["gt_label"]) for r in rows], [ids.index(r["methods"][method]["label"]) for r in rows])
                probability.append({"dataset": "Replica", "scene": "OBJECT_POOL", "method": method, "population": population,
                                    **{k: v for k, v in quality.items() if k != "bins"}})
        for rank in RANKS:
            primary = {r["scene"]: r for r in scene_rows if r["method"] == "CP_M2_EQUAL_CAL" and r["rank_mode"] == rank and r["scene"] in replica}
            for method in binding["methods"]:
                baseline = {r["scene"]: r for r in scene_rows if r["method"] == method and r["rank_mode"] == rank and r["scene"] in replica}
                if len(primary) == len(baseline) == 8:
                    pairs.append({"primary": "CP_M2_EQUAL_CAL", "comparator": method, "rank_mode": rank,
                                  "scene_order": replica, "metrics": {k: bootstrap_delta([primary[s]["metrics"][k] - baseline[s]["metrics"][k] for s in replica]) for k in METRICS}})
    try:
        assert_report_ready(states)
        ready = True
    except ValueError:
        ready = False
    result = {"status": "FINAL_EVIDENCE_READY" if ready else "PARTIAL",
              "binding": binding["identity"], "states": states, "missing": missing, "matrix_audit": matrix_audit, "scene_rows": scene_rows,
              "pooled_rows": pool_rows, "macro_rows": macro, "query": query, "probability_rows": probability,
              "paired_macro_comparisons": pairs, "diagnostics": diagnostics, "mechanisms": mechanisms,
              "attribution_AP25_AP50": attribution, "robustness": robust, "fresh": fresh, "curve_gate": gate,
              "cost_ledger": str(cost_path / "ledger.json"), "cost_identity": costs["identity"], "inputs": index.entries()}
    result["identity"] = canonical_digest(result)
    output = root / "reports" / result["identity"]
    write_once(output / "report.json", result)
    write_gzip(output / "object_attribution.json.gz", objects)
    return result, output


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    path.write_text(stream.getvalue())


def _table(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "—")) for k in fields) + " |")
    return "\n".join(lines)


def write_report_tables(binding, *, allow_partial=False):
    report, output = collect_report(binding, allow_partial=allow_partial)
    flattened = lambda r: {**{k: r.get(k) for k in ("dataset", "scene", "method", "rank_mode", "aggregation")}, **r["metrics"]}
    _csv(output / "metrics_scene.csv", [flattened(r) for r in report["scene_rows"]])
    _csv(output / "metrics_pooled.csv", [flattened(r) for r in report["pooled_rows"]])
    _csv(output / "metrics_macro.csv", [flattened(r) for r in report["macro_rows"]])
    _csv(output / "calibration_quality.csv", report["probability_rows"])
    costs = read_json(report["cost_ledger"])
    _csv(output / "costs.csv", costs["method_requirements"])
    query_rows = [flattened(r) for q in report["query"].values() if q for r in [*q["pooled"], *q["macro"]]]
    _csv(output / "query_metrics.csv", query_rows)
    write_once(output / "paired_macro_comparisons.json", report["paired_macro_comparisons"])
    rows_a = [{"method": row["method"], "rank": row["rank_mode"], "aggregation": row.get("aggregation", "RELEASED_DATASET_POOL"),
               **{k: f"{100 * row['metrics'][k]:.3f}" for k in METRICS}}
              for row in [*report["macro_rows"], *report["pooled_rows"]] if row["dataset"] == "Replica"]
    rows_b = [{"method": r["method"], "N": r["objects"], **{k: f"{r[k]:.5f}" for k in ("nll", "brier", "ece15", "accuracy")}}
              for r in report["probability_rows"] if r["dataset"] == "Replica" and r["scene"] == "OBJECT_POOL" and r["population"] == "paired"]
    rows_c = [{"scene": s, "identifiable": d["identifiable"], **{k: d["outcomes"]["CP_M2_EQUAL_CAL"].get(k, 0)
              for k in ("CORRECTED", "HARMED", "SOFT_RESCUE", "UNUSED_CORRECT_SOURCE")}} for s, d in report["diagnostics"].items() if d]
    rows_d = [{"dataset": r["dataset"], "method": r["method"], "rank": r["rank_mode"], "aggregation": r["aggregation"],
               **{k: f"{100 * r[k]:.3f}" for k in METRICS}} for r in query_rows]
    rows_e = [{"scene": s, "variant": v, "method": m, "N": populations["paired"]["objects"],
               "accuracy": f"{populations['paired']['accuracy']:.5f}", "distractor_rate": f"{populations['paired']['distractor_selection_rate']:.5f}"}
              for s, record in report["robustness"].items() if record for v, methods in record["metrics"].items() for m, populations in methods.items()]
    for label, rows, fields in (("A_core", rows_a, ("method", "rank", "aggregation", *METRICS)),
                               ("B_probability", rows_b, ("method", "N", "nll", "brier", "ece15", "accuracy")),
                               ("C_objects", rows_c, ("scene", "identifiable", "CORRECTED", "HARMED", "SOFT_RESCUE", "UNUSED_CORRECT_SOURCE")),
                               ("D_query", rows_d, ("dataset", "method", "rank", "aggregation", *METRICS)),
                               ("E_vocabulary", rows_e, ("scene", "variant", "method", "N", "accuracy", "distractor_rate"))):
        (output / ("table_" + label + ".md")).write_text(_table(rows, fields) + "\n")
    old_fits = read_json(Path(binding["composition_root"]) / "calibration/final_temperatures.json")["fits"]
    new_fits = read_json(Path(binding["output_root"]) / "calibration/new_final.json")
    fit_rows = [{"fit": name, "temperature": row["temperature"], "status": row["status"],
                 "objects": row["objects"], "classes": row["classes"], "bound": row["bound_hit"],
                 "nll_before": row["nll_before"], "nll_after": row["nll_after"]} for name, row in old_fits.items()]
    fit_rows += [{"fit": name, "temperature": row["temperature"], "status": row["status"],
                  "objects": str({n: s["objects"] for n, s in row["sources"].items()}),
                  "classes": str({n: s["classes"] for n, s in row["sources"].items()}), "bound": row["bound_hit"],
                  "nll_before": row["nll_before"], "nll_after": row["nll_after"]} for name, row in new_fits.items() if isinstance(row, dict)]
    with (output / "table_B_probability.md").open("a") as handle:
        handle.write("\nFinal CAL fits (original M2 fits unchanged; fold details in scalar JSONs).\n\n" + _table(fit_rows,
                     ("fit", "temperature", "status", "objects", "classes", "bound", "nll_before", "nll_after")) + "\n")
    cost_rows = []
    for method in dict.fromkeys(row["method"] for row in costs["method_requirements"]):
        selected = [r for r in costs["method_requirements"] if r["method"] == method and r["dataset"] == "Replica"]
        cost_rows.append({"method": method, "completed_scenes": len(selected), **{key: sum(r[key] for r in selected) for key in
            ("logical_requests_total", "logical_crop_inputs_total", "unique_required_operations", "proved_shared_operations_saved")}})
    with (output / "table_D_query.md").open("a") as handle:
        handle.write("\nReplica required cost totals across available scenes; N0 map creation remains included.\n\n" + _table(cost_rows,
                     ("method", "completed_scenes", "logical_requests_total", "logical_crop_inputs_total", "unique_required_operations", "proved_shared_operations_saved")) + "\n")
    with (output / "table_E_vocabulary.md").open("a") as handle:
        handle.write("\nFresh prerequisite status: `" + (report["fresh"]["status"] if report["fresh"] else "PENDING") + "`.\n")
    return report, output


def write_documents(binding, report, output):
    from .claims import claim_ledger

    claims = claim_ledger(report)
    write_once(output / "novelty_decision.json", claims)
    table_f = _table([{"claim": r["claim"], "status": r["status"], "matched_control": r["matched_control"],
                      "decision": r["decision"]} for r in claims["claims"]], ("claim", "status", "matched_control", "decision"))
    (output / "table_F_claims.md").write_text(table_f + "\n")
    costs = read_json(report["cost_ledger"])
    pairs = {r["method"]: r["metrics"] for r in report["pooled_rows"] if r["dataset"] == "Replica" and r["rank_mode"] == "OFFICIAL_CURRENT_CLASS"}
    primary, native = pairs["CP_M2_EQUAL_CAL"], pairs["N0"]
    difference = ", ".join(f"{k} {100 * (primary[k] - native[k]):+.3f} pp" for k in METRICS)
    status = "DRAFT — query evidence incomplete" if report["status"] == "PARTIAL" else "Measured study complete; scientific support is mixed"
    final_directory = "../../../artifacts/static_ovmap/m2_reviewer_study_v1/" + Path(binding["output_root"]).name
    table_links = "\n".join(f"- [Table {name[0]} — {name[2:]}]({final_directory}/table_{name}.md)" for name in
                            ("A_core", "B_probability", "C_objects", "D_query", "E_vocabulary", "F_claims"))
    fresh_status = report["fresh"]["status"] if report["fresh"] else "PENDING"
    results = f"""# M2 reviewer evidence results

{status}. Report identity: `{report['identity']}`. Deployment remains N0.

## Scope and interpretation

Ten fixed scenes: two ScanNet CAL scenes and eight historically exposed Replica
transfer scenes. CAL was already exposed to the learned query checkpoint.
CAL selected A2 (N0+static S2) and static S2 as the single-source comparator before
new Replica evaluations. No Replica fitting or comparator/prompt/seed selection.

The core matrix contains {len(report['scene_rows'])}/260 measured scene/rank rows
and {sum(r['dataset'] == 'Replica' for r in report['pooled_rows'])}/26 Replica pools.
APall equals uAP over the evaluator's recorded0.50–0.90 overlaps; AP25 is separate.
Both frozen N0 ranks and official current-class area ranks are reported. Dataset
pools call the released evaluator over the original ordered files; scene means
are separate. No macro bootstrap is described as a pooled interval.

## Main outcome

Official-current-class Replica pooled M2_CAL minus N0: **{difference}**.
This is a metric tradeoff, not uniform superiority. Equal-weight probability
fusion does not give equal effective score influence: replacing only native
canonical-relative scores by cosine changes the result. Common sharpening/shared
temperature largely reproduces source-specific calibration; its necessity is
not established. See full matched controls and paired scene deltas.

On157 fixed all-three-available identifiable Replica objects, M2_RAW→M2_CAL
NLL improves3.32554→1.74688 and Brier0.94642→0.66166; accuracy48.408%→50.318%.
The constant0.01 control has NLL1.74291 and Brier0.66077. Probability improvement
is conditional on this measured population, not all false positives.

## Object-level failure evidence

office1 frozen-rank AP50 falls0.25→0.19117647. The released trace adds blanket
owner80 but loses blanket owner79 and desk owner8. Both Q/S2 predict cloth for
owner79 and tv-screen for owner8, replacing correct native labels. Desk class AP
falls1→0; blanket class AP stays unchanged. All scenes retain duplicate/ignore
events at every recorded overlap; semantic diagnostics use a separate strict
unique geometry IoU>0.5 correspondence. First two corrected/harmed owners are
selected deterministically, including failures rather than flattering examples.

## Query, compute and robustness

CAL did not trigger B100/B400: GAIN fusion uAP0.0366501/mIoU0.234450 versus
COMBINE0.0324184/0.238655 and random seed mean0.0390670/0.241298.
Three random seeds remain separate; no lucky seed selection or ensemble.
Completed B200 acquisitions: {costs['completed_B200_jobs']}/50.
New recorded visual forwards: {costs['study_physical_query_totals']['model_forwards']};
new crops: {costs['study_physical_query_totals']['crop_inputs']}; measured incremental
inference seconds: {costs['study_physical_query_totals']['inference_seconds']:.3f}.
Shared mapping/frontend, source-attributed logical requests, exact operation
unions, and physical cached execution are separate in the cost ledger.
Historical peak memory and initial ad-hoc fold timing are missing, not zero.
Failed-attempt overhead is incompletely measured. These are offline cached
experiments, not an online end-to-end latency benchmark.

Vocabulary testing reused frozen image aggregates and query trajectories, with
zero new image forwards. On the same157 paired objects M2_CAL accuracy is50.318%
(original),55.414%(photo),46.497%(close-up),50.318%(16 added distractors); the last
condition has0% distractor selection. This tests controlled word sets, not global
distractor absence. Singleton softmax is identically1, so arbitrary single-text
retrieval is not supported by this formula. Prompts remain diagnostics, not a
newly selected method.

Fresh status: `{fresh_status}`. All14 authorized local physical families were
already exposed; no fresh scene was recycled or downloaded. The implemented
success path exists but cannot be claimed as real fresh validation.

## Tables and reproducible evidence

{table_links}

All metric CSVs store fractions, not percentages; Markdown AP/IoU tables display
percent. Raw scene rows, paired macro deltas, probability populations, source
operation identities, text receipts and external file hashes accompany the
tables. Missing prerequisites remain explicit. See the claim ledger for the
supported/mixed/not-supported/not-tested distinctions.
"""
    (output / "M2_REVIEWER_RESULTS.md").write_text(results)
    claim_text = "# M2 claim ledger\n\n" + claims["scientific_recommendation"] + "\n\n" + table_f + "\n"
    for row in claims["claims"]:
        claim_text += f"\n## {row['claim']}\n\nStatus: `{row['status']}`. Control: {row['matched_control']}.\n\n{row['limitation']}\n\nDecision: `{row['decision']}`. Numeric evidence is in novelty_decision.json.\n"
    (output / "M2_CLAIM_LEDGER.md").write_text(claim_text)
    handoff = f"""# M2 reviewer study handoff

Status: {status}. Branch: `research/ovimap-m2-reviewer-evidence-v1`.
Base: `8fee8294c1a3e83feeae28782f6ef4700f08d35c`.
External output root: `{binding['output_root']}`.
Report source: `{output}`. Final remote SHA belongs in the external publication
receipt; no recursive self-reference or unverified push claim is made here.

## Rebuild/resume

Run from the study worktree with the existing native environment:

```bash
for phase in bind core query-controls diagnostics robustness fresh report; do
  /home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_m2_reviewer_study.py --phase "$phase" --resume
done
```

Query leaves use separate processes and per-job/per-GPU locks. Existing completed
sources and outputs are verified rather than rerun. Native and S2 text workers
use their original separate Python environments. No model/data downloads or
environment reinstall are required. Do not concurrently duplicate a live job;
inspect processes and locks before resuming. Publication requires final stage
readiness, bundle integrity, a normal branch push and exact full remote SHA match.

## Boundaries

N0 deployment and historical source artifacts are unchanged. Raw images, model
weights, surfaces and large arrays stay in shared storage. The small release
contains actual tables/scalars/ledgers plus precise external references, not a
claim that the large referenced bytes were uploaded. The original supplied
specification and SHA manifest are preserved in docs/paper/static_ovmap/
m2_reviewer_study_v1/. Fresh data is unavailable; the real success path is
implemented but unexecuted on a genuinely new scene. Missing initial timing and
failed-attempt overhead remain disclosed.
"""
    (output / "M2_REVIEWER_HANDOFF.md").write_text(handoff)
    return claims
