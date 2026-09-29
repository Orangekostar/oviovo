"""Render the four required evidence tables and three wave-1 handoff documents."""

import json
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex

from .audit import METRICS
from .report_data import collect

ROOT = Path(__file__).resolve().parents[3]


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]]) + "\n"


def numbers(values, factor=1):
    return [f"{factor * values[k]:.6f}" for k in METRICS]


def results_document(data, interpretation):
    parts = ["# A7 wave-1 measured results\n",
             ("This is development/regression evidence on two previously exposed ScanNet CAL scenes and eight historical Replica scenes. "
             "It is not fresh confirmation. All methods retain the same native geometry, owner registry, projection and complete map. "
             "E02 changes recognition masks only; it does not improve reconstructed shape.\n"),
             "## A. Performance and matched comparisons\n",
             ("Main results are released dataset pools with OFFICIAL_CURRENT_CLASS ranking. Values are percentages; deltas are percentage points. "
             "APall aliases released uAP over actual IoU thresholds 0.50–0.90, with AP25 separate; it is not COCO AP through 0.95.\n")]
    for split, comparison in data["comparisons"].items():
        official = comparison["ranks"]["OFFICIAL_CURRENT_CLASS"]
        parts += [f"### {split.upper()} official released pool\n",
                  table(["Method", "APall", "AP50", "AP25", "mIoU", "mAcc", "ΔAPall vs N0", "ΔAPall vs A7"],
                        [[m, *numbers(v["released_pool_fraction"], 100), f"{v['pooled_delta_N0_pp']['apall']:+.6f}",
                          f"{v['pooled_delta_A7_pp']['apall']:+.6f}"] for m, v in official["methods"].items()]
                        + [[m, "BLOCKED", "—", "—", "—", "—", "—", "—"] for m in comparison["blocked_methods_without_numeric_results"]]),
                  table(["Matched contrast (new − old)", *METRICS], [[name, *numbers(v["pooled_delta_pp"])] for name, v in official["matched_contrasts"].items()])]
        if official["interaction"]:
            parts += ["Interaction is Q+region − Q − region + base A7, using the single CAL-frozen pair.\n",
                      table(["Interaction", *METRICS], [[split, *numbers(official["interaction"]["pooled_interaction_pp"])]])]
        auxiliary = comparison["ranks"]["FROZEN_N0"]["methods"]
        parts += [f"### {split.upper()} auxiliary fixed-rank pools and official scene means\n",
                  table(["Method", "Frozen APall", "Frozen AP50", "Frozen AP25", "Mean APall", "Mean mIoU", "Worst scene ΔAPall vs A7"],
                        [[m, *[f"{100 * auxiliary[m]['released_pool_fraction'][k]:.6f}" for k in ("apall", "ap50", "ap25")],
                          f"{100*v['scene_mean_fraction']['apall']:.6f}", f"{100*v['scene_mean_fraction']['miou']:.6f}",
                          f"{v['worst_scene_delta_A7_pp']['apall']:+.6f}"] for m, v in official["methods"].items()])]
    parts += [("Every per-scene metric, both rank modes, five-metric scene means and deltas remain in comparisons/{cal,replica}.json and the original rows. "
              "Scene means are supplementary and never substituted for the released pool.\n"), "## B. Effective intervention\n"]
    for split, values in data["split_tables"].items():
        parts += [f"### {split.upper()} sources\n", table(
            ["Source", "All owners", "Capped targets", "Requested views", "Usable views", "Nonempty final support", "Genuine owners", "Changed suggestions"],
            [[m, v["owners"], v["static_capped_targets"] if v["static_capped_targets"] is not None else "original Q retention",
              v["requested_views"], v["usable_views"], v["nonempty_final_support_views"] if v["nonempty_final_support_views"] is not None else "cached Q",
              v["available"], v["changed_suggestions_common_available"]] for m, v in values["sources"].items()]),
            table(["Source", "Failed view reasons", "Unavailable owner reasons"],
                  [[m, json.dumps(v["failed_view_reasons"], sort_keys=True), json.dumps(v["fallback_owner_reasons"], sort_keys=True)] for m, v in values["sources"].items()])]
    parts += [("Unavailable sources never vote in A7; DIRECT falls back to N0. Excluded owners remain in every complete-map evaluation. "
              "SHORTLIST is a decision-identical control, not a discovered improvement.\n"), "## C. Corrections and damage\n",
              ("Semantic corrections use the original unique geometry correspondence with IoU > 0.5 and are separate from the released AP matcher. "
              "Correct-class ranks and source correctness are evaluation-only diagnostics.\n")]
    for split, values in data["split_tables"].items():
        parts += [f"### {split.upper()} owner outcomes\n", table(
            ["Method", "Changed vs A7", "Corrected A7", "Harmed A7", "Corrected N0", "Harmed N0", "Unused correct evidence"],
            [[m, *[v[k] for k in ("changed_from_A7", "corrected_A7", "harmed_A7", "corrected_N0", "harmed_N0", "unused_correct_evidence")]] for m, v in values["corrections_and_damage"].items()]),
            table(["Source", "Common identifiable", "Old correct", "New correct"],
                  [[m, v["common_identifiable"], v["old_correct_common"], v["new_correct_common"]] for m, v in values["sources"].items()])]
        scenes = data["comparisons"][split]["scene_order"]
        trace_rows = []
        for method in values["corrections_and_damage"]:
            for threshold in (.25, .5):
                events = [event for s in scenes for event in data["diagnostics"][s]["released_trace_summary"].get(method + "/OFFICIAL_CURRENT_CLASS", {}).get("AP25_AP50", []) if event["overlap_threshold"] == threshold]
                if events:
                    trace_rows.append([method, threshold, sum(v["added_gt_matches"] for v in events), sum(v["lost_gt_matches"] for v in events)])
        parts += ["Actual released matcher changes relative to N0 (counts summed across scenes; these are not AP deltas):\n",
                  table(["Method", "IoU", "Added GT matches", "Lost GT matches"], trace_rows)]
    parts += ["Full released attribution retains duplicate/ignore events and per-class AP changes; the compact geometry table must not be interpreted as AP true positives.\n",
              table(["Scene", "Example method", "Corrected owner IDs", "Harmed owner IDs"],
                    [[s, d["examples"]["method"], d["examples"]["corrected_A7"], d["examples"]["harmed_A7"]] for s, d in data["diagnostics"].items()]),
              "Examples use owner-ID order, at most two corrected and two harmed owners per scene; none were selected for attractive imagery.\n",
              "## D. Costs and decisions\n"]
    for split, values in data["split_tables"].items():
        parts += [f"### {split.upper()} recorded worker work\n", table(
            ["Worker", "Crop inputs", "Dense/SAM images", "Region pools/box decodes", "Reused raw crops", "Text inputs", "Load s", "Worker wall s", "Peak allocated GiB"],
            [[m, v.get("physical_crop_inputs_this_invocation", "—"), v.get("physical_image_encodings_this_invocation", v.get("physical_image_encodings", "—")),
              v.get("physical_region_poolings_this_invocation", v.get("physical_box_decodes", "—")), v.get("reused_raw_crops_this_invocation", "—"),
              v.get("physical_text_inputs_this_invocation", "—"), f"{v.get('model_load_seconds', 0):.3f}" if "model_load_seconds" in v else "unrecorded",
              f"{v.get('wall_seconds', v.get('elapsed_seconds', 0)):.3f}", f"{v['peak_gpu_allocated_bytes']/1024**3:.3f}" if "peak_gpu_allocated_bytes" in v else "—"] for m, v in values["worker_costs"].items()])]
        scenes = data["comparisons"][split]["scene_order"]
        kinds = ("image_crop", "dense_image", "region_pool_projection", "sam2_image", "sam2_box_decode")
        required = [[method, *[sum(data["scene_costs"][scene]["logical_required_visual_operations"][method]["counts_by_kind"].get(kind, 0)
                                  for scene in scenes) for kind in kinds]] for method in values["corrections_and_damage"]]
        parts += ["Required visual operation unions, summed over scenes (heterogeneous units, not FLOPs or latency):\n",
                  table(["Method", *kinds], required)]
    parts += [table(["Downloaded asset", "Verified GiB", "Recorded download seconds"],
                    [[name, f"{value['verified_bytes']/1024**3:.6f}",
                      value["receipt"].get("download_seconds", value["receipt"].get("download_time_seconds"))
                      if value["receipt"].get("download_seconds", value["receipt"].get("download_time_seconds")) is not None else "not recorded"]
                     for name, value in data["downloads"].items()]),
              (f"Changed-source scalar fitting took {sum(data['source_fit_seconds'].values()):.6f} recorded seconds in total. "
              f"There are {len(data['distinct_evaluator_receipt_seconds'])} distinct reused/new evaluator receipts with "
              f"{sum(data['distinct_evaluator_receipt_seconds'].values()):.3f} recorded seconds; this includes historical work and is not new-run latency.\n")]
    parts += [(f"Verified new model/tokenizer/config bytes: {data['new_model_verified_bytes']:,} ({data['new_model_verified_bytes']/1024**3:.3f} GiB), within 25 GiB. "
              "Model download receipts, real adapter smoke costs, scalar-fit times and distinct evaluation receipts are included in report/data.json.\n"),
              ("Required operations are model/input-identified unions in costs/<scene>.json, including original N0 mapping inputs. "
              "Physical cache reuse is reported separately and is not free logical work. Worker wall time includes load/I/O; parallel worker sums are not end-to-end latency. "
              "Per-worker peaks are maxima, not summed. Historical frontend time, unrecorded failed/interrupted attempts and SAM2 download time are incomplete. "
              "Recorded evaluation receipts include historical cache reuse; they do not measure this invocation's new work.\n"),
              *[paragraph + "\n" for paragraph in interpretation["findings"]],
              "Next-wave recommendation (not executed): " + interpretation["next_capability"] + ". " + interpretation["next_rationale"] + "\n"]
    return "\n".join(parts)


def report(binding):
    root = Path(binding["output_root"])
    inputs = InputIndex()
    inputs.identity(__file__)
    inputs.identity(root / "review/interpretation.json")
    interpretation = read_json(root / "review/interpretation.json")
    data_path = root / "report/data.json"
    data = read_json(data_path) if data_path.exists() else collect(binding)
    for item in data["inputs"]:
        inputs.identity(item["path"], item)
    inputs.identity(data_path)
    nomination, pair = read_json(root / "nomination.json"), read_json(root / "composition.json")
    selection = ("# A7 wave-1 selection\n\n"
        f"CAL-frozen research nomination: **{nomination['nomination']}**. Deployment: **{nomination['deployment']}**.\n\n"
        "Selection uses CAL released pooled APall, then mIoU, then AP50 (tolerance 1e-10), then fewer unique required visual operations, then fixed registry order. "
        "Eligible candidates are the measured _A7 variants, E04_MIX50 and the optional combination, with N0 and A7_REFIT as incumbents. "
        "DIRECT rows and standalone Q/S2 are diagnostic controls, excluded by the prespecified nomination rule even when their CAL APall is higher. "
        "Replica was not used to nominate, prune prescribed methods, tune temperatures or choose the combination.\n\n"
        f"The one frozen combination uses {pair['q_variant']} and {pair['region_variant']}; each retains its own CAL-fitted scalar. "
        "No new image inference or scalar search is introduced for composition.\n\n"
        f"Nomination identity: `{nomination['identity']}`. Transfer lock: `{read_json(root / 'transfer_lock.json')['identity']}`.\n\n"
        + table(["CAL ranking", "APall (%)", "mIoU (%)", "AP50 (%)", "Required operations"],
                [[r["method"], f"{100*r['metrics']['apall']:.9f}", f"{100*r['metrics']['miou']:.9f}", f"{100*r['metrics']['ap50']:.9f}", r["unique_required_operations"]] for r in nomination["rankings"]])
        + "\nBoth datasets are previously exposed development/regression evidence. N0 remains the deployment baseline regardless of favorable Replica columns.\n")
    handoff = ["# A7 wave-1 handoff\n", f"External run: `{root}`. Binding: `{binding['identity']}`.\n",
               ("Measured: E01 RAW_EQ/UNIT_EQ/GMED, E02 GLOBAL/SAM2, C0 SO400M, E03 FC_FROZEN_REGION/OVR, E04 SHORTLIST, five controls and one frozen composition. "
               "SAM3 SPATIAL/MIX50 are blocked by actual official gated access (401); no substitute weights, zero metrics or fabricated predictions are supplied.\n"),
               ("FC_FROZEN_REGION is an original frozen OpenCLIP backbone with matched region operators, not a full FC-CLIP benchmark reproduction. "
               "OVR loads the learned author checkpoint backbone strictly; pinned upstream operators are extracted, but the full Detectron wrapper was not executed.\n"),
               "## Assets and provenance\n"]
    for name, value in data["downloads"].items():
        receipt = value["receipt"]
        handoff += [(f"- {name}: revision/code `{receipt.get('revision', receipt.get('code_commit'))}`, verified bytes {value['verified_bytes']}; "
                    f"full checkpoint SHA256 and official source in `assets/{name}/download_receipt.json`.\n")]
    parameter_path = root / "environments/region_parameters.json"
    inputs.identity(parameter_path)
    region_parameters = read_json(parameter_path)
    handoff += [(f"\nEach FC/OVR recognition model has {region_parameters['parameters_per_branch']:,} parameters "
                 f"({region_parameters['visual_parameters_per_branch']:,} visual). Both branches were strictly loaded with 543 keys; "
                 "these counts exclude unused segmentation heads and all shared native-map components. "
                 "SO400M has 1,136,008,498 parameters and SAM2 has 224,446,642, recorded in real worker receipts. "
                 "All visual models are frozen; CAL only fits scalar temperatures.\n")]
    handoff += [("\nSO400M: Apache-2.0, documented WebLI training. Frozen ConvNeXt OpenCLIP: MIT model card, LAION-2B English pretraining. "
                "Native SigLIP and S2 use documented WebLI training. SAM2 code/checkpoints are Apache-2.0 and document SA-V; an exhaustive training-overlap audit is unavailable. "
                "OVR code is Apache-2.0; its inherited configuration documents COCO panoptic training and the backbone originates from LAION. "
                "The complete author-checkpoint training history and overlap with ScanNet/Replica are unresolved, not assumed clean. "
                "Pinned model cards and code licenses provide the supporting source records.\n"),
                "## Environment and restore\n",
                ("Actual Python executables, Torch/CUDA versions and package snapshots are in environments/{semantic,region,sam2}.json and workers.json. "
                "No installed environment was upgraded. Restore the exact model files and historical binding inputs on shared storage before resuming. "
                "External paths and SHA manifests describe bytes not uploaded to GitHub. The source map and original principal JSON are unchanged.\n"),
                ("```bash\n/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_a7_evidence_upgrade.py --phase all --split all --resume --gpus 0,1,2\n"
                "/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_a7_evidence_upgrade.py --phase report --resume\n"
                "/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_a7_evidence_upgrade.py --phase publish --resume\n```\n"),
                ("These commands run from the research worktree. Worker receipts validate fixed model/input/code identities before reuse; a changed scientific input requires a separate attempt. "
                "Full arrays, weights and RGB-D remain external; compact metrics, scalars, decisions, contracts and diagnostics are published.\n"),
                "## Limits\n",
                ("See RESULTS for complete official/frozen rankings, pooling versus scene means, intervention/fallbacks, matched controls, composition interaction and timing omissions. "
                "SAM3 remains an asset-access block. Wave1 does not execute E05–E12 or geometry reconstruction changes. "
                "Publication is verified only by an external receipt matching full local HEAD and the remote branch SHA.\n")]
    documents = {"A7_WAVE1_RESULTS.md": results_document(data, interpretation), "A7_WAVE1_SELECTION.md": selection,
                 "A7_WAVE1_HANDOFF.md": "\n".join(handoff)}
    index = InputIndex()
    for name, body in documents.items():
        path = ROOT / "docs/paper/static_ovmap" / name
        path.write_text(body)
        index.identity(path)
    result = {"status": "REPORTS_RENDERED_FROM_MEASURED_DATA", "data_identity": data["identity"], "documents": index.entries(),
              "inputs": inputs.entries()}
    write_once(root / "report/receipt.json", result)
    return result
