"""Exact pre-selection code/configuration binding before holdout assets open."""

from __future__ import annotations

from pathlib import Path

from .boundary_jobs import file_identity
from .study_execution import read_json, verify_receipt


def confirmation_contract(selection_receipt, runtime, config=None):
    evidence = verify_receipt(selection_receipt)
    selection_path = Path(evidence["selection_path"])
    selection_identity = file_identity(selection_path)
    if selection_identity not in evidence["outputs"]:
        raise ValueError("confirmation selection is outside the frozen receipt outputs")
    selection = read_json(selection_path)
    if (selection["status"] != "FROZEN" or selection["final_candidate"] == "N0"
            or selection.get("confirmation_authorized") is not True):
        raise ValueError("confirmation requires a frozen retained candidate")
    plan = selection["confirmation"]
    if (plan["status"] != "REQUIRED_FROZEN" or not 2 <= len(plan["rows"]) <= 4
            or len(set(plan["rows"])) != len(plan["rows"])
            or not {"N0", selection["final_candidate"]} <= set(plan["rows"])):
        raise ValueError("confirmation methods differ from the at-most-four-row frozen plan")
    frozen_identity = selection["frozen_config"]
    if file_identity(frozen_identity["path"]) != frozen_identity or frozen_identity not in evidence["outputs"]:
        raise ValueError("confirmation configuration changed after final selection")
    frozen = read_json(frozen_identity["path"])
    if frozen["runtime"] != runtime or (config is not None and frozen["study"] != config):
        raise ValueError("confirmation runtime/study configuration differs from the frozen selection")
    if (len(frozen["prediction_code_commit"]) != 40
            or frozen["prediction_code_commit"] != selection["prediction_code_commit"]):
        raise ValueError("confirmation prediction code commit differs")
    for source in frozen["sources"]:
        if file_identity(source["path"]) != source:
            raise ValueError(f"confirmation prediction source changed after freeze: {source['path']}")
    return {"selection": selection, "frozen": frozen, "receipt": file_identity(selection_receipt)}


def confirmation_rows(locked, contract, requested=None):
    if locked != contract["frozen"]["split"]:
        raise ValueError("confirmation split differs from the final selection lock")
    rows = [row for row in locked["selected"] if row["role"] == "confirm"]
    if len(rows) != 2 or len({row["scene_id"] for row in rows}) != 2:
        raise ValueError("exactly two preselected confirmation scenes are required")
    if requested is not None:
        if not set(requested) <= {row["scene_id"] for row in rows}:
            raise ValueError("requested confirmation scene is outside the frozen split")
        rows = [row for row in rows if row["scene_id"] in requested]
    return rows


def require_confirmation_method(scene, method, runtime, config, selection_receipt):
    contract = confirmation_contract(selection_receipt, runtime, config)
    confirmation_rows(contract["frozen"]["split"], contract, [scene])
    if method not in contract["selection"]["confirmation"]["rows"]:
        raise ValueError("confirmation method was not preselected before holdout access")
    return contract
