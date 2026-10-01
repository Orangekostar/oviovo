"""Fresh native/Q lineage with unchanged encoder operators and read-only parents."""

import argparse
import ast
import copy
import inspect
from pathlib import Path
import time

import numpy as np

from static_ovmap.backbone_wave1 import semantic_readout
from static_ovmap.backbone_wave1.features import TensorEncoderCache
from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest

from .binding import ConsumptionIndex, PathResolver, read


class ReadOnlyNativeCache(TensorEncoderCache):
    def __init__(self, model, model_identity, root, parent_roots, *, resolver=None, memo=None):
        super().__init__(model, model_identity, root)
        self.parents = [Path(path).resolve() for path in parent_roots]
        if any(self.root.resolve() == path or self.root.resolve().is_relative_to(path) for path in self.parents):
            self.close()
            raise ValueError("native cache writes cannot target a parent cache")
        self.resolver, self.index = resolver or PathResolver(), ConsumptionIndex(memo)
        self.parent_hits = {}

    def encode(self, **kwargs):
        import torch

        inputs = {}
        for key, value in kwargs.items():
            if hasattr(value, "detach"):
                if value.dtype != torch.float32:
                    raise ValueError("frozen native encoder input must remain FP32")
                inputs[key] = _array_digest(value.detach().cpu().numpy())
            else:
                inputs[key] = value
        key = canonical_digest({"model": self.model_identity, "inputs": inputs})
        destination = self.root / self.model_identity / (key + ".json")
        with exclusive_lock(destination.with_suffix(".lock")):
            candidates = [destination, *(parent / self.model_identity / (key + ".json") for parent in self.parents)]
            for path in candidates:
                if not path.is_file():
                    continue
                self.index.identity(path)
                receipt = self.resolver.rewrite(read(path))
                if (receipt["status"] != "COMPLETE" or receipt["model_identity"] != self.model_identity
                        or receipt["content_identity"] != key or receipt["tensor_inputs"] != inputs
                        or receipt["crop_inputs"] != len(kwargs["pixel_values"])):
                    raise ValueError("native content cache receipt differs from the exact physical input")
                self.index.identity(receipt["arrays"]["path"], receipt["arrays"])
                with np.load(receipt["arrays"]["path"], allow_pickle=False) as arrays:
                    vectors = arrays["raw_vectors"]
                if (vectors.dtype != np.float32 or vectors.ndim != 2
                        or len(vectors) != len(kwargs["pixel_values"]) or not np.isfinite(vectors).all()):
                    raise ValueError("native raw-vector cache has malformed precision or shape")
                if path != destination:
                    receipt.update(read_only_parent_alias=self.index.identity(path), parent_cache_written=False)
                    atomic_write_json(destination, receipt)
                    self.parent_hits[key] = str(path)
                break
        return super().encode(**kwargs)


def validate_query_budget(result, frames):
    rows, attempts = result["decisions"], result["state"].logical_ledger.attempts
    if [row["frame_id"] for row in rows] != list(frames.schedule):
        raise ValueError("query accounting omitted a scheduled frame")
    universe = {request["request_id"] for frame in frames.frames.values() for request in frame["requests"]}
    paid = []
    for row in rows:
        current = {request["request_id"] for request in frames.frames.get(row["frame_id"], {}).get("requests", [])}
        ids = [request["request_id"] for request in row["results"]]
        if not set(ids) <= current:
            raise ValueError("query accounting includes a request outside its actual current-frame universe")
        paid.extend(ids)
    if attempts != len(paid) or len(set(paid)) != attempts or attempts > 200:
        raise ValueError("query ledger, unique paid requests, and prescribed budget disagree")
    if attempts != 200 and (len(universe) >= 200 or set(paid) != universe):
        raise ValueError("fewer than 200 query attempts require true technical universe exhaustion")
    return {"budget": 200, "attempts": attempts, "technical_universe": len(universe),
        "reason": "PRESCRIBED_BUDGET_SPENT" if attempts == 200 else "TRUE_TECHNICAL_UNIVERSE_EXHAUSTION",
        "unpaid_request_ids": sorted(universe - set(paid)), "ordered_paid_request_ids": paid,
        "technical_universe_identity": canonical_digest(sorted(universe)),
        "exact_attempt_ledger": True, "additional_query_acquisitions": 0}


def _adapt_native_function(cache_factory, budget_validator):
    source = inspect.getsource(semantic_readout.run_native_query)
    original = ast.parse(source)
    adapted = copy.deepcopy(original)
    expected = ast.parse('if result["state"].logical_ledger.attempts != 200:\n'
                         '    raise ValueError("Q_GAIN did not spend the prescribed 200 attempted requests")').body[0]
    matches = 0

    class ReplaceBudgetGuard(ast.NodeTransformer):
        def visit_If(self, node):
            nonlocal matches
            if ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False):
                matches += 1
                return ast.copy_location(ast.Expr(ast.Call(ast.Name("_recovery_validate_query_budget", ast.Load()),
                    [ast.Name("result", ast.Load()), ast.Name("replay_frames", ast.Load())], [])), node)
            return self.generic_visit(node)

    adapted = ReplaceBudgetGuard().visit(adapted)
    if matches != 1:
        raise ValueError("original native/Q worker budget guard changed; exact adapter cannot be applied")
    ast.fix_missing_locations(adapted)
    namespace = dict(vars(semantic_readout), TensorEncoderCache=cache_factory,
                     _recovery_validate_query_budget=budget_validator)
    exec(compile(adapted, semantic_readout.__file__, "exec"), namespace)
    return namespace["run_native_query"], {"original_AST": canonical_digest(ast.dump(original)),
        "effective_AST": canonical_digest(ast.dump(adapted)), "changed_AST_nodes": 1,
        "change": "exact 200-attempt guard replaced by exact universe-exhaustion validator",
        "encoder_cache_interface": "same raw FP32 batch API, task aliases to read-only parent arrays"}


def run_native(job):
    root = Path(job["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    index = ConsumptionIndex(root / "input_verifications.json")
    mapping = read(job["map_receipt"])
    if (mapping["status"] != "COMPLETE" or not mapping["geometry_locked_before_semantics_and_labels"]
            or mapping["GT_input"] or mapping["map_id"] != job["map_id"]):
        raise ValueError("fresh native/Q readout requires the complete own-map lock")
    for path in (job["map_receipt"], job["capture_manifest"], job["deferred_metadata"],
                 __file__, semantic_readout.__file__, Path(semantic_readout.__file__).with_name("features.py")):
        index.identity(path)
    for row in job["model"]["files"] + [job["model"]["text"], job["checkpoint"]]:
        index.identity(row["path"], row)
    identity = canonical_digest({"job": job, "inputs": index.entries()})
    path = root / "native_query_receipt.json"
    if path.is_file():
        old = read(path)
        if old["status"] == "COMPLETE":
            if old.get("recovery_input_identity") != identity:
                raise ValueError("completed own-map native/Q worker cannot resume changed inputs")
            for row in old["recovery_inputs"] + [old["native_features"], old["query_scores"], old["query_decisions"]]:
                index.identity(row["path"], row)
            return old
        path.rename(root / ("native_query_receipt.failed_" + str(time.time_ns()) + ".json"))
    encoders, accounting = [], []

    def factory(model, model_identity, cache_root):
        encoder = ReadOnlyNativeCache(model, model_identity, cache_root, job["parent_encoder_cache_roots"],
            resolver=PathResolver(job.get("path_map")), memo=root / "input_verifications.json")
        encoders.append(encoder)
        return encoder

    def validator(result, frames):
        accounting.append(validate_query_budget(result, frames))

    run, adapter = _adapt_native_function(factory, validator)
    receipt = run(job)
    if len(accounting) != 1 or len(encoders) != 1:
        raise RuntimeError("own-map native/Q evidence lacks its exact budget or encoder accounting")
    for row in encoders[0].index.entries():
        index.identity(row["path"], row)
    receipt.update(recovery_input_identity=identity, recovery_inputs=index.entries(), effective_adapter=adapter,
        effective_worker=index.identity(__file__), query_budget_accounting=accounting[0],
        parent_physical_cache_hits=encoders[0].parent_hits, parent_caches_read_only=True,
        physical_forward_attempts=encoders[0].forward_attempts, new_text_forwards=0)
    atomic_write_json(path, receipt)
    index.write_memo(root / "input_verifications.json")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    result = run_native(read(args.job))
    print(result["map_id"], result["status"], "native/Q physical image inputs", result["physical_image_encodings"], flush=True)
