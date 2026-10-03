"""Causal current-owner interpretation of retired native segment records."""

import argparse
import ast
import copy
import inspect
from pathlib import Path

from static_ovmap.backbone_wave1 import semantic_readout
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.query_study import CapturedFrames as OriginalFrames, replay_captured
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
from static_ovmap.recovery_wave2.native_worker import ReadOnlyNativeCache, validate_query_budget

from . import base_sources


def normalize_alias_snapshot(snapshot, aliases):
    if snapshot.get("label_instances_scope") != "all_known_labels":
        raise ValueError("BLOCKED_CAUSAL_LINEAGE: complete current membership is missing")
    aliases = dict(aliases)
    for pair in snapshot.get("aliases", []):
        old, new = int(pair["old_label"]), int(pair["resolved_label"])
        if min(old, new) < 0:
            raise ValueError("BLOCKED_CAUSAL_LINEAGE: invalid segment alias")
        if old != new:
            aliases[old] = new
    rows = snapshot["label_instances"]
    membership = {int(row["segment_label"]): int(row["instance_label"]) for row in rows}
    if len(membership) != len(rows) or any(min(label, owner) < 0 for label, owner in membership.items()):
        raise ValueError("BLOCKED_CAUSAL_LINEAGE: invalid complete current membership")

    def resolve(label):
        visited = set()
        while label in aliases:
            if label in visited:
                raise ValueError("BLOCKED_CAUSAL_LINEAGE: cyclic native segment aliases")
            visited.add(label)
            label = aliases[label]
        if label not in membership:
            raise ValueError("BLOCKED_CAUSAL_LINEAGE: resolved current segment membership is missing")
        return label

    for label in aliases:
        resolve(label)
    # Native swapLabels retires old geometry while its statistics may remain.
    normalized = {**snapshot, "label_instances": [
        {**row, "instance_label": membership[resolve(int(row["segment_label"]))]} for row in rows]}
    return normalized, aliases


class CapturedFrames(OriginalFrames):
    def __init__(self, capture_path):
        super().__init__(capture_path)
        self._compact_aliases = {}

    def load(self, index):
        current = super().load(index)
        if current is not None:
            current["snapshot"], self._compact_aliases = normalize_alias_snapshot(
                current["snapshot"], self._compact_aliases)
        return current


def adapt_native_function(cache_factory, budget_validator):
    original = ast.parse(inspect.getsource(semantic_readout.run_native_query))
    adapted = copy.deepcopy(original)
    guard = ast.parse('if result["state"].logical_ledger.attempts != 200:\n'
        '    raise ValueError("Q_GAIN did not spend the prescribed 200 attempted requests")').body[0]
    matches = {"budget": 0, "frames": 0}

    class Bridge(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.dump(node, include_attributes=False) == ast.dump(guard, include_attributes=False):
                matches["budget"] += 1
                return ast.copy_location(ast.Expr(ast.Call(ast.Name("_recovery_validate_query_budget", ast.Load()),
                    [ast.Name("result", ast.Load()), ast.Name("replay_frames", ast.Load())], [])), node)
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module == "static_ovmap.module_validation.query_study":
                if [(item.name, item.asname) for item in node.names] != [("CapturedFrames", None), ("replay_captured", None)]:
                    raise ValueError("original native/Q capture import changed")
                matches["frames"] += 1
                return ast.copy_location(ast.ImportFrom("static_ovmap.cvpr_compact.query_bridge", node.names, 0), node)
            return node

    adapted = Bridge().visit(adapted)
    if matches != {"budget": 1, "frames": 1}:
        raise ValueError("exact original native/Q bridge cannot be applied")
    ast.fix_missing_locations(adapted)
    namespace = dict(vars(semantic_readout), TensorEncoderCache=cache_factory,
        _recovery_validate_query_budget=budget_validator)
    exec(compile(adapted, semantic_readout.__file__, "exec"), namespace)
    return namespace["run_native_query"], {"original_AST": canonical_digest(ast.dump(original)),
        "effective_AST": canonical_digest(ast.dump(adapted)), "changed_AST_nodes": 2,
        "change": "original exhaustion guard; captured current membership resolves retired segment aliases",
        "encoder_cache_interface": "same raw FP32 batch API, task aliases to read-only parent arrays",
        "causal_replay": "unchanged original replay_captured; original CurrentLineage positive-ancestry guard retained"}


_adapt_native_function = adapt_native_function


def _base_function(name):
    original = ast.parse(inspect.getsource(getattr(base_sources, name)))
    adapted = copy.deepcopy(original)
    matches = 0

    class Bridge(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            nonlocal matches
            if name == "run_native_query" and node.module == "static_ovmap.recovery_wave2.native_worker":
                if [(item.name, item.asname) for item in node.names] != [
                        ("ReadOnlyNativeCache", None), ("_adapt_native_function", None), ("validate_query_budget", None)]:
                    raise ValueError("original compact N/Q adapter import changed")
                matches += 1
                return ast.copy_location(ast.ImportFrom("static_ovmap.cvpr_compact.query_bridge", node.names, 0), node)
            return node

        def visit_Constant(self, node):
            nonlocal matches
            if name == "_run_worker" and node.value == "static_ovmap.cvpr_compact.base_sources":
                matches += 1
                return ast.copy_location(ast.Constant("static_ovmap.cvpr_compact.query_bridge"), node)
            if name == "_run_worker" and node.value == "run.log":
                matches += 1
                return ast.copy_location(ast.Constant("run.alias_bridge.log"), node)
            return node

    adapted = Bridge().visit(adapted)
    if matches != {"run_base_sources": 0, "run_native_query": 1, "_run_worker": 2}[name]:
        raise ValueError("original compact base worker differs from the exact bridge")
    ast.fix_missing_locations(adapted)
    namespace = dict(vars(base_sources), __file__=__file__, _run_worker=_run_worker)
    exec(compile(adapted, base_sources.__file__, "exec"), namespace)
    return namespace[name], {"original_AST": canonical_digest(ast.dump(original)),
        "effective_AST": canonical_digest(ast.dump(adapted)), "changed_AST_nodes": matches}


def _run_worker(binding, python, job, kind, index):
    if kind == "existing-fc":
        return base_sources._run_worker(binding, python, job, kind, index)
    return _base_function("_run_worker")[0](binding, python, job, kind, index)


def run_native_query(job):
    worker, bridge = _base_function("run_native_query")
    result = worker(job)
    index = ConsumptionIndex(Path(job["output_root"]) / "input_verifications.json")
    result.update(compact_base_bridge=bridge, inherited_base_producer=index.identity(base_sources.__file__))
    result["identity"] = canonical_digest({key: value for key, value in result.items() if key != "identity"})
    atomic_write_json(Path(job["output_root"]) / "native_query_receipt.json", result)
    return result


def run_base_sources(binding, scene):
    return _base_function("run_base_sources")[0](binding, scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, choices=["native-query"])
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    result = run_native_query(read(args.job))
    print(result["status"], result["physical_image_encodings"], flush=True)
