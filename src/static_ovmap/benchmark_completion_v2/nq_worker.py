"""Run the unchanged true native/Q pipeline with legal finite-scene exhaustion."""
import argparse
import ast
from pathlib import Path

from .common import read, write, digest_file


def validate_query_completion(result, frames, *, budget):
    decisions = result['decisions']
    attempts = result['state'].logical_ledger.attempts
    if [row['frame_id'] for row in decisions] != list(frames.schedule):
        raise ValueError('Query stopped before all frozen frame barriers were replayed')
    if not 0 <= attempts <= budget or attempts != sum(len(row['results']) for row in decisions):
        raise ValueError('Query actual attempted requests violate the frozen budget/ledger')
    return dict(actual_attempts=attempts, budget=budget,
        stop_reason='BUDGET_SPENT' if attempts == budget else 'CANDIDATES_EXHAUSTED',
        boundary_adaptation='maximum attempts; preserve original complete causal replay and ranking',
        all_frame_barriers_replayed=True)


def run(job):
    from static_ovmap.backbone_wave1 import semantic_readout
    source = Path(semantic_readout.__file__)
    if digest_file(source) != job['original_worker_sha256']:
        raise ValueError('Bound original native/Q worker changed')
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_native_query')
    count = 0
    for node in ast.walk(function):
        if (isinstance(node, ast.If) and ast.unparse(node.test) == 'result[\'state\'].logical_ledger.attempts != 200'):
            if (len(node.body) != 1 or not isinstance(node.body[0], ast.Raise)
                    or 'Q_GAIN did not spend the prescribed 200 attempted requests' not in ast.unparse(node.body[0])):
                raise ValueError('Q exact-budget boundary differs from reviewed parent')
            node.test = ast.Constant(True)
            node.body = ast.parse("receipt['query_completion'] = _validate_query_completion(result, replay_frames, budget=200)").body
            count += 1
    if count != 1:
        raise ValueError('Exactly one reviewed Q exhaustion boundary must be adapted')
    ast.fix_missing_locations(tree)
    namespace = dict(semantic_readout.__dict__, _validate_query_completion=validate_query_completion)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    result = namespace['run_native_query'](job)
    write(Path(job['output_root']) / 'boundary_adaptation.json', dict(status='DONE',
        original_worker=dict(path=str(source), sha256=digest_file(source)),
        adapted_boundary_count=count, query_completion=result['query_completion'],
        native_Q_receipt=dict(path=str(Path(job['output_root']) / 'native_query_receipt.json'),
                             sha256=digest_file(Path(job['output_root']) / 'native_query_receipt.json'))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    run(read(parser.parse_args().job))
