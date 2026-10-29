"""Initialize the author's unused auxiliary AR branch; preserve every AP AST."""
import ast
import copy
import hashlib
from pathlib import Path

from .common import digest_file, source_identity, verified, write


def adapt(namespace, source):
    cached = namespace.get('_OVMAP_AUXILIARY_RECALL_COMPAT')
    if cached:
        return cached
    source = Path(source)
    original = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_matches')
    if namespace['evaluate_matches'].__code__.co_filename != str(source):
        raise ValueError('Auxiliary recall repair requires the actual original evaluator function')
    patched = copy.deepcopy(original)
    branches = [n for n in ast.walk(patched) if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name) and n.test.id == 'has_gt'
                and [ast.unparse(s) for s in n.orelse] ==
                ["ap_current = float('nan')", "rc_current = float('nan')"]]
    if len(branches) != 1:
        raise ValueError('Pinned absent-GT auxiliary recall boundary differs')
    branch = branches[0]
    assignment = ast.parse("ar_current = float('nan')").body[0]
    ast.copy_location(assignment, branch.orelse[0])
    branch.orelse.insert(1, assignment)
    # Removing this one auxiliary assignment must restore the entire original
    # function, including AP, matches, confidence, thresholds and return values.
    branch.orelse.pop(1)
    if ast.dump(patched) != ast.dump(original):
        raise ValueError('Auxiliary repair changed the original AP function')
    branch.orelse.insert(1, assignment)
    ast.fix_missing_locations(patched)
    exec(compile(ast.Module(body=[patched], type_ignores=[]), str(source), 'exec'), namespace)
    evidence = dict(status='DONE', rule='ABSENT_GT_AUXILIARY_AR_NAN_ONLY', AP_AST_unchanged=True,
        original_source=dict(path=str(source), sha256=digest_file(source)),
        original_function_AST_sha256=hashlib.sha256(ast.dump(original).encode()).hexdigest(),
        adapted_function_AST_sha256=hashlib.sha256(ast.dump(patched).encode()).hexdigest(),
        inserted_statement="ar_current = float('nan')", patched_source_file=False,
        producer=source_identity(('score_compat',)))
    namespace['_OVMAP_AUXILIARY_RECALL_COMPAT'] = evidence
    return evidence


def install():
    """Apply only in a new scoring process; no active producer files are edited."""
    from . import standard_score
    original = standard_score.evaluator
    if getattr(original, '_ovmap_auxiliary_recall_compat', False):
        return

    def evaluator(binding, benchmark):
        namespace = original(binding, benchmark)
        root = Path(binding['output_root'])
        protocol = verified(root / 'protocol_lock.json')
        evidence = adapt(namespace, protocol['official_sources']['evaluator_scan']['path'])
        write(root / 'review/G3_evaluator_auxiliary_compat.json', dict(evidence,
            protocol=protocol['identity'], new_GPU_calls=0, new_training_updates=0))
        return namespace

    evaluator._ovmap_auxiliary_recall_compat = True
    standard_score.evaluator = evaluator
