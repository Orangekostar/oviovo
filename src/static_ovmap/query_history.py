"""Capture native query history without changing native retention or inference."""
import ast
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np


def align_history_queries(native, full, receipt):
    """Namespace full-pool IDs and prove the original cache's observation bridge."""
    if receipt.get('status') != 'PASS' or receipt.get('history_scope') != 'complete_executed_native_queries':
        raise ValueError('complete native query history receipt required')
    mapping = receipt['native_to_full_indices']
    if set(native) != set(full) or set(mapping) != {str(k) for k in native}:
        raise ValueError('query identity owner sets differ')
    bridge, tagged = {}, {}
    for owner, observations in native.items():
        indices = mapping[str(owner)]
        if len(indices) != len(observations) or len(set(indices)) != len(indices):
            raise ValueError('query identity index count differs')
        tagged[owner] = [replace(o, source_query_id=f'{owner}:full:{i}') for i, o in enumerate(full[owner])]
        for original, index in zip(observations, indices):
            if not isinstance(index, int) or not 0 <= index < len(full[owner]):
                raise ValueError('query identity index out of range')
            candidate = full[owner][index]
            fields = ('scene_id', 'instance_id', 'frame_id', 'feature_space_id',
                      'visible_area_px', 'crop_bbox_xyxy', 'source_config_hash')
            if (any(getattr(original, k) != getattr(candidate, k) for k in fields)
                    or not np.array_equal(original.feature, candidate.feature)):
                raise ValueError('native/full query identity differs')
            bridge[original.source_query_id] = tagged[owner][index].source_query_id
    if (sum(map(len, native.values())) != receipt['retained_query_count']
            or sum(map(len, full.values())) != receipt['full_query_count']):
        raise ValueError('query identity total count differs')
    return tagged, bridge


def instrument_mapper(source, filename):
    tree = ast.parse(source, filename=filename)
    mains = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']
    if len(mains) != 1:
        raise ValueError('expected one native main function')
    main = mains[0]
    if any(isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)) for node in ast.walk(main)):
        raise ValueError('native main return/yield requires an explicit capture contract')
    main.body.append(ast.parse('_static_capture(locals())').body[0])
    return compile(ast.fix_missing_locations(tree), filename, 'exec')


def capture_history(full, retained, output, *, max_top_vis):
    """Validate native top-area retention, then write all executed query records.

    B0 must continue using its original retained cache: full cache order is execution order.
    """
    keys = ('frame_id', 'feat', 'pose', 'box_2d', 'vis_area')
    result, mapping = {}, {}
    nonempty = {int(k) for k, v in full.items() if len(v['frame_id'])}
    if nonempty != {int(k) for k in retained}:
        raise ValueError('native retained owner set differs from complete history')
    for owner in sorted(nonempty):
        record, native = full[owner], retained[owner]
        count = len(record['frame_id'])
        if any(len(record[k]) != count for k in keys):
            raise ValueError('query history fields have unequal lengths')
        indices = np.argsort(np.asarray(record['vis_area']))[-max_top_vis:]
        for key in keys:
            if not np.array_equal(np.asarray(record[key])[indices], np.asarray(native[key])):
                raise ValueError(f'native retained {owner}/{key} contradicts full history')
        result[owner] = {k: np.asarray(record[k]).copy() for k in keys}
        result[owner]['color'] = np.asarray(native['color']).copy()
        mapping[str(owner)] = indices.tolist()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'full_query_cache.pkl').open('wb') as handle:
        pickle.dump(result, handle, protocol=4)
    receipt = {'status': 'PASS', 'history_scope': 'complete_executed_native_queries',
               'order': 'per_owner_execution_order', 'owner_count': len(result),
               'full_query_count': sum(len(r['frame_id']) for r in result.values()),
               'retained_query_count': sum(len(v) for v in mapping.values()),
               'native_to_full_indices': mapping,
               'native_retention': f'np.argsort(vis_area)[-{max_top_vis}:]',
               'baseline_rule': 'B0 uses original retained cache, not last8 of full history',
               'extra_image_queries': 0, 'gt_used': False}
    (output/'history_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt
