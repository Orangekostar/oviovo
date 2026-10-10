"""Real-map zero-update checks using prediction arrays and fixed projection only."""
from pathlib import Path
import time

import numpy as np

from static_ovmap.disagreement_query.outputs import build_payload

from .binding import load_scene
from .common import ConsumptionIndex, canonical_digest, verified, write, _array_digest


def zero_update_parity(g1, d2, raw, eligible, nearest, matched):
    rebuilt = build_payload(g1,d2,raw,{},eligible,nearest,matched,'ZERO_UPDATE_CHECK',{})
    checks = dict(geometry_exact=rebuilt.geometry==g1.geometry,
                  owner_array_exact=np.array_equal(rebuilt.owner_ids,g1.owner_ids),
                  semantic_array_exact=np.array_equal(rebuilt.semantic_labels,g1.semantic_labels),
                  official_current_ranks_exact=rebuilt.instance_ranks==g1.instance_ranks,
                  prediction_key_exact=rebuilt.prediction_key==g1.prediction_key)
    if not all(checks.values()):
        raise ValueError(f'Real zero-update whole-output parity failed: {checks}')
    return dict(checks, prediction_key=g1.prediction_key,source_vertices=len(g1.owner_ids),
                owners=len(np.unique(g1.owner_ids[g1.owner_ids>0])),
                owner_digest=_array_digest(g1.owner_ids),semantic_digest=_array_digest(g1.semantic_labels),
                rank_digest=canonical_digest(g1.instance_ranks),new_GPU_inference=0,new_GT_annotation_reads=0)


def verify_zero_update_scene(binding, scene):
    root = Path(binding['output_root']); bank = verified(root/'regression'/scene/'manifest.json')
    index = ConsumptionIndex(); started = time.perf_counter()
    inputs = dict(binding=binding['identity'],scene=scene,bank=bank['identity'],
                  check_producer=index.identity(__file__),
                  output_operator=index.identity(Path(__file__).parents[1]/'disagreement_query/outputs.py'))
    key = canonical_digest(inputs); path = root/'output_checks'/(scene+'.json')
    if path.exists():
        previous = verified(path)
        if previous['input_identity'] != key:
            raise ValueError('Real-map zero-update check inputs changed')
        return previous
    _, original = load_scene(binding,scene)
    proof = zero_update_parity(original.g1,original.d2,original.raw,bank['eligible_owners'],
                               original.nearest,original.matched)
    if proof['owners'] != len(bank['full_G1_positive_registry']):
        raise ValueError('Zero-update check omitted the full positive G1 registry')
    result = write(path,dict(status='COMPLETE',scene=scene,input_identity=key,input_contract=inputs,
                   eligible_owners=bank['eligible_owners'],**proof,elapsed_seconds=time.perf_counter()-started))
    print('REAL_ZERO_UPDATE_PARITY',scene,proof['owners'],flush=True)
    return result


def verify_zero_updates(binding):
    scenes = [s for names in binding['cohorts'].values() for s in names]
    rows = [verify_zero_update_scene(binding,s) for s in scenes]
    if len(rows) != 26:
        raise ValueError('Complete26-scene real zero-update checks required')
    return write(Path(binding['output_root'])/'output_checks/summary.json',dict(status='COMPLETE',
                 ordered_scenes=scenes,scene_receipts={r['scene']:r['identity'] for r in rows},
                 new_GPU_inference=0,new_GT_annotation_reads=0,whole_output_and_ranks_exact=True))
