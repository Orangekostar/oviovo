"""Conditional resident-model cold calls; never invoked without a strict pass."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np
import torch

from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.cvpr_compact.region_worker import FCSession
from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache

from . import fc_worker
from .acquisition import seal_choices
from .binding import load_binding, load_scene
from .common import ConsumptionIndex, PathResolver, canonical_digest, plain, read, verified, write
from .outputs import predict_scene
from .policies import score_bank
from .query_plan import plan_scene


def simple_choice(binding, scene, plan, policy):
    owners = {}
    frames = {int(f['frame_id']): f for f in plan['frames']}
    for owner, row in plan['owners'].items():
        if not row['query_eligible']:
            owners[owner] = {policy: {'policy': policy, 'second_region_id': None,
                                     'reason': row['reason'], 'visible_evidence_keys': []}}
            continue
        bank = [row['anchor'], *[f for f in row['candidate_frame_ids'] if f != row['anchor']]]
        with np.load(row['support']['path'], allow_pickle=False) as arrays:
            positions = {int(f): i for i, f in enumerate(arrays['frame_ids'])}
            second, audit = score_bank(policy, xyz=arrays['xyz'], area=arrays['area'],
                footprints=arrays['O'][[positions[f] for f in bank]], center=row['center'],
                cameras=[np.asarray(frames[f]['pose_c2w'])[:3, 3] for f in bank],
                pixels=[row['views'][str(f)]['pixels'] for f in bank], frame_ids=bank)
        owners[owner] = {policy: {**audit, 'anchor_region_id': f"{owner}:{row['anchor']}:FULL",
            'second_region_id': f'{owner}:{second}:FULL', 'visible_evidence_keys': [],
            'choice_before_second_acquisition': True, 'candidate_semantic_scores_accessed': False}}
    return write(Path(binding['output_root'])/'choices'/scene/'receipt.json', plain({
        'status': 'CHOICES_LOCKED', 'scene': scene, 'owners': owners, 'GT_used': False,
        'future_candidate_scores_accessed': False, 'input_identity': canonical_digest([plan['identity'], policy])}))


def run(root):
    source = load_binding(root)
    regression = verified(Path(root)/'regression.json')
    if regression.get('strict_target_pass') is not True:
        raise ValueError('cold worker requires an actual full strict target pass')
    torch.set_num_threads(4)
    selected = regression['selected_method']
    control = 'QS_US' if selected != 'QS_US' else 'AREA_MEAN'
    qs, us = verified(Path(root)/'screen_selection.json')['QS'], verified(Path(root)/'screen_selection.json')['US']
    paths = {selected: tuple(regression['paths'][selected]),
             control: ('AREA', 'MEAN') if control == 'AREA_MEAN' else (qs, us)}
    aliases = {control: selected} if paths[control] == paths[selected] else {}
    if aliases:
        paths.pop(control)
    names = source['specification']['timing']['scenes']
    cfg = {**source['assets']['execution_config'], 'gpu': source['gpu'], 'path_map': source['path_map']}
    lock = Path(cfg['gpu_lock']).with_name('.visual-gpu-'+source['gpu']+'.lock')
    cfg['gpu_lock'] = str(lock)
    index = ConsumptionIndex(Path(root)/'cold/model_verifications.json')
    data = PathResolver(source['path_map']).rewrite(read(source['scenes'][names[0]]['context']['path']))
    session = FCSession(cfg, data, index, cache=None)
    original_factory, original_cache, original_lock = fc_worker.FCSession, fc_worker.ContentCache, fc_worker.exclusive_lock
    rows, images, pools = [], 0, 0
    with exclusive_lock(lock), torch.inference_mode():
        session.load_model()
        index.write_memo(Path(root)/'cold/model_verifications.json')
        def resident_factory(config, context, leaf_index, cache=None):
            current = original_factory(config, context, leaf_index, cache=cache)
            if current.model_key != session.model_key:
                raise ValueError('cold call changed the resident model')
            current.model, current.operators = session.model, session.operators
            return current
        fc_worker.FCSession = resident_factory
        fc_worker.ContentCache = lambda parents, *args, **kwargs: original_cache([], *args, **kwargs)
        fc_worker.exclusive_lock = lambda _: nullcontext()
        try:
            sequence = [(s, m) for s in names for m in paths]
            for repetition in (0, 1):
                for ordinal, (scene, method) in enumerate(sequence if repetition == 0 else list(reversed(sequence))):
                    call_root = Path(root)/'cold/calls'/f'{repetition}_{ordinal:02d}_{scene}_{method}'
                    receipt_path = call_root/'call.json'
                    if receipt_path.exists():
                        row = verified(receipt_path)
                        rows.append(row); images += row['FC_image_inputs']; pools += row['pool_heads']
                        continue
                    # This binding has no derived parent observation or visual cache.
                    binding = write(call_root/'source_binding.json', {**source, 'output_root': str(call_root),
                        'minimal_root': str(call_root/'no_parent_observation_cache')})
                    load_scene(binding, scene)  # Fixed G1, source scores, text/locations resident before t0.
                    policy, updater = paths[method]
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    plan = plan_scene(binding, scene); t_plan = time.perf_counter()
                    anchor_ids = []
                    for owner, item in plan['owners'].items():
                        if item['query_eligible']:
                            anchor_ids.append(f"{owner}:{item['anchor']}:FULL")
                            if policy == 'DISAGREEMENT': anchor_ids.extend(item['probe_region_ids'])
                    def acquire(stage, requests, choice=None, additional_images=0, additional_pools=0):
                        leaf_index = ConsumptionIndex(call_root/'input_verifications.json')
                        job = write(call_root/'acquisition'/stage/'job.json', {
                            'root': str(call_root), 'stage': stage, 'category': 'COLD',
                            'plans': {scene: leaf_index.identity(call_root/'plans'/scene/'manifest.json')},
                            'requests': {scene: requests}, 'choices': {} if choice is None else {
                                scene: leaf_index.identity(call_root/'choices'/scene/'receipt.json')},
                            'spent_images': images+additional_images, 'spent_pools': pools+additional_pools,
                            'image_limit': 1024, 'pool_limit': 3072})
                        return fc_worker.run(call_root/'acquisition'/stage/'job.json')
                    anchor = acquire('cold_anchor', anchor_ids); t_anchor = time.perf_counter()
                    if policy == 'DISAGREEMENT':
                        choices = seal_choices(binding, {scene: plan}, anchor)[scene]
                    else:
                        choices = simple_choice(binding, scene, plan, policy)
                    t_choice = time.perf_counter()
                    second_ids = sorted({v[policy]['second_region_id'] for v in choices['owners'].values()
                                         if v[policy].get('second_region_id')})
                    second = acquire('cold_second', second_ids, choices,
                        anchor['counts'].get('FC_encoding_attempts', 0), anchor['counts'].get('region_pool_attempts', 0))
                    t_second = time.perf_counter()
                    prediction = predict_scene(binding, 'cold', scene, plan, choices, anchor, second,
                                               {method: (policy, updater)})
                    torch.cuda.synchronize(); finish = time.perf_counter()
                    count_images = sum(r['counts'].get('FC_encoding_attempts', 0) for r in (anchor, second))
                    count_pools = sum(r['counts'].get('region_pool_attempts', 0) for r in (anchor, second))
                    images += count_images; pools += count_pools
                    if count_images > 32 or count_pools > 96 or images > 1024 or pools > 3072:
                        raise RuntimeError('cold envelope exceeded')
                    key = prediction['methods'][method]['prediction_key']
                    if method == 'AREA_MEAN':
                        scientific = 'QS_US' if (qs, us) == ('AREA', 'MEAN') else None
                    else:
                        scientific = method
                    equal = key == verified(Path(root)/'predictions/regression'/scene/'receipt.json')['methods'][scientific]['prediction_key'] if scientific else None
                    if equal is False: raise ValueError('cold output differs from frozen science')
                    rows.append(write(receipt_path, {'status': 'COMPLETE', 'scene': scene, 'method': method,
                        'repetition': repetition, 'seconds': finish-begin,
                        'stage_seconds': {'preparation': t_plan-begin, 'anchor': t_anchor-t_plan,
                            'selection': t_choice-t_anchor, 'second': t_second-t_choice, 'payload_write': finish-t_second},
                        'FC_image_inputs': count_images, 'pool_heads': count_pools,
                        'peak_allocated_bytes': max(r['peak_cuda_allocated_bytes'] for r in (anchor, second)),
                        'peak_reserved_bytes': max(r['peak_cuda_reserved_bytes'] for r in (anchor, second)),
                        'prediction_key': key, 'scientific_output_exact': equal,
                        'derived_caches_empty_at_call_start': True, 'model_load_in_boundary': False,
                        'OS_page_cache_controlled': False, 'simple_unused_probe_heads': 0 if policy != 'DISAGREEMENT' else None}))
            if len(rows) != 16*len(paths) or len(rows) > 32:
                raise ValueError('cold campaign coverage differs')
            for scene in names:
                for method in paths:
                    repeated = [r['prediction_key'] for r in rows if r['scene'] == scene and r['method'] == method]
                    if len(set(repeated)) != 1: raise ValueError('cold repetitions changed output')
        finally:
            fc_worker.FCSession, fc_worker.ContentCache, fc_worker.exclusive_lock = original_factory, original_cache, original_lock
    return write(Path(root)/'timing.json', {'status': 'CONDITIONAL_COLD_COMPLETE', 'rows': rows,
        'calls': len(rows), 'logical_aliases': aliases, 'new_FC_images': images, 'new_pool_heads': pools,
        'model_load_seconds_excluded': session.model_load_seconds, 'strict_pass_identity': regression['identity'],
        'NOT_end_to_end_system': True, 'online_FPS_claim': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    run(parser.parse_args().root)
