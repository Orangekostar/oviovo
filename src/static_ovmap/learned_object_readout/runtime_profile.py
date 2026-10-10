"""One cache-only replay of the unchanged real-map readout, outside science."""
from pathlib import Path
import shutil
import time

import numpy as np

from .common import ConsumptionIndex, verified, write


def cache_only_profile(binding):
    import torch
    from . import features, prediction
    root = Path(binding['output_root'])
    receipt = root/'runtime_profile/receipt.json'
    original = verified(root/'recognition/regression.json')
    if receipt.exists():
        previous = verified(receipt)
        if previous['original_readout']!=original['identity']:
            raise ValueError('Cached profile belongs to a different readout')
        return previous
    replay = root/'runtime_profile/replay'
    if (replay/'recognition/regression.json').exists():
        raise ValueError('Completed unreceipted replay must be preserved and reviewed')
    replay.mkdir(parents=True,exist_ok=True)
    (replay/'features').mkdir(exist_ok=True)
    for name in ('class_split.json','data_profile.json','dev_nomination.json','repeat_nomination.json'):
        shutil.copyfile(root/name,replay/name)
    for name,target in (('external',root/'external'),('regression',root/'regression'),
                        ('features/scientific',root/'features/scientific')):
        path = replay/name
        if not path.exists(): path.symlink_to(target,target_is_directory=True)
    probe_binding = {k:v for k,v in binding.items() if k!='identity'}
    probe_binding['output_root'] = str(replay)
    probe_binding = write(replay/'source_binding.json',probe_binding)
    capture_source = verified(root/'features/regression.json')
    dense_operator = next(iter(capture_source['frames'].values()))['dense_projection']
    immutable = ConsumptionIndex()
    tracked = [root/'recognition/regression.json',root/'features/regression.json',
               root/'dev_nomination.json',root/'repeat_nomination.json']
    for nomination in (verified(root/'dev_nomination.json'),verified(root/'repeat_nomination.json')):
        tracked.extend(Path(r['path']) for r in nomination['checkpoints'].values())
    before = {str(p):immutable.identity(p) for p in tracked}
    grid_count = len(list((root/'features/scientific/grids').glob('*.json')))
    stages = {}; captured_instances = []; started = time.perf_counter()
    original_fc,original_write = features.FrozenFC,prediction.write

    class CachedFC(original_fc):
        def __init__(self,*args,**kwargs):
            torch.cuda.reset_peak_memory_stats()
            super().__init__(*args,**kwargs)
            if (self.dense_operator['sha256'],self.dense_operator['bytes'])!=(dense_operator['sha256'],dense_operator['bytes']):
                raise ValueError('Replay dense projection differs from scientific cache')
            self.dense_operator = dense_operator
            def cache_miss(*args,**kwargs):
                raise RuntimeError('CACHE_ONLY_PROFILE_MISS: no encoder or cache writes allowed')
            self.raw_cache.lookup = cache_miss
            captured_instances.append(self)

    def observed_write(path,values):
        result = original_write(path,values)
        if Path(path)==replay/'features/regression.json':
            torch.cuda.synchronize()
            stages['CACHE_ONLY_MAP_CAPTURE'] = dict(seconds=time.perf_counter()-started,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                timing_boundary='LOCK_WAIT_MODEL_LOAD_RGB_PREPROCESS_CACHE_VERIFY_AND_CAPTURE_RECEIPT',
                memory_boundary='FROZEN_FC_LOAD_AND_CACHE_CAPTURE_NO_LEARNED_HEADS')
        return result

    features.FrozenFC,prediction.write = CachedFC,observed_write
    try:
        result = prediction.regression_readout(probe_binding)
        torch.cuda.synchronize()
        total_seconds = time.perf_counter()-started
    finally:
        features.FrozenFC,prediction.write = original_fc,original_write
    if len(captured_instances)!=1 or captured_instances[0].images!=0:
        raise ValueError('Cache-only profile performed an encoder call')
    if result['records'].keys()!=original['records'].keys() or result['methods']!=original['methods']:
        raise ValueError('Replay changed methods or object coverage')
    max_difference = 0.
    for key,row in result['records'].items():
        reference = original['records'][key]
        with np.load(row['arrays']['path'],allow_pickle=False) as actual, np.load(reference['arrays']['path'],allow_pickle=False) as expected:
            for name in ('methods','valid_ids'):
                if not np.array_equal(actual[name],expected[name]): raise ValueError('Replay changed array ordering')
            for name in ('embeddings','cosines'):
                difference = float(np.max(np.abs(actual[name]-expected[name])))
                max_difference = max(max_difference,difference)
                if difference>1e-6: raise ValueError('Replay differs from original readout')
            if not np.array_equal(actual['cosines'].argmax(-1),expected['cosines'].argmax(-1)):
                raise ValueError('Replay changed semantic predictions')
    for p in tracked:
        if immutable.identity(p)!=before[str(p)]: raise ValueError('Profile changed scientific input')
    if len(list((root/'features/scientific/grids').glob('*.json')))!=grid_count:
        raise ValueError('Profile wrote an additional scientific grid')
    stages['CACHE_ONLY_MAP_READOUT'] = dict(seconds=total_seconds-stages['CACHE_ONLY_MAP_CAPTURE']['seconds'],
        peak_allocated_GiB=result['peak_allocated_bytes']/2**30,
        peak_reserved_GiB=result['peak_reserved_bytes']/2**30,
        timing_boundary='HEAD_LOAD_CACHED_GRIDS_FORWARD_ARRAY_EXPORT_AND_LOCK_RELEASE',
        memory_boundary='AFTER_HEAD_LOAD_RESET_ALL_SIX_HEADS_AND_FROZEN_FC_RESIDENT')
    return write(receipt,dict(status='CACHE_ONLY_PROFILE_COMPLETE',original_readout=original['identity'],
        replay_readout=result['identity'],stages=stages,elapsed_seconds=total_seconds,
        verification_seconds=time.perf_counter()-started-total_seconds,
        objects=len(result['records']),methods=result['methods'],frames=len(capture_source['frames']),
        learned_head_forwards=len(result['records'])*6,physical_image_encodings=0,optimizer_updates=0,
        cold_timing_runs=0,excluded_from_science=True,max_absolute_difference=max_difference,
        all_predictions_equal=True,scientific_inputs_unchanged=True))
