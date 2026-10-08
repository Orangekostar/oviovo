"""Frozen FP32 FC signed-mask pools on original RGB; no AnyUp or new text."""

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import cv2
import numpy as np
import torch

from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor,signed_mask
from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.cvpr_compact.area_fallback import region_vector,PROTOCOL
from static_ovmap.cvpr_compact.region_worker import FCSession
from static_ovmap.evidence_exploration.acquisition import cosine_record
from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache
from static_ovmap.recovery_wave2.recovery_sources import fc_image_identity

from .binding import load_binding
from .common import (ConsumptionIndex,PathResolver,_array_digest,canonical_digest,
                     read,verified,write)


def run(root):
    binding=load_binding(root);root=Path(binding['output_root']);cfg={**binding['assets']['execution_config'],
        'gpu':binding['gpu'],'path_map':binding['path_map']}
    cfg['gpu_lock']=str(Path(cfg['gpu_lock']).with_name('.visual-gpu-'+binding['gpu']+'.lock'))
    index=ConsumptionIndex(root/'readout/verifications.json');session=None;all_counts=Counter();scenes={};loads=[]
    torch.set_num_threads(4)
    with exclusive_lock(cfg['gpu_lock']),torch.inference_mode():
        for scene in binding['scenes']:
            manifest=verified(root/'semantic_masks'/scene/'manifest.json');path=root/'readout'/scene/'receipt.json'
            producer=[index.identity(__file__),index.identity(Path(__file__).parents[1]/'cvpr_compact/area_fallback.py'),
                      index.identity(Path(__file__).parents[1]/'a7_evidence_upgrade/region_adapter.py')]
            key=canonical_digest({'manifest':manifest['identity'],'FC':binding['assets']['identity'],'producer':producer})
            if path.exists():
                prior=verified(path)
                if prior['input_identity']!=key:raise ValueError('paired FC dependency changed')
                if prior['status']=='COMPLETE':
                    for dependency in prior['dependencies']:index.identity(dependency['path'],dependency)
                    scenes[scene]=prior['identity'];continue
            begin=time.perf_counter();counts=Counter();records={};dependencies=[]
            data=PathResolver(binding['path_map']).rewrite(read(binding['scenes'][scene]['context']['path']))
            current=FCSession(cfg,data,index,cache=None)
            if session is None:session=current
            elif current.model_key!=session.model_key:raise ValueError('physical FC model changed across scenes')
            session.text,session.ids,session.text_identity=current.text,current.ids,current.text_identity
            parents=binding['assets']['read_only_dense_cache_roots']
            cache=ContentCache(parents,root/'content_cache/fc',session.model_key,index=index,resolver=PathResolver(binding['path_map']))
            groups={}
            for rid,region in manifest['regions'].items():groups.setdefault(region['frame_id'],[]).append(region)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            for fid,regions in sorted(groups.items()):
                first=regions[0];valid=[r for r in regions if r['qualified']]
                for region in regions:
                    dependencies.append(index.identity(region['mask']['path'],region['mask']))
                    if not region['qualified']:records[region['region_id']]={'available':False,'scores':None,'reason':'RAW_MASK_BELOW_100_PIXELS_OR_BBOX',
                        'pixels':region['pixels'],'mask_digest':region['mask_digest']}
                if not valid:continue
                if session.model is None:
                    session.load_model();loads.append({'FC_model':session.model_key,'seconds':session.model_load_seconds,
                        'weight_audit':session.weight_audit,'worker_python':sys.executable,'torch':torch.__version__})
                    write(root/'readout/model_loads'/(str(time.time_ns())+'.json'),loads[-1])
                dependencies.append(index.identity(first['RGB']['path'],first['RGB']))
                image=cv2.imread(first['RGB']['path'],cv2.IMREAD_UNCHANGED)
                if image is None or image.dtype!=np.uint8 or list(image.shape[:2])!=first['original_size']:raise ValueError('original full-resolution RGB decode/alignment failed')
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);tensor,size=image_tensor(rgb,'cuda')
                tensor_key=canonical_digest({'model':session.model_key,'tensor':_array_digest(tensor.cpu().numpy())})
                image_key=fc_image_identity({'image_sha256':first['RGB']['sha256']},session.model_key)
                dense_row=cache.lookup('dense',tensor_key)
                if dense_row is not None:
                    if dense_row['image_content_key']!=image_key:raise ValueError('FC dense cache pixel/model identity mismatch')
                    dependencies.append(index.identity(dense_row['arrays']['path'],dense_row['arrays']))
                    with np.load(dense_row['arrays']['path'],allow_pickle=False) as arrays:dense=torch.from_numpy(arrays['dense'].copy()).cuda()
                    counts['parent_dense_cache_hits' if dense_row['cache_origin']=='PARENT' else 'task_dense_cache_hits']+=1
                else:
                    counts['FC_encoding_attempts']+=1
                    dense=session.operators['extract_features_convnext'](SimpleNamespace(clip_model=session.model),tensor)['clip_vis_dense']
                    torch.cuda.synchronize();counts['new_FC_image_inputs']+=1
                    dense_row=cache.write('dense',tensor_key,{'dense':dense.cpu().numpy()},
                        {'input_tensor_key':tensor_key,'image_content_key':image_key})
                    dependencies.append(index.identity(dense_row['arrays']['path'],dense_row['arrays']))
                if dense.dtype!=torch.float32 or dense.ndim!=4 or dense.shape[0]!=1 or not torch.isfinite(dense).all():raise ValueError('frozen FC dense representation invalid')
                for region in valid:
                    with np.load(region['mask']['path'],allow_pickle=False) as arrays:mask=arrays['mask']
                    if _array_digest(mask)!=region['mask_digest']:raise ValueError('paired semantic mask changed')
                    signed,_,_=signed_mask(mask,size,tensor.shape[-2:],dense.shape[-2:],'cuda');counts['region_pool_attempts']+=1
                    base={'region_id':region['region_id'],'mask_digest':region['mask_digest'],'RGB':region['RGB'],
                        'input_tensor_key':tensor_key,'image_content_key':image_key,'FC_model':session.model_key,
                        'FC_text':session.text_identity,'valid_ids':session.ids,'pooling_protocol':PROTOCOL,
                        'original_size':region['original_size'],'resized_size':list(size),'padded_size':list(tensor.shape[-2:]),
                        'dense_shape':list(dense.shape),'pixels':region['pixels'],'bbox':region['bbox']}
                    try:feature,audit=region_vector(session.model,session.operators,dense,signed)
                    except ValueError as exc:
                        if str(exc) not in ('EMPTY_AREA_MASK_SUPPORT','INVALID_REGION_FEATURE'):raise
                        counts['unavailable_region_pools']+=1
                        records[region['region_id']]={**base,'available':False,'scores':None,'reason':str(exc)};continue
                    value=cosine_record(session,feature.cpu().numpy().copy(),region['pixels'])
                    if not value['feature_available'] or not np.isfinite(value['scores']).all():raise RuntimeError('FC cosine vector invalid')
                    counts['successful_region_pools']+=1
                    records[region['region_id']]={**base,**value,'available':True,'pooling':audit,'reason':'FROZEN_FC_ORIGINAL_SIGNED_MASK'}
                del dense,tensor
            torch.cuda.synchronize()
            if set(records)!=set(manifest['regions']):raise RuntimeError('paired FC omitted a selected raw-source record')
            row=write(path,{'status':'COMPLETE','scene':scene,'input_identity':key,'regions':records,
                'dependencies':list({d['path']:d for d in dependencies}.values()),'counts':dict(counts),
                'elapsed_seconds':time.perf_counter()-begin,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'new_AnyUp':0,'new_NQ':0,
                'FC_model':session.model_key,'FC_text':session.text_identity,'actual_worker_python':sys.executable})
            index.write_memo(root/'readout/verifications.json');all_counts.update(counts);scenes[scene]=row['identity']
            print('PAIRED_FC',scene,dict(counts),flush=True)
    if all_counts['region_pool_attempts']>128 or all_counts['new_FC_image_inputs']>64:raise RuntimeError('fixed semantic acquisition budget exceeded')
    return write(root/'readout/acquisition_summary.json',{'status':'FC_ACQUISITION_COMPLETE','scenes':scenes,
        'actual_new_work_counts':dict(all_counts),'model_loads':loads,'new_AnyUp':0,'new_NQ':0,'new_maps':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);run(parser.parse_args().root)
