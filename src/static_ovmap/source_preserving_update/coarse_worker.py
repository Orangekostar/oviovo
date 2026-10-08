"""Same-view coarse FC only. Never constructs or executes an AnyUp model."""

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
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.cvpr_compact.region_worker import FCSession
from static_ovmap.evidence_exploration.acquisition import cosine_record
from static_ovmap.minimal_instance_repair.recognition_plan import unpack_region
from static_ovmap.module_validation.contracts import atomic_write_json,canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex,PathResolver,read
from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache
from static_ovmap.recovery_wave2.recovery_sources import fc_image_identity

from .binding import load_binding,seal,free_space
from .evidence import common_domain,paired_means


def acquire_frame(binding,manifest,fid,session,cache,index):
    root=Path(binding['output_root']);free_space(root)
    required=set(manifest['required_frames'][str(fid)])
    views={v['region_id']:v for r in manifest['incumbents'].values() for v in r['full_views'] if v['region_id'] in required}
    if set(views)!=required:raise ValueError('locked required FULL records differ')
    first=next(iter(views.values()))
    if session.model_key!=manifest['FC_model'] or session.ids!=manifest['valid_ids'] or session.text_identity['sha256']!=manifest['FC_text']['sha256']:
        raise ValueError('same-view FC model/text/class order changed')
    producer={'worker':index.identity(__file__),'coarse_operator':index.identity(Path(__file__).parents[1]/'cvpr_compact/area_fallback.py'),
              'original_preprocessing':index.identity(Path(__file__).parents[1]/'a7_evidence_upgrade/region_adapter.py')}
    key=canonical_digest({'evidence_identity':manifest['identity'],'views':views,'producer':producer,'FC_model':session.model_key,
        'FC_text':session.text_identity,'dtype':'float32'})
    path=root/'coarse'/manifest['scene']/str(fid)/'receipt.json'
    if path.exists():
        old=read(path);_verified_identity(old)
        if old['status']=='COMPLETE' and old['input_identity']==key:
            for item in old['dependencies']:index.identity(item['path'],item)
            return old
        if old['status']=='COMPLETE':raise ValueError('coarse dependency changed; invalidate this frame and descendants')
        path.rename(path.with_name('receipt.failed_'+str(time.time_ns())+'.json'))
    start=time.perf_counter();counts=Counter();dependencies=[]
    record={'status':'RUNNING','scene':manifest['scene'],'frame_id':int(fid),'input_identity':key,
        'records':{},'counts':{},'producer':producer,'dependencies':dependencies}
    atomic_write_json(path,record)
    try:
        if session.model is None:
            session.load_model()
            atomic_write_json(root/'coarse/model_loads'/(str(time.time_ns())+'.json'),seal({
                'FC_model':session.model_key,'FC_model_load_seconds':session.model_load_seconds,
                'weight_audit':session.weight_audit,'models_required':['FC_FROZEN'],
                'actual_worker_python':sys.executable,'torch':torch.__version__,
                'new_AnyUp_model_loads':0,'new_AnyUp_QK':0}))
        dependencies.append(index.identity(first['RGB']['path'],first['RGB']))
        image=cv2.imread(first['RGB']['path'],cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('original uint8 RGB decode failed')
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);tensor,size=image_tensor(rgb,'cuda')
        tensor_key=canonical_digest({'model':session.model_key,'tensor':_array_digest(tensor.cpu().numpy())})
        image_key=fc_image_identity({'image_sha256':first['RGB']['sha256']},session.model_key)
        if any(v['input_tensor_key']!=tensor_key or v['image_content_key']!=image_key
               or v['RGB']['sha256']!=first['RGB']['sha256'] for v in views.values()):
            raise ValueError('FULL image tensor/model identity differs from the saved parent')
        dense_row=None;direct=Path(first['parent_dense_receipt'])
        if direct.exists() and Path(first['parent_dense_arrays']['path']).exists():
            dependencies.append(index.identity(direct));dense_row=read(direct)
            if (dense_row['input_tensor_key']!=tensor_key or dense_row['image_content_key']!=image_key
                    or dense_row['arrays']['sha256']!=first['parent_dense_arrays']['sha256']):
                raise ValueError('direct parent dense receipt changed')
            dense_row={**dense_row,'receipt_path':str(direct),'cache_origin':'DIRECT_PARENT'}
            counts['direct_parent_dense_cache_hits']+=1
        if dense_row is None:
            dense_row=cache.lookup('dense',tensor_key)
            if dense_row is not None:
                if dense_row['image_content_key']!=image_key:raise ValueError('matching dense cache changed RGB')
                counts['parent_dense_cache_hits' if dense_row['cache_origin']=='PARENT' else 'task_dense_cache_hits']+=1
        if dense_row is not None:
            dependencies.append(index.identity(dense_row['arrays']['path'],dense_row['arrays']))
            with np.load(dense_row['arrays']['path'],allow_pickle=False) as arrays:
                dense=torch.from_numpy(arrays['dense'].copy()).to('cuda')
        else:
            counts['FC_encoding_attempts']+=1
            dense=session.operators['extract_features_convnext'](SimpleNamespace(clip_model=session.model),tensor)['clip_vis_dense']
            torch.cuda.synchronize();counts['new_FC_image_inputs']+=1
            dense_row=cache.write('dense',tensor_key,{'dense':dense.cpu().numpy()},
                                 {'input_tensor_key':tensor_key,'image_content_key':image_key})
            dependencies.append(index.identity(dense_row['arrays']['path'],dense_row['arrays']))
        if (dense.dtype!=torch.float32 or list(dense.shape)!=first['actual_dense_shape']
                or dense.ndim!=4 or dense.shape[0]!=1 or not torch.isfinite(dense).all()):
            raise ValueError('FC dense shape/dtype/finite contract differs')
        for rid,view in sorted(views.items()):
            mask=unpack_region(view,index);dependencies.append(index.identity(view['mask']['path'],view['mask']))
            if list(mask.shape)!=view['original_size'] or list(size)!=view['resized_size'] or list(tensor.shape[-2:])!=view['padded_size']:
                raise ValueError('exact FULL original/resized/padded shapes changed')
            signed,_,_=signed_mask(mask,size,tensor.shape[-2:],dense.shape[-2:],'cuda')
            region_key=canonical_digest({'scene':manifest['scene'],'owner':view['owner'],'support':view['support_hash'],
                'RGB':image_key,'tensor':tensor_key,'mask':view['mask_digest'],'bbox':view['bbox'],
                'original_size':view['original_size'],'resized_size':view['resized_size'],'padded_size':view['padded_size'],
                'model':session.model_key,'head_and_operator':producer,'text':session.text_identity,
                'valid_ids':session.ids,'protocol':PROTOCOL,'dtype':'float32'})
            item={'region_id':rid,'frame_id':int(fid),'bank':view['bank'],'mask_digest':view['mask_digest'],
                'feature_content_key':region_key,'AnyUp_feature_content_key':view['feature_content_key'],
                'FC_model':session.model_key,'FC_text':session.text_identity,'valid_ids':session.ids,
                'input_tensor_key':tensor_key,'image_content_key':image_key}
            counts['coarse_pool_attempts']+=1
            try:
                feature,audit=region_vector(session.model,session.operators,dense,signed)
            except ValueError as exc:
                if str(exc) not in ('EMPTY_AREA_MASK_SUPPORT','INVALID_REGION_FEATURE'):raise
                counts['unavailable_coarse_pools']+=1
                record['records'][rid]={**item,'available':False,'reason':str(exc),'scores':None};continue
            vector=feature.cpu().numpy().copy();value=cosine_record(session,vector,view['pixels'])
            if not value['feature_available'] or not np.isfinite(value['scores']).all():raise ValueError('nonfinite coarse cosine record')
            counts['successful_coarse_pools']+=1
            record['records'][rid]={**item,**value,'available':True,'pooling':audit,'reason':'SAME_VIEW_COARSE_COMPLETE'}
        torch.cuda.synchronize()
        record.update(status='COMPLETE',actual_dense_shape=list(dense.shape),dense_receipt=dense_row['receipt_path'],
            dense_arrays=dense_row['arrays'],input_tensor_key=tensor_key,image_content_key=image_key,
            actual_worker_python=sys.executable,new_AnyUp_QK=0,new_NQ_inference=0,new_frontend_inference=0,new_projections=0)
    except BaseException as exc:
        record.update(status='FAILED',reason=type(exc).__name__+': '+str(exc));raise
    finally:
        record.update(counts=dict(counts),elapsed_seconds=time.perf_counter()-start,
                      dependencies=list({r['path']:r for r in dependencies}.values()))
        atomic_write_json(path,seal(record))
    print('COARSE',manifest['scene'],fid,dict(counts),flush=True)
    return seal(record)


def seal_eligibility(binding,manifest,frames):
    if any(f['status']!='COMPLETE' for f in frames):raise ValueError('unfinished coarse is an execution block')
    coarse={rid:item for f in frames for rid,item in f['records'].items()}
    required={r for ids in manifest['required_frames'].values() for r in ids}
    if set(coarse)!=required:raise ValueError('coarse coverage differs from locked required pairs')
    rows={}
    for owner,row in manifest['incumbents'].items():
        eligible,reasons=common_domain(row,coarse)
        if eligible:
            values=[coarse[v['region_id']] for v in row['full_views']]
            a,c=paired_means(row['full_views'],values,manifest['valid_ids'],manifest['valid_ids'])
            if not np.array_equal(a,row['historical']['aggregate_scores']):raise ValueError('FULL aggregate differs from parent')
        rows[owner]={'eligible':eligible,'exclusions':reasons,'selected':True,
            'protected_raw_zero_count':row['protected_raw_zero_count'],'FULL_pairs':len(row['full_views'])}
    result=seal({'status':'COMMON_DOMAIN_LOCKED','scene':manifest['scene'],'evidence_identity':manifest['identity'],
        'coarse_frames':{str(f['frame_id']):f['identity'] for f in frames},'incumbents':rows,
        'selected_count':len(rows),'eligible_count':sum(r['eligible'] for r in rows.values()),'GT_used':False})
    atomic_write_json(Path(binding['output_root'])/'eligibility'/(manifest['scene']+'.json'),result)
    return result


def run(root):
    binding=load_binding(root);root=Path(binding['output_root']);inventory=read(root/'evidence_manifest.json')
    cfg={**binding['assets']['execution_config'],'gpu':binding['gpu'],'path_map':binding['path_map']}
    lock=Path(cfg['gpu_lock']);cfg['gpu_lock']=str(lock.with_name('.visual-gpu-'+binding['gpu']+'.lock'))
    index=ConsumptionIndex(root/'coarse/verifications.json');session=None;counts=Counter();scenes={};blocked=[]
    torch.set_num_threads(4)
    with exclusive_lock(cfg['gpu_lock']),torch.inference_mode():
        for scene in inventory['scenes']:
            manifest=read(root/'evidence'/scene/'manifest.json');frames=[]
            try:
                data=PathResolver(binding['path_map']).rewrite(read(binding['scenes'][scene]['context']['path']))
                current=FCSession(cfg,data,index,cache=None)
                if session is None:
                    session=current
                elif current.model_key!=session.model_key:raise ValueError('worker physical FC model changed')
                session.text,session.ids,session.text_identity=current.text,current.ids,current.text_identity
                parents=[str(Path(binding['parent_root'])/'content_cache/fc'),*binding['assets']['read_only_dense_cache_roots']]
                cache=ContentCache(parents,root/'content_cache/fc',session.model_key,index=index,resolver=PathResolver(binding['path_map']))
                for fid in sorted(map(int,manifest['required_frames'])):
                    try:frames.append(acquire_frame(binding,manifest,fid,session,cache,index))
                    except (OSError,RuntimeError) as exc:
                        # One identical transient retry; deterministic representation absence is handled inside the leaf.
                        atomic_write_json(root/'coarse'/scene/str(fid)/'retry.json',seal({'reason':str(exc),'identical_retry':1}))
                        frames.append(acquire_frame(binding,manifest,fid,session,cache,index))
                domain=seal_eligibility(binding,manifest,frames);scenes[scene]=domain['identity']
                for f in frames:counts.update(f['counts'])
            except (OSError,ValueError,RuntimeError,KeyError) as exc:
                block=seal({'scene':scene,'status':'BLOCKED_COARSE','reason':type(exc).__name__+': '+str(exc)})
                atomic_write_json(root/'coarse'/scene/'blocked.json',block);blocked.append(block)
            index.write_memo(root/'coarse/verifications.json')
    if counts['successful_coarse_pools']>inventory['actual_coarse_region_bound'] or counts['new_FC_image_inputs']>inventory['actual_unique_FC_image_bound']:
        raise ValueError('actual successful acquisition exceeds inventory bounds')
    result=seal({'status':'COARSE_COMPLETE' if len(scenes)==26 else 'PARTIAL_DEPENDENCY_BLOCK',
        'scenes':scenes,'blocked':blocked,'counts':dict(counts),'inventory_identity':inventory['identity'],
        'new_AnyUp_QK':0,'new_NQ_inference':0,'new_frontend_inference':0,'new_projections':0})
    atomic_write_json(root/'coarse/summary.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args();run(args.root)
