"""One staged FP32 FC worker; exact tensor caches and task-owned region keys."""
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
from .common import ConsumptionIndex,PathResolver,_array_digest,canonical_digest,producer,read,unchanged_receipt,verified,write


def content_key(region,pixel_digest,model,text,ids,operators):
    return canonical_digest({'RGB_pixels':pixel_digest,'mask':region['mask_digest'],
        'pose_c2w':region['pose_c2w'],'intrinsics':region['intrinsics'],
        'physical_support':region['physical_support_identity'],'geometry':region['geometry'],
        'model':model,'text':text['sha256'],'ordered_ids':list(ids),'pooling':PROTOCOL,'operators':operators})


def run(job_path):
    job=verified(job_path);binding=load_binding(job['root']);root=Path(binding['output_root'])
    dest=root/'acquisition'/job['stage'];index=ConsumptionIndex(root/'acquisition/verifications.json')
    operators=producer(index,'fc_worker.py')+[index.identity(Path(__file__).parents[1]/'cvpr_compact/area_fallback.py'),
        index.identity(Path(__file__).parents[1]/'a7_evidence_upgrade/region_adapter.py')]
    key=canonical_digest({'job':job['identity'],'operators':operators})
    previous=unchanged_receipt(dest/'receipt.json',key,index)
    if previous and previous['status']=='COMPLETE':return previous
    cfg={**binding['assets']['execution_config'],'gpu':binding['gpu'],'path_map':binding['path_map']}
    cfg['gpu_lock']=str(Path(cfg['gpu_lock']).with_name('.visual-gpu-'+binding['gpu']+'.lock'))
    counts=Counter();records={};dependencies=[];session=None;loads=[];start=time.perf_counter();torch.set_num_threads(4)
    try:
        with exclusive_lock(cfg['gpu_lock']),torch.inference_mode():
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            for scene,request_ids in job['requests'].items():
                plan=verified(job['plans'][scene]['path']);index.identity(job['plans'][scene]['path'],job['plans'][scene])
                if plan['status']!='PLAN_LOCKED':raise ValueError('query plan is not locked')
                if len(set(request_ids))!=len(request_ids):raise ValueError('duplicate physical request in union')
                selected=[]
                for rid in request_ids:
                    r=plan['regions'][rid];o=plan['owners'][str(r['owner'])]
                    if 'anchor' in job['stage'] or job['category']=='ENGINEERING':
                        if r['frame_id']!=o['anchor']:raise ValueError('candidate FC scores requested before choices seal')
                    else:
                        choices=verified(job['choices'][scene]['path'])
                        index.identity(job['choices'][scene]['path'],job['choices'][scene])
                        if choices['status']!='CHOICES_LOCKED':raise ValueError('second queries require independent choice locks')
                        allowed={x['second_region_id'] for row in choices['owners'].values() for x in row.values() if x.get('second_region_id')}
                        if rid not in allowed:raise ValueError('unselected future second view requested')
                    selected.append({**r,'physical_support_identity':o['support']['sha256'],'geometry':plan['geometry']})
                if not selected:
                    records[scene]={};continue
                data=PathResolver(binding['path_map']).rewrite(read(binding['scenes'][scene]['context']['path']))
                current=FCSession(cfg,data,index,cache=None)
                if session is None:session=current
                elif current.model_key!=session.model_key:raise ValueError('physical FC model changes across scenes')
                session.text,session.ids,session.text_identity=current.text,current.ids,current.text_identity
                parents=binding['assets']['read_only_dense_cache_roots']
                cache=ContentCache(parents,root/'content_cache/fc',session.model_key,index=index,resolver=PathResolver(binding['path_map']))
                groups={};records[scene]={}
                for r in selected:groups.setdefault(r['frame_id'],[]).append(r)
                for fid,regions in sorted(groups.items()):
                    first=regions[0];dependencies.append(index.identity(first['RGB']['path'],first['RGB']))
                    image=cv2.imread(first['RGB']['path'],cv2.IMREAD_UNCHANGED);counts['RGB_decode_attempts']+=1
                    if image is None or image.dtype!=np.uint8 or list(image.shape[:2])!=first['original_size']:raise ValueError('original RGB decode/shape failed')
                    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);pixel_digest=_array_digest(rgb);pending=[]
                    for r in regions:
                        dependencies.append(index.identity(r['mask']['path'],r['mask']))
                        ck=content_key(r,pixel_digest,session.model_key,session.text_identity,session.ids,operators)
                        cached=None if job['category']=='ENGINEERING' else cache.lookup('regions',ck)
                        if cached:
                            with np.load(cached['arrays']['path'],allow_pickle=False) as arr:
                                cos=arr['cosines']
                                if cached['available'] and (cos.shape!=(len(session.ids),) or not np.isfinite(cos).all()):raise ValueError('cached full vocabulary scores invalid')
                            dependencies.append(index.identity(cached['arrays']['path'],cached['arrays']))
                            records[scene][r['region_id']]={**cached,'region_id':r['region_id'],'content_key':ck,
                                'scores':cos.tolist() if cached['available'] else None,'cache_origin':cached['cache_origin']}
                            counts['region_cache_hits']+=1
                        else:pending.append((r,ck))
                    if not pending:continue
                    if session.model is None:
                        session.load_model();loads.append({'seconds':session.model_load_seconds,'model':session.model_key,
                            'weight_audit':session.weight_audit,'python':sys.executable,'torch':torch.__version__})
                        write(dest/'model_loads'/(str(time.time_ns())+'.json'),loads[-1])
                    tensor,size=image_tensor(rgb,'cuda')
                    tensor_key=canonical_digest({'model':session.model_key,'tensor':_array_digest(tensor.cpu().numpy())})
                    image_key=fc_image_identity({'image_sha256':first['RGB']['sha256']},session.model_key)
                    dense_row=cache.lookup('dense',tensor_key)
                    if dense_row:
                        if dense_row['image_content_key']!=image_key:counts['exact_tensor_different_RGB_container']+=1
                        with np.load(dense_row['arrays']['path'],allow_pickle=False) as arr:dense=torch.from_numpy(arr['dense'].copy()).cuda()
                        counts['parent_dense_cache_hits' if dense_row['cache_origin']=='PARENT' else 'task_dense_cache_hits']+=1
                    else:
                        if counts['FC_encoding_attempts']+job['spent_images']>=job['image_limit']:raise RuntimeError('image-encoding cap would be exceeded')
                        counts['FC_encoding_attempts']+=1
                        dense=session.operators['extract_features_convnext'](SimpleNamespace(clip_model=session.model),tensor)['clip_vis_dense']
                        torch.cuda.synchronize();counts['new_FC_image_inputs']+=1
                        dense_row=cache.write('dense',tensor_key,{'dense':dense.cpu().numpy()},{'input_tensor_key':tensor_key,'image_content_key':image_key})
                    dependencies.append(index.identity(dense_row['arrays']['path'],dense_row['arrays']))
                    if dense.dtype!=torch.float32 or dense.ndim!=4 or dense.shape[0]!=1 or not torch.isfinite(dense).all():raise ValueError('invalid FP32 dense FC representation')
                    for region,ck in pending:
                        if counts['region_pool_attempts']+job['spent_pools']>=job['pool_limit']:raise RuntimeError('pool/head cap would be exceeded')
                        with np.load(region['mask']['path'],allow_pickle=False) as arr:mask=arr['mask']
                        if mask.dtype!=bool or list(mask.shape)!=region['original_size'] or _array_digest(mask)!=region['mask_digest']:raise ValueError('locked original mask changed')
                        signed,_,_=signed_mask(mask,size,tensor.shape[-2:],dense.shape[-2:],'cuda')
                        counts['region_pool_attempts']+=1;counts['FULL_pool_attempts' if region['role']=='FULL' else 'PROBE_pool_attempts']+=1
                        base={'content_key':ck,'region_id':region['region_id'],'owner':region['owner'],'frame_id':fid,
                            'role':region['role'],'tile':region['tile'],'RGB':region['RGB'],'RGB_pixel_digest':pixel_digest,
                            'depth':region['depth'],'mask':region['mask'],'mask_digest':region['mask_digest'],
                            'pose_c2w':region['pose_c2w'],'intrinsics':region['intrinsics'],
                            'physical_support_identity':region['physical_support_identity'],'geometry':region['geometry'],
                            'FC_model':session.model_key,'FC_text':session.text_identity,'valid_ids':session.ids,
                            'pooling_protocol':PROTOCOL,'input_tensor_key':tensor_key,'image_content_key':image_key,
                            'pixels':region['pixels'],'bbox':region['bbox'],'precision':'float32'}
                        available=True;reason='FROZEN_FC_V2';vector=np.empty((0,),np.float32);cos=np.empty((0,),np.float32);audit={}
                        try:
                            feature,audit=region_vector(session.model,session.operators,dense,signed)
                        except ValueError as exc:
                            if str(exc) not in ('EMPTY_AREA_MASK_SUPPORT','INVALID_REGION_FEATURE'):raise
                            available=False;reason=str(exc);counts['unavailable_regions']+=1
                            if reason=='INVALID_REGION_FEATURE':counts['head_projections']+=1
                        if available:
                            counts['head_projections']+=1;vector=feature.cpu().numpy().copy();value=cosine_record(session,vector,region['pixels'])
                            if not value['feature_available']:raise RuntimeError('unexplained FC score unavailability')
                            cos=np.asarray(value['scores'],np.float32)
                            if cos.shape!=(len(session.ids),) or not np.isfinite(cos).all():raise RuntimeError('nonfinite/misaligned FC cosines')
                            counts['successful_pool_heads']+=1
                        engineering_parity=None
                        if job['category']=='ENGINEERING' and region['role']=='FULL':
                            if counts['region_pool_attempts']+job['spent_pools']>=job['pool_limit']:raise RuntimeError('engineering cap would be exceeded')
                            counts['region_pool_attempts']+=1;counts['FULL_pool_attempts']+=1
                            reference_signed,_,_=signed_mask(mask,size,tensor.shape[-2:],dense.shape[-2:],'cuda')
                            reference_available=True;reference_reason='FROZEN_FC_V2';reference_cos=None;reference_feature=None
                            try:reference_feature,_=region_vector(session.model,session.operators,dense,reference_signed)
                            except ValueError as exc:
                                if str(exc) not in ('EMPTY_AREA_MASK_SUPPORT','INVALID_REGION_FEATURE'):raise
                                reference_available=False;reference_reason=str(exc)
                                if reference_reason=='INVALID_REGION_FEATURE':counts['head_projections']+=1
                            if reference_available:
                                counts['head_projections']+=1;counts['successful_pool_heads']+=1
                                reference_cos=np.asarray(cosine_record(session,reference_feature.cpu().numpy().copy(),region['pixels'])['scores'],np.float32)
                            checks={'available_exact':reference_available==available,'signed_mask_exact':bool(torch.equal(reference_signed,signed)),
                                'feature_exact':bool(np.array_equal(reference_feature.cpu().numpy(),vector)) if available and reference_available else None,
                                'cosines_exact':bool(np.array_equal(reference_cos,cos)) if available and reference_available else None,
                                'known_unavailability_exact':reference_reason==reason}
                            if checks['available_exact'] is False or checks['signed_mask_exact'] is False or checks['known_unavailability_exact'] is False or checks['feature_exact'] is False or checks['cosines_exact'] is False:
                                raise ValueError('real ordinary-FC adapter parity failed')
                            engineering_parity={'checks':checks,'same_actual_dense_forward':True,
                                'reference_pool_and_head_executed_separately':True,'compared_full_vocabulary':len(session.ids)}
                        metadata={**base,'available':available,'reason':reason,'pooling_audit':audit}
                        cached=cache.write('regions',ck,{'feature':vector,'cosines':cos},metadata)
                        dependencies.append(index.identity(cached['arrays']['path'],cached['arrays']))
                        records[scene][region['region_id']]={**cached,'scores':cos.tolist() if available else None,
                            'engineering_adapter_parity':engineering_parity}
                    del dense,tensor
                if set(records[scene])!=set(request_ids):raise RuntimeError('selected FC union omitted required region')
            torch.cuda.synchronize()
        result=write(dest/'receipt.json',{'status':'COMPLETE','input_identity':key,'job_identity':job['identity'],
            'stage':job['stage'],'category':job['category'],'records':records,'counts':dict(counts),
            'dependencies':list({d['path']:d for d in dependencies}.values()),'model_loads':loads,
            'elapsed_seconds':time.perf_counter()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'actual_python':sys.executable,
            'new_NQ':0,'new_AnyUp':0,'new_text_encoding':0,'new_maps':0})
        index.write_memo(root/'acquisition/verifications.json');print('FC_STAGE',job['stage'],dict(counts),flush=True);return result
    except Exception as exc:
        write(dest/'failures'/(str(time.time_ns())+'.json'),{'status':'EXECUTION_FAILED','input_identity':key,
            'job_identity':job['identity'],'stage':job['stage'],'category':job['category'],
            'known_attempt_counts':dict(counts),'unknown_success_counts':None,'reason':type(exc).__name__+': '+str(exc),
            'elapsed_seconds':time.perf_counter()-start,'partial_records':records})
        index.write_memo(root/'acquisition/verifications.json');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);run(parser.parse_args().job)
