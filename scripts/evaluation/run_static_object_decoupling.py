#!/usr/bin/env python3
"""Frozen two-run prediction-only object/semantic/ranking intervention."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.candidate_semantics import array_digest,semantic_cache_key,score_views,OfflineNativeEncoder
from src.static_ovmap.object_arbitration import arbitrate,validate_final,query_quality,mask_areas,intersection_count
from src.static_ovmap.final_instance_ranking import final_source_ranks


def read(p):
    return json.loads(Path(p).read_text())


def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def identity(p):
    p=Path(p)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha256_file(p)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=read(args.config)
    start=time.perf_counter()
    out=Path(cfg['runtime_assets_to_resolve']['new_output_root'])/'predictions'
    out.mkdir(parents=True,exist_ok=False)
    write(out/'frozen_config.json',cfg)
    old=read(ROOT/cfg['existing_asset_configs'][0]);local=read(ROOT/cfg['existing_asset_configs'][1])
    pool=Path(cfg['recorded_roots_to_verify']['frozen_prediction_root'])
    shared=Path(cfg['recorded_roots_to_verify']['local_ownership_root'])/'predictions/shared'
    prior=read(shared.parent/'input_binding.json')
    asset=cfg['runtime_assets_to_resolve'];text_path=Path(asset['native_text_cache'])
    text_meta=read(text_path.with_suffix('.json'));text=np.load(text_path)
    inventory={};model=Path(asset['native_model_root'])
    checks={model/'model.safetensors':text_meta['feature_space_identity']['model_sha256'],
            model/'preprocessor_config.json':text_meta['feature_space_identity']['preprocessor_sha256'],
            text_path:text_meta['output_sha256']}
    for path,expected in checks.items():
        inventory[str(path)]=identity(path)
        if inventory[str(path)]['sha256']!=expected:
            raise ValueError('native model/text identity mismatch: '+str(path))
    for path in sorted(model.glob('*.json')):
        inventory[str(path)]=identity(path)
    receipt=read(asset['rgb_identity_receipt']);frames=[];rgb_bindings={}
    for entry in read(shared/'summary.json')['frames']:
        fid=entry['frame_id'];path=shared/f'frame_{fid:06d}.npz';frame=dict(np.load(path));frame['frame_id']=fid
        if len(np.unique(frame['pixel_ids']))!=len(frame['pixel_ids']):
            raise ValueError('shared zbuffer pixel duplicates')
        frames.append(frame);inventory[str(path)]=identity(path)
        rgb=Path(local['scene_root'])/'results'/f'frame{fid:06d}.jpg'
        inventory[str(rgb)]=identity(rgb)
        if receipt['input_sha256'].get(str(rgb))!=inventory[str(rgb)]['sha256']:
            raise ValueError('RGB differs from bound cloud receipt')
        rgb_bindings[fid]=inventory[str(rgb)]
    coordinates=None;prepared={}
    for run in cfg['runs']:
        directory=pool/run;bound=next(x for x in prior['runs'] if x['run']==run)
        for name in ['masks.npy','coord.npy','candidates.json','AT_U00.json']:
            path=directory/name;inventory[str(path)]=identity(path)
            if inventory[str(path)]['sha256']!=bound['sources'][name]:raise ValueError('frozen source mismatch')
        coord=np.load(directory/'coord.npy',mmap_mode='r')
        if coordinates is not None and not np.array_equal(coordinates,coord):raise ValueError('run coordinate order differs')
        coordinates=coord
        masks=np.load(directory/'masks.npy',mmap_mode='r');records=read(directory/'candidates.json')
        o=read(directory/'AT_O_AREA.json');before=np.load(o['owner_path'])
        ovi=[r['canonical_index'] for r in records if r['source']=='OVI'];sf=[r['canonical_index'] for r in records if r['source']=='SpaCeFormer']
        # Historical source spelling is asserted, rather than silently losing candidates.
        if len(ovi)+len(sf)!=len(records):raise ValueError('unknown source family')
        t0path=next(r['t0'] for r in old['runs'] if r['id']==run)
        if t0path is None:raise ValueError('unsupported T0 config shape')
        inventory[str(t0path)]=identity(t0path)
        if inventory[str(t0path)]['sha256']!=bound['t0_sha256']:raise ValueError('T0 identity changed')
        t0=np.load(t0path)
        quality=query_quality(records,t0['query_ids'],t0['objectness_mask_scores'])
        stage=time.perf_counter()
        e2,active,actions,counts=arbitrate(masks,before,ovi,sf,quality,frames,cfg['geometry_actions'])
        dest=out/run;dest.mkdir();np.save(dest/'OD_E2_OBJECT_owners.npy',e2)
        write(dest/'actions.json',{'actions':actions,'counts':counts,'seconds':time.perf_counter()-stage})
        areas=mask_areas(masks);targets={}
        for i in ovi:
            strength=0.
            for j in sf:
                if records[i]['original_class_id']==records[j]['original_class_id']:continue
                inter=intersection_count(masks[i],masks[j]);coverage=min(inter/max(1,areas[i]),inter/max(1,areas[j]))
                if coverage>=cfg['semantic_reassessment']['disagreement_pair_directional_coverage_min']:
                    strength=max(strength,float(coverage))
            if strength:targets[i]=strength
        requests={}
        for method,owners,ids in [('OD_E1_SEMANTIC',before,sorted(targets)),('OD_E3_OBJECT_SEMANTIC',e2,[i for i in active if i in targets or i in sf])]:
            for i in ids:
                mask=owners==i+1;digest=array_digest(mask)
                key=(i,digest)
                if key not in requests:
                    requests[key]={'candidate':i,'mask_hash':digest,'mask':mask,'methods':[],
                                   'priority':0 if i in sf else 1,'strength':targets.get(i,0.)}
                requests[key]['methods'].append(method)
        selected=sorted(requests.values(),key=lambda x:(x['priority'],-x['strength'],x['candidate'],x['mask_hash']))
        cap=cfg['semantic_reassessment']['target_masks_per_run_max']
        for index,target in enumerate(selected):target['budget_included']=index<cap
        prepared[run]=(masks,records,o,before,e2,active,selected)
        write(dest/'geometry_archive.json',{'active':active,'owner':identity(dest/'OD_E2_OBJECT_owners.npy'),
              'targets':[{k:v for k,v in t.items() if k!='mask'} for t in selected],
              'ancestry':[{**records[i],'final_mask_sha256':array_digest(e2==i+1),
                'action_id':next((a['action_id'] for a in actions if a['accepted'] and i in a['children']),None)} for i in active],
              'GT_input':False})
        print(run,'E2 active',len(active),'accepted',sum(a['accepted'] for a in actions),'targets',len(selected),flush=True)
    source_files=[Path(__file__),ROOT/'src/static_ovmap/candidate_semantics.py',ROOT/'src/static_ovmap/object_arbitration.py',ROOT/'src/static_ovmap/final_instance_ranking.py']
    source_identity={str(p.relative_to(ROOT)):sha256_file(p) for p in source_files}
    write(out/'input_binding.json',{'inventory':inventory,'source':source_identity,'prior_binding':identity(shared.parent/'input_binding.json'),
          'GT_input':False,'config':identity(args.config),'shared_positive_mask_only':True})
    domain_digest=array_digest(coordinates)
    encoder=OfflineNativeEncoder(str(model),cfg['device']);cache={};cache_dir=out/'semantic_cache';cache_dir.mkdir()
    for run,(masks,records,o,before,e2,active,targets) in prepared.items():
        dest=out/run;decisions=[];labels={m:list(o['labels']) for m in ['OD_E1_SEMANTIC','OD_E3_OBJECT_SEMANTIC']}
        run_batches=encoder.batches;attempted=0;cache_hits=0;usable=0
        for target in targets:
            i=target['candidate'];mask=target['mask'];view_records=[];features=[];weights=[]
            if target['budget_included']:
                visible=sorted([(int(np.count_nonzero(mask[f['point_ids']])),f['frame_id'],f) for f in frames],key=lambda x:(-x[0],x[1]))
                for count,fid,frame in visible[:cfg['semantic_reassessment']['views_per_target_max']]:
                    if count<cfg['semantic_reassessment']['usable_region_pixels_min']:continue
                    attempted+=1
                    pixel=np.zeros((680,1200),bool);pixel.flat[frame['pixel_ids'][mask[frame['point_ids']]]]=True
                    ys,xs=np.where(pixel);bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]
                    binding={'source_domain_sha256':domain_digest,'target_mask_sha256':target['mask_hash'],
                      'frame_id':fid,'rgb_sha256':rgb_bindings[fid]['sha256'],'bbox':bbox,
                      'feature_space_id':text_meta['feature_space_id'],'precision':cfg['precision'],
                      'encoder_source_sha256':source_identity['src/static_ovmap/candidate_semantics.py'],
                      'processor_sha256':text_meta['feature_space_identity']['preprocessor_sha256'],
                      'crop':text_meta['feature_space_identity']['crop']}
                    key=semantic_cache_key(binding,pixel)
                    record={**binding,'key':key,'visible_pixels':count,'pixel_mask_sha256':array_digest(pixel),
                            'mask_bbox_occupancy':float(count/max(1,(bbox[2]-bbox[0]+1)*(bbox[3]-bbox[1]+1)))}
                    if key in cache:
                        feature=cache[key];cache_hits+=1;record['cache_hit']=True
                    else:
                        with Image.open(rgb_bindings[fid]['path']) as image:rgb=np.asarray(image.convert('RGB'))
                        try:feature=encoder.encode(rgb,pixel,bbox)
                        except ValueError as exc:
                            if str(exc)!='degenerate_native_crop':raise
                            record['abstention']=str(exc);view_records.append(record);continue
                        np.save(cache_dir/(key+'.npy'),feature);cache[key]=feature;record['cache_hit']=False
                        if not (out/'real_crop_smoke.json').exists():
                            write(out/'real_crop_smoke.json',{**record,'feature_shape':list(feature.shape),'six_crop_forward':True})
                    features.append(feature);weights.append(count);view_records.append(record);usable+=1
            decision=score_views(features,weights,text['text'],text['valid_ids'],records[i]['original_class_id'],
                        text_meta['feature_space_id'],str(text['feature_space_id']),canonical=text['canonical'],
                        margin=cfg['semantic_reassessment']['aggregate_cosine_advantage_over_current_min'],
                        agreement=cfg['semantic_reassessment']['view_top1_agreement_min'])
            if not target['budget_included']:decision['reason']='target_budget_excluded'
            decision.update({k:v for k,v in target.items() if k!='mask'});decision['views']=view_records
            decisions.append(decision)
            for method in target['methods']:labels[method][i]=decision['class_id']
        write(dest/'semantics.json',{'targets':decisions,'attempted_view_requests':attempted,'usable_view_requests':usable,
                     'cache_hits':cache_hits,'encoder_batches':encoder.batches-run_batches,'crop_inputs':6*(encoder.batches-run_batches)})
        for method,owners,kept,lab in [('OD_E1_SEMANTIC',before,o['kept'],labels['OD_E1_SEMANTIC']),
                   ('OD_E2_OBJECT',e2,active,o['labels']),('OD_E3_OBJECT_SEMANTIC',e2,active,labels['OD_E3_OBJECT_SEMANTIC']),
                   ('OD_E4_FINAL_RANK',e2,active,labels['OD_E3_OBJECT_SEMANTIC'])]:
            validate_final(owners,masks,kept)
            path=dest/(method+'_owners.npy')
            if not path.exists():np.save(path,owners)
            ranks=list(o['rank_scores'])
            if method=='OD_E4_FINAL_RANK':
                for i,value in zip(kept,final_source_ranks(owners,kept)):ranks[i]=float(value)
            write(dest/(method+'.json'),{'status':'COMPLETE_PREDICTION','run':run,'condition':method,
                'owner_path':str(path),'owner_sha256':sha256_file(path),'kept':kept,'labels':lab,'rank_scores':ranks,
                'rank_scores_serialized':[f'{x:.6f}' for x in ranks],'GT_input':False,
                'final_mask_hashes':{str(i):array_digest(owners==i+1) for i in kept}})
        print(run,'semantic changes',sum(d['reason']=='accepted' for d in decisions),'new batches',encoder.batches-run_batches,flush=True)
    write(out/'prediction_manifest.json',{'status':'COMPLETE','source':source_identity,'command':[sys.executable,*sys.argv],
            'encoder_batches':encoder.batches,'crop_inputs':encoder.crop_inputs,'new_3d_segmenter_forwards':0,
            'new_2d_segmenter_forwards':0,'model_load_seconds':encoder.load_seconds,'crop_seconds':encoder.crop_seconds,
            'encoding_seconds':encoder.encoding_seconds,'seconds':time.perf_counter()-start,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'peak_gpu_allocated_bytes':encoder.torch.cuda.max_memory_allocated(cfg['device'])})


if __name__=='__main__':main()
