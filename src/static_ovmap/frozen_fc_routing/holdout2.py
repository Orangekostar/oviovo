"""Four prelocked official families; annotation parsing occurs only after nominations."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time
from .common import ConsumptionIndex,PathResolver,arrays_record,canonical_digest,read,verified,write

from .transfer import require_transfer

def run(binding,fc):
    import torch
    from static_ovmap.learned_object_readout.inventory import inventory_roots,RAW_SUFFIXES
    from static_ovmap.module_validation.scannet_download import DownloaderLayout,transfer
    from static_ovmap.learned_object_readout.data import prepare_family
    from .data import ObjectLoader
    from .recognition import evaluate,load_head
    from .mechanisms import Collector
    from .diagnostics import summarize_rows
    root=Path(binding['output_root']);lock=require_transfer(binding);plan=verified(root/'H2_plan.json')
    if plan['identity']!=lock['H2_plan']:raise ValueError('H2 family names changed')
    path=root/'holdout2/receipt.json'
    if path.exists():
        previous=verified(path)
        if previous['status']!='PARTIAL_ACTIVE_STAGE':
            if previous['predictor_lock']!=lock['identity']:raise ValueError('Completed H2 predictors differ')
            return previous
        if previous['selected_names']!=plan['selected']:raise ValueError('Blocked H2 names differ')
        archive=path.parent/'failed_attempts'/f'{time.time_ns()}.json'
        archive.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,archive)
    started=time.perf_counter();hroot=root/'holdout2';hroot.mkdir(parents=True,exist_ok=True)
    index=ConsumptionIndex(root/'holdout2/verifications.json')
    index.identity(binding['scan_authority']['path'],binding['scan_authority'])
    if read(binding['scan_authority']['path']).get('authorization')!='USER_EXPLICITLY_CONFIRMED':
        raise ValueError('Bound ScanNet acquisition authority differs')
    acquisition=PathResolver(binding['path_map']).rewrite(read(Path(binding['prior_data_root'])/'acquisition_plan.json'))
    script=acquisition['layout']['script_path'];index.identity(script,{'sha256':acquisition['layout']['script_sha256']})
    layout=DownloaderLayout.from_script(Path(script))
    inventory=inventory_roots([str(hroot/'data/scannet'),*binding['data_roots'],str(Path(binding['lr_parent_root'])/'data/scannet')])
    selected=[]
    for scene in plan['selected']:
        existing=inventory['scans'].get(scene)
        if existing and existing['complete']:files=existing['files']
        else:
            files={suffix:str(hroot/'data/scannet/scans'/scene/(scene+suffix)) for suffix in RAW_SUFFIXES}
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    jobs={s:pool.submit(transfer,layout.scene_url(scene,s),Path(p)) for s,p in files.items()}
                    downloads={s:f.result() for s,f in jobs.items()}
                write(hroot/'downloads'/(scene+'.json'),dict(scene=scene,downloads=downloads,prelocked_plan=plan['identity']))
            except Exception as exc:
                write(path,dict(status='PARTIAL_ACTIVE_STAGE',scene=scene,error=repr(exc),missing_outcomes=None,
                     selected_names=plan['selected'],no_outcome_based_replacement=True,elapsed_seconds=time.perf_counter()-started))
                raise
        selected.append(dict(scene=scene,family=scene.split('_')[0],files=files))
    for n in ('class_split.json',):shutil.copyfile(root/n,hroot/n)
    tsv=Path(binding['prior_data_root'])/'scannetv2-labels.combined.tsv';dest=hroot/'data/scannet'/tsv.name
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(tsv,dest)
    shadow={**binding,'output_root':str(hroot)}
    # Same production generator and camera registration, with explicit new H2 role.
    manifests=[prepare_family(shadow,s,'H2') for s in selected]
    items=[o for m in manifests for o in m['objects']]
    feature_path=hroot/'features.json'
    if feature_path.exists():features=verified(feature_path)
    else:
        frames={}
        before=fc.images
        for obj in items:
            for f in obj['views']:
                key=obj['scene']+':'+str(f['frame_id'])
                if key not in frames:frames[key]=fc.capture(f)
        if len(frames)>384 or fc.images-before>384:raise ValueError('H2 FC image cap exceeded')
        features=write(feature_path,dict(status='COMPLETE',frames=frames,physical_images=len(frames),
                       physical_encodings=fc.images-before,physical_encodings_persisted=sum(r['physical_image_encodings'] for r in {v['key']:v for v in frames.values()}.values()),raw_C=1536,projected_D=768))
    results={};details={};summary=[];loader=ObjectLoader(fc,features)
    if items:
        fc_row=evaluate(None,loader,items,fc,base_only=False,save_path=hroot/'recognition/FC8.json');results['FC8']=fc_row['identity']
        for name,row in lock['heads'].items():
            head,_=load_head(row['checkpoint']['path'],fc.device)
            collector=Collector(binding,fc,loader,'H2',name)
            result=evaluate(head,loader,items,fc,base_only=False,reference=[r for r in fc_row['records'] if r['base']],
                            save_path=hroot/'recognition'/(name+'.json'),observer=collector)
            results[name]=result['identity'];details[name]=collector.finish(hroot/'diagnostics'/name)['identity']
        for name in results:
            value=verified(hroot/'recognition'/(name+'.json'));reference={r['key']:r for r in fc_row['records']};flat=[]
            for r in value['records']:
                for condition,prediction in enumerate(r['predictions']):
                    correct=int(prediction==r['class_id']);old=reference[r['key']]['predictions'][condition]
                    flat.append(dict(key=r['key'],family=r['family'],class_id=r['class_id'],base=r['base'],condition=condition,
                        method=name,correct=correct,NLL=r['NLL'][condition],correction=int(correct and old!=r['class_id']),harm=int(not correct and old==r['class_id'])))
            summary.extend(summarize_rows(flat))
        write(hroot/'recognition_summary.json',dict(status='COMPLETE',rows=summary,used_for_selection=False))
    novel=[o for o in items if not o['base']]
    index.write_memo(root/'holdout2/verifications.json')
    return write(path,dict(status='COMPLETE' if len(items)>=32 and len(novel)>=20 and len({o['family'] for o in novel})>=2 else 'LIMITED_CONFIRMATION_SUPPORT',predictor_lock=lock['identity'],
          plan=plan['identity'],selected_names=plan['selected'],family_manifests=[m['identity'] for m in manifests],
          original_objects=len(items),variant_records=4*len(items),families=len(manifests),recognition=results,diagnostics=details,
          novel_original_objects=len(novel),novel_families=len({o['family'] for o in novel}),
          novel_status='SUFFICIENT' if len(novel)>=20 and len({o['family'] for o in novel})>=2 else 'LIMITED_CONFIRMATION_SUPPORT',
          proposal_recognition_only=True,independent_whole_map_AP=False,features=features['identity'],elapsed_seconds=time.perf_counter()-started))
