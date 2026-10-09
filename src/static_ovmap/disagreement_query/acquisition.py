"""Policy evidence access and staged ordinary-FC acquisition (no future reads)."""
from types import MappingProxyType
from pathlib import Path
import os

import numpy as np

from static_ovmap.backbone_wave1.runtime import execute

from .common import REPO,POLICIES,ConsumptionIndex,canonical_digest,plain,read,verified,write
from .policies import anchor_interest,score_bank


class EvidenceAccess:
    def __init__(self,records,allowed):
        self._records=records;self.allowed=frozenset(allowed)

    def get(self,key):
        if key not in self.allowed:raise ValueError('evidence not visible before independent selection seal')
        if key not in self._records:raise ValueError('allowed evidence has not been acquired')
        return MappingProxyType(self._records[key])


def _spent(binding,category,screen_only=False):
    root=Path(binding['output_root'])/'acquisition';counts={'images':0,'pools':0}
    for path in root.glob('*/receipt.json'):
        r=verified(path)
        if r['category']!=category or (screen_only and not r['stage'].startswith('screen_')):continue
        counts['images']+=r['counts'].get('FC_encoding_attempts',0);counts['pools']+=r['counts'].get('region_pool_attempts',0)
    for path in root.glob('*/failures/*.json'):
        r=verified(path)
        if r['category']!=category or (screen_only and not r['stage'].startswith('screen_')):continue
        counts['images']+=r['known_attempt_counts'].get('FC_encoding_attempts',0)
        counts['pools']+=r['known_attempt_counts'].get('region_pool_attempts',0)
    return counts


def acquire(binding,stage,plans,requests,*,category='SCIENCE',choices=None):
    root=Path(binding['output_root']);dest=root/'acquisition'/stage;index=ConsumptionIndex(root/'input_verifications.json')
    plan_files={scene:index.identity(root/'plans'/scene/'manifest.json') for scene in plans}
    choice_files={scene:index.identity(root/'choices'/scene/'receipt.json') for scene in (choices or {})}
    semantics={'root':str(root),'stage':stage,'category':category,'plans':plan_files,'requests':requests,'choices':choice_files}
    semantics_key=canonical_digest(semantics);job_path=dest/'job.json'
    if job_path.exists():
        job=verified(job_path)
        if job['semantic_identity']!=semantics_key:raise ValueError('staged FC request inputs changed')
    else:
        spent=_spent(binding,category,screen_only=stage.startswith('screen_'))
        limits=binding['specification']['resources']
        if category=='ENGINEERING':image_limit=4;pool_limit=12
        elif stage.startswith('screen_'):image_limit=128;pool_limit=576
        else:image_limit=832;pool_limit=3040
        job=write(job_path,{**semantics,'semantic_identity':semantics_key,'spent_images':spent['images'],
            'spent_pools':spent['pools'],'image_limit':image_limit,'pool_limit':pool_limit})
    index.write_memo(root/'input_verifications.json')
    worker=REPO/'src/static_ovmap/disagreement_query/fc_worker.py'
    result_path=dest/'receipt.json'
    worker_index=ConsumptionIndex(root/'acquisition/verifications.json')
    operators=[worker_index.identity(worker),worker_index.identity(REPO/'src/static_ovmap/cvpr_compact/area_fallback.py'),
        worker_index.identity(REPO/'src/static_ovmap/a7_evidence_upgrade/region_adapter.py')]
    key=canonical_digest({'job':job['identity'],'operators':operators})
    if result_path.exists():
        result=verified(result_path)
        if result['input_identity']!=key:raise ValueError('staged FC producer changed')
        if result['status']=='COMPLETE':
            for dep in result['dependencies']:worker_index.identity(dep['path'],dep)
            worker_index.write_memo(root/'acquisition/verifications.json');return result
    previous_failures=list((dest/'failures').glob('*.json'))
    if len(previous_failures)>1:raise RuntimeError('bounded retry exhausted for this FC leaf')
    if previous_failures and verified(previous_failures[0])['input_identity']==key:
        raise RuntimeError('retry requires a concrete producer/input fix, not identical rerun')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=binding['gpu'],PYTHONPATH=str(REPO/'src')+':'+str(REPO),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    command=execute([binding['FC_python'],'-u','-m','static_ovmap.disagreement_query.fc_worker','--job',job_path],
                    REPO,dest/'run.log',env=env)
    result=verified(result_path)
    write(dest/'execution.json',{'status':'ACTUAL_WORKER_TERMINAL','command':command,'receipt_identity':result['identity']})
    return result


def anchor_requests(plans):
    requests={}
    for scene,plan in plans.items():
        rows=[]
        for o,r in plan['owners'].items():
            if not r['query_eligible']:continue
            rows.append(f"{o}:{r['anchor']}:FULL");rows.extend(r['probe_region_ids'])
        requests[scene]=rows
    return requests


def seal_choices(binding,plans,anchor_receipt):
    root=Path(binding['output_root']);index=ConsumptionIndex(root/'input_verifications.json');choices={}
    for scene,plan in plans.items():
        key=canonical_digest({'plan':plan['identity'],'anchor_acquisition':anchor_receipt['identity'],
            'producer':[index.identity(__file__),index.identity(Path(__file__).with_name('policies.py'))]})
        path=root/'choices'/scene/'receipt.json'
        if path.exists():
            row=verified(path)
            if row['input_identity']!=key:raise ValueError('independent choice dependencies changed')
            choices[scene]=row;continue
        records=anchor_receipt['records'][scene];owners={};frames={int(f['frame_id']):f for f in plan['frames']}
        ids=list(map(int,read(binding['scenes'][scene]['context']['path'])['models']['native']['valid_ids']))
        temperature=float(read(binding['scenes'][scene]['context']['path'])['temperatures']['F'])
        for owner,r in plan['owners'].items():
            if not r['query_eligible']:
                owners[owner]={policy:{'policy':policy,'reason':r['reason'],'second_region_id':None,
                    'visible_evidence_keys':[],'choice_before_second_acquisition':True} for policy in POLICIES};continue
            anchor=r['anchor'];bank=[anchor,*[f for f in r['candidate_frame_ids'] if f!=anchor]]
            with np.load(r['support']['path'],allow_pickle=False) as arr:
                positions={int(fid):i for i,fid in enumerate(arr['frame_ids'])};rows=[positions[f] for f in bank]
                x,a,footprint=arr['xyz'],arr['area'],arr['O'][rows];anchor_uv=arr['uv'][positions[anchor]]
            arid=f'{owner}:{anchor}:FULL';allowed=[arid,*r['probe_region_ids']]
            view=EvidenceAccess(records,allowed);full=view.get(arid);tiles={}
            for rid in r['probe_region_ids']:
                tile=view.get(rid)
                if tile['available']:tiles[int(tile['tile'])]=np.asarray(tile['scores'],np.float64)
            interest,heterogeneity=anchor_interest(a,footprint[0],anchor_uv,r['anchor_bbox'],
                np.asarray(full['scores'],np.float64) if full['available'] else None,tiles,
                ids.index(r['old_class']),ids.index(r['other_class']),temperature)
            owners[owner]={}
            for policy in POLICIES:
                second,audit=score_bank(policy,xyz=x,area=a,footprints=footprint,center=r['center'],
                    cameras=[np.asarray(frames[f]['pose_c2w'])[:3,3] for f in bank],
                    pixels=[r['views'][str(f)]['pixels'] for f in bank],frame_ids=bank,
                    interest=interest if policy=='DISAGREEMENT' else None)
                owners[owner][policy]={**audit,'old_class':r['old_class'],'other_class':r['other_class'],
                    'anchor_region_id':arid,'second_region_id':f'{owner}:{second}:FULL',
                    'visible_evidence_keys':allowed if policy=='DISAGREEMENT' else [],
                    'visible_content_keys':[records[k]['content_key'] for k in allowed] if policy=='DISAGREEMENT' else [],
                    'heterogeneity':heterogeneity if policy=='DISAGREEMENT' else None,
                    'choice_before_second_acquisition':True,'candidate_semantic_scores_accessed':False}
        choices[scene]=write(path,plain({'status':'CHOICES_LOCKED','scene':scene,'input_identity':key,
            'plan_identity':plan['identity'],'anchor_acquisition_identity':anchor_receipt['identity'],
            'owners':owners,'GT_used':False,'future_candidate_scores_accessed':False}))
    index.write_memo(root/'input_verifications.json');return choices


def second_requests(choices,policies=POLICIES):
    return {scene:sorted({row[p]['second_region_id'] for row in choice['owners'].values() for p in policies
                         if row[p].get('second_region_id')}) for scene,choice in choices.items()}
