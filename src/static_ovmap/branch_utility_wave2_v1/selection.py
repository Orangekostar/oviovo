"""Finite seed17 DEV selection, then immutable seed29 repeat."""
from pathlib import Path
from functools import cmp_to_key
import numpy as np
import torch
from static_ovmap.preservation_routing.selection import summarize
from .common import verified,write,immutable_write,arrays_record,ConsumptionIndex,METHODS
from .capture import recognition_records
from .utility_model import UtilityRegressor,Normalizer
from .training import load_bank,ARMS
TOL=1e-10

def select_action(actions,scores,tau,coupled=False):
    legal=[i for i,a in enumerate(actions) if a['kind']!='KEEP' and (not coupled or a['kind']=='B')]
    if not legal:return 0
    order={'KEEP':0,'G':1,'L':2,'B':3}
    winner=min(legal,key=lambda i:(-float(scores[i]),order[actions[i]['kind']],actions[i]['frame_id']))
    return winner if float(scores[winner])>tau else 0

def compare_candidates(a,b):
    for key,sign in [('A8',-1),('M8',-1),('C8',-1),('A4',-1),('mean_edits',1),('step',1),('tau',-1)]:
        d=a.get(key,0)-b.get(key,0)
        if abs(d)>TOL:return sign*(1 if d>0 else -1)
    order=['FG2',*METHODS];d=order.index(a['method'])-order.index(b['method'])
    if d:return 1 if d>0 else -1
    if a['method']=='STATIC':
        order=['MEAN','MASK_AREA','COSINE_MEDOID'];d=order.index(a['base'])-order.index(b['base'])
        if d:return 1 if d>0 else -1
        d=b['gamma']-a['gamma']
        if d:return 1 if d>0 else -1
    return 0

def promotion(candidate,reference):
    return candidate['A8']>=reference['A8']+.005-TOL and all(candidate[k]>=reference[k]-.005-TOL for k in ('M8','C8','A4')) and candidate['net_corrections8']>0

def lock_profiles(root,profiles,c_star):
    return immutable_write(Path(root)/'selection_lock.json',dict(status='LOCKED_SEED17_ONLY',profiles=profiles,C_STAR=c_star,seed29_candidates_read=False,selection_uses_maps=False))

def candidate_metrics(records4,records8,reference4,reference8,edits4,edits8,profile):
    four=summarize(records4,reference4);eight=summarize(records8,reference8)
    return dict(profile,A8=eight['A'],M8=eight['M'],C8=eight['C'],A4=four['A'],mean_edits=.5*(edits4+edits8),net_corrections8=eight['net_corrections'],corrections8=eight['corrections'],harm8=eight['harm'],metrics4=four,metrics8=eight)

def references(root,seed):
    root=Path(root);four=verified(root/'recognition'/f'dev_FG2_s{seed}_p4.json')['records'];eight=verified(root/'recognition'/f'dev_FG2_s{seed}_p8.json')['records']
    return four,eight,dict(A8=summarize(eight)['A'],M8=summarize(eight)['M'],C8=summarize(eight)['C'],A4=summarize(four)['A'])

def controller_scores(root,seed,method,step):
    root=Path(root);train=verified(root/'training.json');arm=ARMS[1] if method=='NO_NEW_CORRESPONDENCE' else ARMS[0];receipt=train['arms'][f'{seed}:{arm}'];folder=root/'training'/f'seed{seed}'
    normal=verified(folder/'normalizer.json');norm=Normalizer(normal['mean'],normal['std']);features,_,groups,_,bankid=load_bank(root,'dev',seed)
    device='cuda' if torch.cuda.is_available() else 'cpu';model=UtilityRegressor(seed).to(device);row=receipt['checkpoints'][str(step)];ConsumptionIndex().identity(row['path'],row)
    data=torch.load(row['path'],map_location='cpu',weights_only=False)
    if data['normalizer']!=normal['identity'] or data['seed']!=seed or data['step']!=step:raise ValueError('Controller/normalizer identity differs')
    model.load_state_dict(data['model']);model.eval()
    with torch.no_grad():pred=model(torch.from_numpy(norm.transform(features,method=='NO_NEW_CORRESPONDENCE')).to(device)).cpu().numpy()
    return pred,groups

def policy_choices(units,seed,prefix,pred,groups,tau,coupled=False):
    choices={}
    for entry in units:
        r=verified(entry['path']);g=next(g for g in r['groups'] if g['seed']==seed and g['prefix']==prefix);actions=[r['rows'][i] for i in g['indices']];scores=[0.]+pred[groups[(r['object']['key'],r['condition'],prefix)]].tolist()
        choice=select_action(actions,scores,tau,coupled);choices[r['object']['key'],r['condition']]=g['indices'][choice]
    return choices

def evaluate_profile(root,seed,profile,score_cache=None,save=False):
    root=Path(root);bank=verified(root/'action_bank.json');units=[u for u in bank['units'] if u['role']=='dev'];ref4,ref8,reference=references(root,seed);method=profile['method'];pred=groups=None
    if method!='STATIC':
        key=(seed,'WITH_NEW_CORRESPONDENCE' if method=='COUPLED' else method,profile['step']);cache=score_cache if score_cache is not None else {}
        if key not in cache:cache[key]=controller_scores(root,seed,method,profile['step'])
        pred,groups=cache[key]
    records={};extras={};output={}
    for prefix in (2,4,8):
        static=f'{profile["base"]}:{profile["gamma"]:g}' if method=='STATIC' and prefix>2 else None
        choices=policy_choices(units,seed,prefix,pred,groups,profile['tau'],method=='COUPLED') if method!='STATIC' and prefix>2 else None
        rows,extra=recognition_records(units,prefix,seed,static=static,selected=choices,include_arrays=save);records[prefix]=rows;extras[prefix]=extra
        if save:
            path=root/'locked_readouts'/f'seed{seed}'/f'{method}_p{prefix}.json';arr=arrays_record(path.with_suffix('.npz'),dict(embeddings=extra['embeddings'],full200_cosines=extra['cosines'],object_keys=np.asarray([r['key'] for r in rows]),class_ids=np.asarray([r['class_id'] for r in rows])),ConsumptionIndex())
            saved=write(path,dict(status='COMPLETE',profile=profile,seed=seed,prefix=prefix,records=rows,metrics=summarize(rows,ref4 if prefix==4 else ref8 if prefix==8 else None),mean_edits=extra['mean_edits'],edit_types=extra['edit_types'],arrays=arr,prefix2_always_KEEP=True));output[str(prefix)]=dict(path=str(path),identity=saved['identity'])
    metrics=candidate_metrics(records[4],records[8],ref4,ref8,extras[4]['mean_edits'],extras[8]['mean_edits'],profile);metrics['passed']=promotion(metrics,reference);metrics['deltas']={k:metrics[k]-reference[k] for k in ('A8','M8','C8','A4')}
    return metrics,output

def select(root):
    root=Path(root);trained=verified(root/'training.json')['status']=='COMPLETE';candidates=[];cache={}
    for base in ('MEAN','MASK_AREA','COSINE_MEDOID'):
        for gamma in (0.,.5,1.):
            profile=dict(method='STATIC',base=base,gamma=gamma,step=0,tau=0.)
            metrics,_=evaluate_profile(root,17,profile,cache);candidates.append(metrics)
    if trained:
        for method in ('COUPLED','NO_NEW_CORRESPONDENCE','WITH_NEW_CORRESPONDENCE'):
            for step in (500,1000,1500,2000):
                for tau in (0.,.01,.03,.1):
                    metrics,_=evaluate_profile(root,17,dict(method=method,step=step,tau=tau),cache);candidates.append(metrics)
    expected=57 if trained else 9
    if len(candidates)!=expected:raise ValueError('Finite DEV cell count differs')
    selected={method:sorted([r for r in candidates if r['method']==method],key=cmp_to_key(compare_candidates))[0] for method in METHODS if any(r['method']==method for r in candidates)}
    controls=[r for method,r in selected.items() if method!='WITH_NEW_CORRESPONDENCE'];c_star=sorted(controls,key=cmp_to_key(compare_candidates))[0]['method'];lock=lock_profiles(root,selected,c_star);readouts={}
    for method,profile in selected.items():_,readouts[method]=evaluate_profile(root,17,profile,cache,save=True)
    return write(root/'selection.json',dict(status='COMPLETE',seed=17,candidates=candidates,logical_cells=expected,selected=selected,C_STAR=c_star,lock_identity=lock['identity'],locked_readouts=readouts,edit_fraction_definition='EQUAL_LOGICAL_DEV_PREFIX4_AND8_CONDITION_ROWS;STATIC_CONFIGURATION_HAS_ZERO_DISCRETE_EDITS',seed29_reselection=False))

def repeat(root):
    root=Path(root);selection=verified(root/'selection.json');lock=verified(root/'selection_lock.json');results={};readouts={};cache={}
    for method,profile in lock['profiles'].items():results[method],readouts[method]=evaluate_profile(root,29,profile,cache,save=True)
    passed={m:selection['selected'][m]['passed'] and results[m]['passed'] for m in results};c_star=lock['C_STAR']
    if passed.get('WITH_NEW_CORRESPONDENCE',False):activated=[m for m in METHODS if m in results];reason='WITH_NEW_CORRESPONDENCE_PASSED_BOTH_SEEDS_ALL_AVAILABLE_CONTROLS_INCLUDED'
    elif passed[c_star]:activated=[c_star];reason='LOCKED_STRONGEST_CONTROL_PASSED_BOTH_SEEDS'
    else:activated=[];reason='WITH_NEW_CORRESPONDENCE_OR_LOCKED_C_STAR_DID_NOT_PASS_BOTH_FIXED_SEED_GATES'
    activation=write(root/'activation.json',dict(status='ACTIVATED' if activated else 'NOT_ACTIVATED_BY_FIXED_RULE',methods=activated,C_STAR=c_star,passed_both=passed,reason=reason,selection_lock=lock['identity'],seed29_reselection=False))
    return write(root/'repeat.json',dict(status='COMPLETE',seed=29,locked_seed17_profiles=lock['profiles'],results=results,locked_readouts=readouts,activation=activation['identity'],seed29_reselection=False))
