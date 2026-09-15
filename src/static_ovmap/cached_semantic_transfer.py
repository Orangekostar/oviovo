"""Prediction-only cached semantic suggestions; no encoder or evaluation imports."""
from fractions import Fraction
import hashlib
import json
import numpy as np

METHODS=('ST_A_NATIVE_TOP1','ST_B_SF_TRANSFER','ST_C_SF_SUPPORT_GATE')


def digest(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str(value.dtype).encode()+repr(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def payload_key(binding):
    return hashlib.sha256(json.dumps(binding,sort_keys=True,allow_nan=False).encode()).hexdigest()


def native_suggestions(targets,owners,kept,valid_ids):
    result={};raw={}
    for target in targets:
        if 'OD_E1_SEMANTIC' not in target['methods'] or target['candidate'] not in kept:continue
        i=target['candidate']
        if target['mask_hash']!=digest(owners==i+1):raise ValueError('E1 target mask mismatch')
        if i in raw:
            if raw[i]!=target:raise ValueError('inconsistent duplicate native target')
            continue
        raw[i]=target
        scores=np.asarray(target.get('aggregate_cosine',[]),dtype=float)
        views=np.asarray(target.get('per_view_cosine',[]),dtype=float)
        row={'prior_class_id':target.get('class_id'),'prior_gate_reason':target.get('reason'),
             'encoded_views':len(views),'proposed_label':None,'reason':'no_aggregate','tie':False,
             'score_record_digest':hashlib.sha256(json.dumps(target,sort_keys=True,allow_nan=True).encode()).hexdigest(),
             'aggregate_cosine':scores.tolist() if np.isfinite(scores).all() else None,
             'view_agreement':target.get('view_agreement'),'advantage':target.get('advantage')}
        if not target.get('budget_included',False):row['reason']='budget_excluded'
        elif not len(views):row['reason']='no_encoded_input'
        elif scores.shape!=(len(valid_ids),) or not np.isfinite(scores).all():row['reason']='invalid_aggregate'
        else:
            if views.shape!=(len(views),len(valid_ids)) or not np.isfinite(views).all():raise ValueError('invalid native view scores')
            best=int(np.argmax(scores));label=int(valid_ids[best])
            if target.get('proposed_class',label)!=label:raise ValueError('stored proposed_class disagrees with aggregate')
            row.update(proposed_label=label,reason='raw_top1',tie=bool(np.count_nonzero(scores==scores[best])>1))
        result[i]=row
    return result


def aligned_quality(records,t0):
    ids=np.asarray(t0['query_ids']);classes=np.asarray(t0['class_ids']);lookup={int(q):i for i,q in enumerate(ids)}
    if len(lookup)!=len(ids) or classes.shape!=ids.shape:raise ValueError('invalid query identity')
    quality=None
    if all(k in t0 for k in ['scores','objectness_mask_scores','class_probabilities']):
        quality=np.asarray(t0['scores'])
        components=np.asarray(t0['objectness_mask_scores'])*np.asarray(t0['class_probabilities'])
        if quality.shape!=ids.shape or not np.isfinite(quality).all() or np.any(quality<0):raise ValueError('invalid SF quality')
        if not np.allclose(quality,components,rtol=1e-5,atol=1e-7):raise ValueError('T0 product convention mismatch')
    result={}
    for record in records:
        if record['source']!='SpaCeFormer':continue
        row=lookup[record['spaceformer_query_id']]
        if row!=record['original_t0_row'] or int(classes[row])!=record['original_class_id']:raise ValueError('SF query/row/class misalignment')
        result[record['canonical_index']]=None if quality is None else float(quality[row])
    return result


def pair_intersections(owners,masks,receivers,donors,chunk=65536):
    table=np.zeros((len(receivers),len(donors)),np.int64)
    width=int(np.max(owners))+1
    for col,j in enumerate(donors):
        counts=np.zeros(width,np.int64)
        for start in range(0,len(owners),chunk):
            selected=owners[start:start+chunk][masks[j,start:start+chunk]]
            counts+=np.bincount(selected,minlength=width)
        table[:,col]=counts[np.asarray(receivers)+1]
    return table


def donor_choice(receiver_area,donors):
    eligible=[]
    for donor in donors:
        n,area=int(donor['intersection']),int(donor['area'])
        if not receiver_area or not area or 2*n<receiver_area or 2*n<area:continue
        union=receiver_area+area-n
        eligible.append({**donor,'receiver_coverage':n/receiver_area,'donor_coverage':n/area,'iou':n/union,'union':union})
    eligible.sort(key=lambda d:(-Fraction(d['intersection'],d['union']),d['query_id'],d['canonical_index']))
    if not eligible:return None,[],[]
    selected=eligible[0]
    ties=[d['canonical_index'] for d in eligible if d['intersection']*selected['union']==selected['intersection']*d['union']]
    return selected,eligible,ties


def support_gate(original,selected,eligible):
    if selected is None:return {'accepted':False,'reason':'no_donor','support':None,'groups':[]}
    groups={}
    for d in eligible:
        key=(d['mask_hash'],d['class_id'])
        if key not in groups:groups[key]={'members':[],'class_id':d['class_id'],'iou':d['iou'],'quality':d['quality']}
        group=groups[key];group['members'].append(d['canonical_index'])
        if d['quality'] is None:group['quality']=None
        elif group['quality'] is not None:group['quality']=max(group['quality'],d['quality'])
    groups=list(groups.values());result={'accepted':False,'reason':'quality_unavailable','support':None,'groups':groups}
    if selected['class_id']==original:result.update(accepted=True,reason='same_label');return result
    if any(g['quality'] is None for g in groups):return result
    total=sum(g['iou']*g['quality'] for g in groups)
    if total<=0:result['reason']='zero_weight';return result
    mass=sum(g['iou']*g['quality'] for g in groups if g['class_id']==selected['class_id'])
    support=mass/total;accepted=support>=2/3
    result.update(support=support,accepted=accepted,reason='support_pass' if accepted else 'support_reject')
    return result


def assert_fixed(baseline,doc):
    for key in ['kept','rank_scores','rank_scores_serialized','owner_path']:
        if doc[key]!=baseline[key]:raise ValueError('fixed OVI contract changed: '+key)
    outside=set(range(len(baseline['labels'])))-set(baseline['kept'])
    if any(doc['labels'][i]!=baseline['labels'][i] for i in outside):raise ValueError('donor output labels changed')
