"""Geometry-only single-sweep complete-object arbitration against shared 2D masks."""
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment



def mask_areas(masks, chunk=65536):
    counts=np.zeros(len(masks),np.int64)
    for start in range(0,masks.shape[1],chunk):
        counts+=np.count_nonzero(masks[:,start:start+chunk],axis=1)
    return counts


def intersection_count(a,b,chunk=65536):
    return sum(int(np.count_nonzero(a[start:start+chunk]&b[start:start+chunk]))
               for start in range(0,len(a),chunk))


def union_masks(masks,indices,chunk=65536):
    union=np.zeros(masks.shape[1],bool)
    for start in range(0,masks.shape[1],chunk):
        union[start:start+chunk]=np.any(masks[indices,start:start+chunk],axis=0)
    return union


def group_agreement(predicted, observed, min_pixels=32):
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    valid = observed > 0  # archived samples are positive-mask evidence, not all visibility
    if np.count_nonzero(valid) < min_pixels:
        return None
    p, o = predicted[valid], observed[valid]
    pids, oi = np.unique(o, return_inverse=True)
    active = p > 0
    if not active.any():
        return 0.
    ids, pi = np.unique(p[active], return_inverse=True)
    intersections = np.zeros((len(ids),len(pids)),np.int64)
    np.add.at(intersections,(pi,oi[active]),1)
    # Rectangular maximum matching equals dummy-zero augmented matching.
    row,col = linear_sum_assignment(-intersections)
    return float(2*intersections[row,col].sum()/(active.sum()+len(o)))


def propose_actions(masks, ovi_ids, sf_ids, config):
    areas = mask_areas(masks)
    qualified = union_masks(masks,ovi_ids)
    actions=[]
    counts={'unresolved_sf_parent_relations':0,'child_shortlist_excluded':0,
            'group_overlap_rejected':0,'group_union_rejected':0,'zero_area_sf':0}
    for parent in ovi_ids:
        child_pool=[]
        for child in sf_ids:
            if not areas[child] or not areas[parent]:
                continue
            intersection=intersection_count(masks[parent],masks[child])
            inside,coverage=intersection/areas[child],intersection/areas[parent]
            if min(inside,coverage)>=config['replace_bidirectional_coverage_min']:
                actions.append({'type':'REPLACE_1_TO_1','parent':int(parent),
                                'children':[int(child)],'strength':min(inside,coverage)})
            elif inside<config['child_inside_parent_min']:
                counts['unresolved_sf_parent_relations']+=1
            if inside>=config['child_inside_parent_min']:
                child_pool.append((coverage,child))
        child_pool.sort(key=lambda x:(-x[0],x[1]))
        limit=config['shortlist_children_per_parent_max']
        counts['child_shortlist_excluded']+=max(0,len(child_pool)-limit)
        for size in config['children_per_parent']:
            for children in combinations(sorted(c for _,c in child_pool[:limit]),size):
                if any(intersection_count(masks[a],masks[b])/min(areas[a],areas[b])>
                       config['child_pair_overlap_over_smaller_max'] for a,b in combinations(children,2)):
                    counts['group_overlap_rejected']+=1
                    continue
                union=union_masks(masks,list(children))
                coverage=float(intersection_count(union,masks[parent])/areas[parent])
                if coverage<config['child_union_parent_coverage_min']:
                    counts['group_union_rejected']+=1
                    continue
                actions.append({'type':'REPLACE_1_TO_GROUP','parent':int(parent),
                                'children':list(map(int,children)),'strength':coverage})
    for child in sf_ids:
        if not areas[child]:
            counts['zero_area_sf']+=1
            continue
        outside=float(intersection_count(masks[child],~qualified)/areas[child])
        if outside>=config['add_outside_qualified_ovi_fraction_min']:
            actions.append({'type':'ADD_UNCOVERED_OBJECT','parent':None,
                            'children':[int(child)],'strength':outside})
    actions.sort(key=lambda a:(-a['strength'],a['parent'] if a['parent'] is not None else len(masks),
                               tuple(a['children']),a['type']))
    cap=config['shortlisted_actions_per_run_max']
    counts.update(potential_actions=len(actions),budget_excluded=max(0,len(actions)-cap))
    for i,action in enumerate(actions):
        action['action_id']=f'action_{i:04d}'
    counts['excluded_actions']=actions[cap:]
    return actions[:cap],counts


def action_footprint(masks, action):
    footprint=union_masks(masks,action['children'])
    if action['parent'] is not None:
        footprint |= masks[action['parent']]
    return footprint


def tentative_action(before, masks, action, sf_quality, retention_min):
    parent=action['parent']
    protected=(before>0) & (before != (parent+1 if parent is not None else -1))
    after=before.copy()
    if parent is not None:
        after[after==parent+1]=0
    fractions={}
    for child in sorted(action['children'],key=lambda c:(-sf_quality[c],c)):
        available=masks[child]&~protected&(after==0)
        fraction=float(np.count_nonzero(available)/max(1,np.count_nonzero(masks[child])))
        fractions[str(child)]=fraction
        after[available]=child+1
    if any(f<retention_min for f in fractions.values()):
        return None,'support_retention',fractions
    if not np.array_equal(after[protected],before[protected]):
        raise AssertionError('unrelated OVI modified')
    return after,'valid',fractions


def arbitrate(masks, before, ovi_ids, sf_ids, sf_quality, frames, config):
    actions,counts=propose_actions(masks,ovi_ids,sf_ids,config)
    accepted=[]
    for action in actions:
        after,reason,retention=tentative_action(before,masks,action,sf_quality,
                          config['activation_original_support_retention_min'])
        action.update(reason=reason,retention=retention,frames=[],accepted=False)
        if after is None:
            continue
        footprint=action_footprint(masks,action)
        deltas=[]
        for frame in frames:
            pts=frame['point_ids']
            inside=footprint[pts]
            obs=frame['mask_ids'][inside]
            # Shared zbuffer has exactly one source sample per recorded pixel.
            pb,pa=before[pts[inside]],after[pts[inside]]
            qb=group_agreement(pb,obs,config['usable_region_pixels_min'])
            qa=group_agreement(pa,obs,config['usable_region_pixels_min'])
            action['frames'].append({'frame_id':frame['frame_id'],'region_pixels':int(len(obs)),
                       'before_coverage':int(np.count_nonzero(pb)), 'after_coverage':int(np.count_nonzero(pa)),
                       'before_Q':qb,'after_Q':qa})
            if qb is not None and qa is not None:
                deltas.append(qa-qb)
        gain=float(np.mean(deltas)) if deltas else 0.
        fraction=float(np.mean(np.asarray(deltas)>0)) if deltas else 0.
        action.update(mean_gain=gain,positive_fraction=fraction,usable_views=len(deltas))
        if len(deltas)<config['usable_group_views_min']:
            action['reason']='insufficient_views'
        elif gain<config['mean_group_agreement_gain_min']:
            action['reason']='insufficient_group_gain'
        elif fraction<config['positive_group_view_fraction_min']:
            action['reason']='insufficient_positive_views'
        else:
            action['reason']='eligible'
            accepted.append(action)
    final=before.copy()
    used=np.zeros(len(before),bool)
    used_ids=set()
    active=set(map(int,ovi_ids))
    for action in sorted(accepted,key=lambda a:(-a['mean_gain'],a['parent'] if a['parent'] is not None else len(masks),tuple(a['children']))):
        footprint=action_footprint(masks,action)
        ids=set(action['children']) | ({action['parent']} if action['parent'] is not None else set())
        if used_ids&ids or np.any(used&footprint):
            action['reason']='conflicting_action'
            continue
        after,_,_=tentative_action(before,masks,action,sf_quality,config['activation_original_support_retention_min'])
        final[footprint]=after[footprint]
        used |= footprint
        used_ids |= ids
        active.update(action['children'])
        if action['parent'] is not None:
            active.remove(action['parent'])
        action.update(accepted=True,reason='committed')
    validate_final(final,masks,sorted(active))
    return final,sorted(active),actions,counts


def validate_final(owners,masks,active):
    if len(owners)!=masks.shape[1] or np.any(owners<0):
        raise ValueError('source geometry domain changed')
    if not set(np.unique(owners)).issubset({0}|{i+1 for i in active}):
        raise ValueError('inactive owner leaked')
    for i in active:
        if np.any((owners==i+1)&~masks[i]):
            raise ValueError('owner escapes original source support')


def query_quality(records, query_ids, scores):
    """Map stable SF query IDs to original T0 rows; never use canonical row as query row."""
    lookup={int(q):i for i,q in enumerate(query_ids)}
    if len(lookup)!=len(query_ids) or len(scores)!=len(query_ids):
        raise ValueError('duplicate/misaligned SF query IDs')
    result={}
    for record in records:
        if record['source']!='SpaCeFormer':continue
        row=lookup[record['spaceformer_query_id']]
        if row!=record['original_t0_row']:
            raise ValueError('source query/row mapping changed')
        result[record['canonical_index']]=float(scores[row])
    return result
