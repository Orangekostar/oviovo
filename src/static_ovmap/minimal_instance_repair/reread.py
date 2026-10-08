"""Original-incumbent selection and the two prescribed ordinary-AnyUp decisions."""

import numpy as np

from static_ovmap.module_validation.native_capture import _array_digest


def select_incumbents(unit_set, probabilities, *, limit=16):
    selected = []
    for name,unit in unit_set.units.items():
        if unit.kind != 'I':
            continue
        values = probabilities[str(unit.owner)]['probabilities']
        if values is None:
            continue
        vector = np.asarray(values,np.float64)
        if vector.ndim != 1 or len(vector) < 2 or not np.isfinite(vector).all():
            raise ValueError('available D2 probabilities must be finite full-vocabulary vectors')
        ordered = np.sort(vector)
        selected.append({'unit':name,'owner':unit.owner,'support_hash':unit.support_hash,
                         'D2_probability_margin':float(ordered[-1]-ordered[-2]),
                         'old_class':int(probabilities[str(unit.owner)]['label'])})
    return sorted(selected,key=lambda r:(r['D2_probability_margin'],r['support_hash']))[:limit]


def qualified_mask(mask, *, minimum_pixels=100, minimum_extent=2):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError('recognition requires a full-resolution boolean observation mask')
    y,x = np.nonzero(mask)
    bbox = [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1] if len(x) else None
    return (len(x) >= minimum_pixels and bbox[2]-bbox[0] >= minimum_extent
            and bbox[3]-bbox[1] >= minimum_extent),bbox


def recognition_masks(full, panoptic, *, core_min_fraction=.25, intersection_min_fraction=.50,
                      minimum_pixels=100):
    import cv2
    full = np.asarray(full)
    qualified,bbox = qualified_mask(full,minimum_pixels=minimum_pixels)
    if not qualified:
        return [],[{'type':'FULL','reason':'NO_QUALIFIED_FULL_MASK'}]
    radius = int(np.clip(np.ceil(.01*np.sqrt(full.sum())),1,8))
    core = cv2.erode(full.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8),
                     borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    definitions = [('FULL',full)]
    missing = []
    if core.sum() >= minimum_pixels and core.sum() >= core_min_fraction*full.sum():
        definitions.append(('CORE',core))
    else:
        missing.append({'type':'CORE','reason':'INSUFFICIENT_CORE_SUPPORT'})
    if panoptic is None:
        missing.append({'type':'FRONTEND_INTERSECTION','reason':'BLOCKED_MISSING_FRONTEND_RASTER'})
    else:
        panoptic = np.asarray(panoptic)
        if panoptic.shape != full.shape or not np.issubdtype(panoptic.dtype,np.integer) or np.any(panoptic<0):
            raise ValueError('frontend intersection requires aligned integer pre-insertion IDs')
        ids,counts = np.unique(panoptic[full & (panoptic>0)],return_counts=True)
        dominant = int(ids[np.lexsort((ids,-counts))[0]]) if len(ids) else 0
        intersection = full & (panoptic == dominant) if dominant else np.zeros_like(full)
        if intersection.sum() >= minimum_pixels and intersection.sum() >= intersection_min_fraction*full.sum():
            definitions.append(('FRONTEND_INTERSECTION',intersection))
        else:
            missing.append({'type':'FRONTEND_INTERSECTION','reason':'INSUFFICIENT_FRONTEND_INTERSECTION'})
    result,seen = [],set()
    for kind,mask in definitions:
        digest = _array_digest(mask)
        if digest in seen:
            missing.append({'type':kind,'reason':'DUPLICATE_MASK','mask_digest':digest})
            continue
        seen.add(digest)
        result.append({'type':kind,'mask':mask,'mask_digest':digest,'pixels':int(mask.sum()),
                       'core_radius':radius if kind=='CORE' else None})
    return result,missing


def reread_decision(old_class, valid_ids, observations, *, margin=.01, minimum_regions=3,
                    minimum_extra=1, vote_fraction=2/3, full_margin=0., median_margin=.01):
    ids = list(map(int,valid_ids))
    if old_class not in ids:
        raise ValueError('old D2 label is outside the frozen category array')
    usable,missing,seen = [],[],set()
    # FULL takes precedence when a caller supplies duplicate region records.
    for item in sorted(observations,key=lambda r:(r['frame_id'],r['type']!='FULL')):
        values = item.get('scores')
        if values is None:
            missing.append({'frame_id':item['frame_id'],'type':item['type'],'reason':item.get('reason','REPRESENTATION_UNAVAILABLE')})
            continue
        vector = np.asarray(values,np.float64)
        if vector.shape != (len(ids),) or not np.isfinite(vector).all():
            raise ValueError('successful reread requires finite unchanged-vocabulary FC cosine scores')
        identity = (item['frame_id'],item['mask_digest'])
        if identity in seen:
            missing.append({'frame_id':item['frame_id'],'type':item['type'],'reason':'DUPLICATE_MASK'})
            continue
        seen.add(identity)
        usable.append(item)
    full = [row for row in usable if row['type']=='FULL']
    result = {'old_class':old_class,'IR06_class':old_class,'IR07_class':old_class,
              'successful_regions':len(usable),'successful_full_views':len(full),'missing':missing,
              'observations':usable,'tests':{},'proposed_class':None,'aggregate_margin':None}
    if not full:
        return {**result,'IR06_reason':'KEEP_NO_FULL_VIEW','IR07_reason':'KEEP_NO_FULL_VIEW'}
    average = np.mean([row['scores'] for row in full],axis=0)
    c,old = int(np.argmax(average)),ids.index(old_class)
    proposed,delta = ids[c],float(average[c]-average[old])
    tests = {'aggregate_margin':delta >= margin,
        'both_banks':{row['bank'] for row in full}=={'proposal','verification'} and len(full)==2,
        'minimum_distinct_regions':len(usable)>=minimum_regions,
        'minimum_extra_regions':sum(row['type']!='FULL' for row in usable)>=minimum_extra,
        'both_full_argmax_agree':all(int(np.argmax(row['scores']))==c for row in full),
        'both_full_margins_nonnegative':all(row['scores'][c]-row['scores'][old]>=full_margin for row in full),
        'region_vote_fraction':sum(int(np.argmax(row['scores']))==c for row in usable)/len(usable)>=vote_fraction,
        'median_region_margin':float(np.median([row['scores'][c]-row['scores'][old] for row in usable]))>=median_margin}
    simple = proposed != old_class and tests['aggregate_margin']
    stable = simple and all(tests.values())
    result.update(proposed_class=proposed,aggregate_scores=average.tolist(),aggregate_margin=delta,tests=tests,
        IR06_class=proposed if simple else old_class,IR07_class=proposed if stable else old_class,
        IR06_reason='RELABEL_ACCEPTED' if simple else ('KEEP_SAME_CLASS' if proposed==old_class else 'KEEP_AGGREGATE_MARGIN'),
        IR07_reason='RELABEL_ACCEPTED' if stable else ('KEEP_SAME_CLASS' if proposed==old_class else 'KEEP_STABILITY_REJECTED'),
        prevented_by=[key for key,passed in tests.items() if not passed])
    return result
