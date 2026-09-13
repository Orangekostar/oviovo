"""Evaluation-only exhaustive regional confusion and ownership transitions."""
import numpy as np


def confusion_metrics(confusion):
    c=np.asarray(confusion,dtype=np.int64)
    support=c.sum(axis=1); union=support+c.sum(axis=0)-np.diag(c)
    supported=np.flatnonzero(support[1:])+1
    return {'semantic_miou':float(np.mean(np.diag(c)[supported]/union[supported])) if len(supported) else None,
            'semantic_macc':float(np.mean(np.diag(c)[supported]/support[supported])) if len(supported) else None}


def regional_confusions(gt, prediction, regions, valid_ids):
    gt,prediction,regions=map(np.asarray,(gt,prediction,regions))
    ids=np.r_[0,np.asarray(valid_ids,dtype=np.int64)]
    if gt.ndim!=1 or gt.shape!=prediction.shape or gt.shape!=regions.shape or len(np.unique(ids))!=len(ids):
        raise ValueError('aligned evaluation arrays and unique nonzero valid IDs required')
    if not np.isin(prediction,ids).all():
        raise ValueError('prediction outside declared vocabulary')
    valid=np.isin(gt,ids[1:]); lookup={int(v):i for i,v in enumerate(ids)}
    # Vocabulary is small; explicit lookup supports non-contiguous IDs without relabeling metrics.
    gt_ord=np.zeros(len(gt),dtype=np.int64);pred_ord=np.zeros(len(gt),dtype=np.int64)
    for value,index in lookup.items():
        gt_ord[gt==value]=index;pred_ord[prediction==value]=index
    registry, region_ord=np.unique(regions,return_inverse=True)
    width=len(ids)
    encoded=(region_ord[valid]*width+gt_ord[valid])*width+pred_ord[valid]
    cubes=np.bincount(encoded,minlength=len(registry)*width*width).reshape(len(registry),width,width)
    global_c=cubes.sum(axis=0,dtype=np.int64)
    return {'confusion_ids':ids.tolist(),'global_confusion':global_c.tolist(),
        'metrics':confusion_metrics(global_c), 'valid_gt_vertices':int(valid.sum()),
        'regions':{str(int(code)):{'confusion':cubes[i].tolist(),'valid_gt_vertices':int(cubes[i].sum()),
                    'metrics':confusion_metrics(cubes[i])} for i,code in enumerate(registry)}}


def region_transitions(gt,before,after,owner_before,owner_after,regions,valid_ids):
    arrays=list(map(np.asarray,(gt,before,after,owner_before,owner_after,regions)))
    gt,before,after,owner_before,owner_after,regions=arrays
    if any(x.shape!=gt.shape for x in arrays) or gt.ndim!=1:
        raise ValueError('aligned transition arrays required')
    valid=np.isin(gt,valid_ids);bc=before==gt;ac=after==gt
    result={}
    for code in np.unique(regions):
        mask=valid&(regions==code)
        result[str(int(code))]={'valid_gt_vertices':int(mask.sum()),
            'correct_to_wrong':int((mask&bc&~ac).sum()),'wrong_to_correct':int((mask&~bc&ac).sum()),
            'wrong_to_wrong':int((mask&~bc&~ac).sum()),'correct_to_correct':int((mask&bc&ac).sum()),
            'owner_changes':int((mask&(owner_before!=owner_after)).sum()),
            'class_changes':int((mask&(before!=after)).sum()),
            'same_class_owner_changes':int((mask&(before==after)&(owner_before!=owner_after)).sum())}
    return result
