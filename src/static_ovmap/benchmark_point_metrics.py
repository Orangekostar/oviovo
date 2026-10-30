"""GT-blind instance-to-point projection and full-denominator semantic metrics.

Use the common AP object-class vocabulary, original vertex order and frozen
confidence. Overlaps use greatest confidence then earliest original mask column;
invalid-class masks are excluded as in AP. No GT enters projection. The unknown
column retains all uncovered/unsupported valid GT points as false negatives.
"""
import numpy as np


def _ids(values):
    result=np.asarray(values)
    if (result.ndim!=1 or not np.issubdtype(result.dtype,np.integer)
            or not len(result) or np.any(result<=0) or len(np.unique(result))!=len(result)):
        raise ValueError('Unique positive integer class IDs are required')
    return result.astype(np.int64,copy=False)


def project_masks(*,indptr,indices,labels,confidence,target_points,class_ids):
    ids=_ids(class_ids)
    ptr,idx,lab=map(np.asarray,(indptr,indices,labels));conf=np.asarray(confidence)
    if (type(target_points) is not int or target_points<0 or ptr.ndim!=1 or idx.ndim!=1
            or lab.ndim!=1 or conf.shape!=lab.shape or len(ptr)!=len(lab)+1
            or not all(np.issubdtype(x.dtype,np.integer) for x in (ptr,idx,lab))
            or ptr[0]!=0 or ptr[-1]!=len(idx) or np.any(np.diff(ptr)<0)
            or np.any(idx<0) or np.any(idx>=target_points) or not np.isfinite(conf).all()):
        raise ValueError('Malformed CSR mask, class, confidence or point domain')
    active=np.flatnonzero(np.isin(lab,ids))
    order=active[np.argsort(-conf[active],kind='stable')]
    output=np.zeros(target_points,np.int64)
    for position in order:
        points=idx[ptr[position]:ptr[position+1]]
        output[points[output[points]==0]]=lab[position]
    return output


def confusion_counts(gt,prediction,class_ids):
    ids=_ids(class_ids);gt,pred=map(np.asarray,(gt,prediction))
    if (gt.ndim!=1 or pred.shape!=gt.shape or
            not all(np.issubdtype(x.dtype,np.integer) for x in (gt,pred))):
        raise ValueError('Semantic labels must share the complete integer point axis')
    # IDs are the small fixed official semantic taxonomy, not instance IDs.
    lookup=np.zeros(int(ids.max())+1,np.int64);lookup[ids]=np.arange(1,len(ids)+1)
    mapped=[]
    for values in (gt,pred):
        valid=(values>=0)&(values<len(lookup));axis=np.zeros(len(values),np.int64)
        axis[valid]=lookup[values[valid]];mapped.append(axis)
    g,p=mapped;keep=g!=0;width=len(ids)+1
    return np.bincount(g[keep]*width+p[keep],minlength=width*width).reshape(width,width)


def summarize(matrix,class_ids):
    ids=_ids(class_ids);matrix=np.asarray(matrix)
    if (matrix.shape!=(len(ids)+1,)*2 or not np.issubdtype(matrix.dtype,np.integer)
            or np.any(matrix<0) or np.any(matrix[0])):
        raise ValueError('Full-point confusion must retain an unknown column and no void GT row')
    gt=matrix.sum(axis=1)[1:];pred=matrix.sum(axis=0)[1:];tp=matrix.diagonal()[1:]
    present=gt>0;union=gt+pred-tp;count=int(gt.sum());correct=int(tp.sum())
    rows=[dict(class_id=int(label),gt_points=int(g),predicted_points=int(p),true_positive_points=int(t),
        iou=float(t/u) if g else None,accuracy=float(t/g) if g else None)
        for label,g,p,t,u in zip(ids,gt,pred,tp,union,strict=True)]
    return dict(miou=float(np.mean(tp[present]/union[present])) if present.any() else None,
        macc=float(np.mean(tp[present]/gt[present])) if present.any() else None,
        oacc=correct/count if count else None,defined_classes=int(present.sum()),
        valid_gt_points=count,correct_points=correct,unknown_prediction_points=int(matrix[:,0].sum()),
        per_class=rows,metric_unit='FRACTION',macro_classes='GT_PRESENT_CLASSES_AFTER_DATASET_CONFUSION_POOLING')
