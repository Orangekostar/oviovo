"""Finite weight selection consumes the fusion DEV lock only."""
from functools import cmp_to_key
import numpy as np

def select_weight(candidates):
    finite=[c for c in candidates if all(c['metrics'].get(k) is not None and np.isfinite(c['metrics'][k]) for k in ('apall','miou'))]
    def compare(a,b):
        for key in ('apall','miou'):
            difference=a['metrics'][key]-b['metrics'][key]
            if abs(difference)>1e-10:return -1 if difference>0 else 1
        def tie(row):
            w=row['weight']
            return (sum(v>0 for v in w),sum(abs(v-ref) for v,ref in zip(w,(.25,.25,.5),strict=True)),tuple(w))
        return (tie(a)>tie(b))-(tie(a)<tie(b))
    return sorted(finite,key=cmp_to_key(compare))[0] if finite else None

def no_dev_lock(reason='INCOMPLETE_FIXED_FOUR_SCENE_FUSION_DEV'):
    return dict(status='NOT_SELECTED_NO_FUSION_DEV',weight=None,reason=reason,selection_role='FUSION_DEV',regression_used=False)

def validate_prediction_lock(lock,scenes,methods):
    pairs=[(r['scene'],r['method']) for r in lock['records']]
    expected={(s,m) for s in scenes for m in methods}
    if lock['role']!='FUSION_DEV' or lock['status']!='ALL_PREDICTIONS_LOCKED' or lock['scenes']!=list(scenes) or lock['methods']!=list(methods) or len(pairs)!=len(set(pairs)) or set(pairs)!=expected:
        raise ValueError('Fusion DEV prediction lock is incomplete or role differs')

def freeze_weights(root):
    from pathlib import Path
    from .common import verified,write,immutable_record,unit_identity
    from .binding import load_binding
    root=Path(root);binding=load_binding(root);spec=binding['task_specification']
    prediction=verified(root/'fusion_dev_predict.json');score=verified(root/'fusion_dev_score.json')
    if score['status']=='NOT_SELECTED_NO_FUSION_DEV':
        lock=no_dev_lock(score['reason']);lock['input_identity']=unit_identity('freeze',dict(prediction=prediction['identity'],score=score['identity']),producers=('common','selection'))
        write(root/'weight_selection.json',dict(lock,candidates=[]))
        return immutable_record(root/'weight_lock.json',lock)
    methods=[f'W{i:02d}' for i in range(15)]
    validate_prediction_lock(prediction,spec['fusion_dev'],methods)
    if score['status']!='COMPLETE' or score['logical_pools']!=15:raise ValueError('All15 fixed DEV pools must precede weight selection')
    candidates=[dict(method=m,weight=list(w),metrics=score['pools']['fusion_dev'][m]['metrics'],pool_identity=score['pools']['fusion_dev'][m]['identity']) for m,w in zip(methods,spec['weights']['grid'],strict=True)]
    winner=select_weight(candidates)
    key=unit_identity('freeze',dict(prediction=prediction['identity'],score=score['identity']),producers=('common','selection'))
    lock=dict(status='COMPLETE' if winner else 'NOT_SELECTED_NO_FUSION_DEV',input_identity=key,weight=winner['weight'] if winner else None,method=winner['method'] if winner else None,reason=None if winner else 'ALL_DEV_AP_OR_MIOU_UNDEFINED',selection_role='FUSION_DEV',regression_used=False,complete_candidate_count=15,ordered_scenes=spec['fusion_dev'],prediction_lock=prediction['identity'],score_identity=score['identity'],selection_order=spec['weights']['selection_order'],tolerance=1e-10)
    write(root/'weight_selection.json',dict(lock,candidates=candidates))
    return immutable_record(root/'weight_lock.json',lock)
