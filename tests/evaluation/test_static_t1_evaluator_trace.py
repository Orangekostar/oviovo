from pathlib import Path
import os
import numpy as np
import pytest

from src.static_ovmap.released_loader import load_released_module


def evaluator_fixture():
    source = Path(os.environ.get('OVIMAP_EVALUATOR_ROOT', '/home/ww/vv/paper2/OVI-MAP'))/'scripts/eval_utils.py'
    if not source.exists():
        pytest.skip('receipt-bound released evaluator unavailable')
    evaluator = load_released_module(source)
    evaluator['init']('Replica')
    return evaluator


def prepare(evaluator, tmp_path, gt, masks, labels, scores):
    gt_path = tmp_path/'gt.npy'; np.save(gt_path, gt)
    lines = []
    for i,mask in enumerate(masks):
        path = tmp_path/f'p{i}.npy'; np.save(path, mask)
        lines.append(f'p{i}.npy {labels[i]} {scores[i]:.6f}')
    manifest = tmp_path/'pred.txt'; manifest.write_text('\n'.join(lines)+'\n')
    g,p = evaluator['assign_instances_for_scan'](str(tmp_path),str(manifest),str(gt_path))
    return {str(gt_path):{'gt':g,'pred':p}}


def test_trace_preserves_duplicate_score_events_ignored_masks_and_no_gt(tmp_path):
    from src.static_ovmap.released_trace import trace_released_matches
    e = evaluator_fixture(); a,b = e['VALID_CLASS_IDS'][:2]
    gt = np.r_[np.full(200,a*1000),np.zeros(200,dtype=int),np.full(50,b*1000)]
    masks=np.zeros((6,len(gt)),bool)
    masks[0,:200]=True; masks[1,:200]=True; masks[2,200:400]=True
    masks[3,:99]=True; masks[4,:200]=True; masks[5,350:450]=True
    matches=prepare(e,tmp_path,gt,masks,[a,a,a,a,0,b],[.6,.9,.8,.2,.7,.5])
    baseline=e['evaluate_matches'](matches)
    actual,trace=trace_released_matches(e,matches)
    assert np.array_equal(actual,baseline,equal_nan=True)
    assert trace['parity']['ap_exact'] and trace['parity']['pr_and_fn_exact']
    assert any(x['event']=='duplicate' for x in trace['events'])
    assert any(x['event']=='ignore_test' and not x['counted_fp'] for x in trace['events'])
    assert np.isnan(actual[0,1]).all()
    files={p['filename'] for label in matches[next(iter(matches))]['pred'].values() for p in label}
    assert str(tmp_path/'p3.npy') not in files and str(tmp_path/'p4.npy') not in files


def test_exact_released_half_threshold_is_strict_and_fn_is_recorded(tmp_path):
    from src.static_ovmap.released_trace import trace_released_matches
    e=evaluator_fixture(); a=e['VALID_CLASS_IDS'][0]
    gt=np.full(200,a*1000); masks=np.zeros((1,200),bool);masks[0,:100]=True
    matches=prepare(e,tmp_path,gt,masks,[a],[.5])
    ap,trace=trace_released_matches(e,matches)
    index=np.flatnonzero(e['overlaps']==.5)[0]
    assert ap[0,0,index]==0
    state=next(s for s in trace['states'] if s['class_index']==0 and s['overlap_index']==int(index))
    assert state['hard_false_negatives']==1
    assert any(x['event']=='hard_fn' and x['overlap_threshold']==.5 for x in trace['events'])
