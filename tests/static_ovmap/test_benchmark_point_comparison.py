"""Real file scoring preserves full GT and rejects changed cached inputs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from static_ovmap.benchmark_completion_v2.common import write,digest_file


def runner():
    script=Path(__file__).resolve().parents[2]/'scripts/evaluation/compute_ovmap_point_metrics.py'
    spec=importlib.util.spec_from_file_location('point_comparison',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    m=runner();arrays=tmp_path/'masks.npz'
    np.savez(arrays,indptr=np.array([0,1]),indices=np.array([0]),labels=np.array([10]),confidence=np.array([1.]))
    manifest=tmp_path/'manifest.json';record=write(manifest,dict(status='DONE',arrays=m.descriptor(arrays)))
    gt=tmp_path/'GT.txt';np.savetxt(gt,[10001,20001,10002],fmt='%d')
    gt_receipt=tmp_path/'GT.json';g=write(gt_receipt,dict(status='DONE',GT=m.descriptor(gt)))
    job=dict(benchmark='fixture',scene='scene0',class_ids=[10,20],target_points=3,
        GT=m.descriptor(gt),GT_receipt=m.descriptor(gt_receipt),GT_identity=g['identity'],
        predictions=dict(Method=dict(kind='CSR',identity=record['identity'],manifest=m.descriptor(manifest),arrays=m.descriptor(arrays))))
    return m,job


def test_actual_file_scoring_keeps_uncovered_errors_and_resume_has_no_writes(tmp_path):
    m,job=fixture(tmp_path);args=(str(tmp_path/'out'),job,'fixed_source')
    result=m.score_scene(args);x=result['rows']['Method']
    assert x['confusion']==[[0,0,0],[1,1,0],[1,0,0]]
    assert x['metrics']['valid_gt_points']==3 and x['metrics']['miou']==.25 and x['metrics']['macc']==.25
    path=tmp_path/'out/scenes/fixture/scene0.json';before=(digest_file(path),path.stat().st_mtime_ns)
    assert m.score_scene(args)==result
    assert (digest_file(path),path.stat().st_mtime_ns)==before


@pytest.mark.parametrize('source',['GT','arrays'])
def test_cached_score_refuses_changed_original_bytes(tmp_path,source):
    m,job=fixture(tmp_path);args=(str(tmp_path/'out'),job,'fixed_source');m.score_scene(args)
    descriptor=job['GT'] if source=='GT' else job['predictions']['Method']['arrays']
    with Path(descriptor['path']).open('ab') as stream:stream.write(b'changed')
    with pytest.raises(AssertionError):m.score_scene(args)


def test_ap_alias_requires_same_frozen_prediction_and_arrays(tmp_path):
    m,job=fixture(tmp_path);arrays=job['predictions']['Method']['arrays']
    original=write(tmp_path/'original.json',dict(prediction_identity='same',arrays=arrays))
    alias=dict(original,identity='different_manifest',method='seed29')
    score=dict(prediction=original['identity'],prediction_identity='same',alias_of='seed17')
    lock=dict(methods=dict(seed17=dict(path=str(tmp_path/'original.json'),identity=original['identity'])))
    m.validate_standard_ap(score,alias,lock)
    with pytest.raises(AssertionError):m.validate_standard_ap(dict(score,prediction='unrelated'),alias,lock)
    with pytest.raises(AssertionError):m.validate_standard_ap(score,dict(alias,prediction_identity='changed'),lock)
    changed=tmp_path/'changed.npz';np.savez(changed,indptr=[0,1],indices=[1],labels=[10],confidence=[1.])
    with pytest.raises(AssertionError):m.validate_standard_ap(score,dict(alias,arrays=m.descriptor(changed)),lock)
