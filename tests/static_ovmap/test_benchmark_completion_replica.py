import numpy as np
import pytest


def test_replica_geometry_requires_author_instance_tuple_and_preserves_point_order(tmp_path):
    import torch
    from plyfile import PlyData
    from static_ovmap.benchmark_completion_v2.replica_io import prepare_geometry
    from static_ovmap.benchmark_completion_v2.common import StageFailure
    gt=tmp_path/'author'
    gt.mkdir()
    xyz=np.array([[2,0,1],[1,4,0],[0,-2,3]],np.float32)
    binding=dict(output_root=str(tmp_path/'out'),asset_config=dict(datasets=dict(replica48=dict(gt_root=str(gt)))))
    torch.save((xyz,np.zeros_like(xyz),np.ones(3,dtype=np.int64)),gt/'room0.pth')
    with pytest.raises(StageFailure,match='four-field'):
        prepare_geometry(binding,'room0')
    torch.save((xyz,np.zeros_like(xyz),np.ones(3,dtype=np.int64),np.arange(3)),gt/'room0.pth')
    path=prepare_geometry(binding,'room0')
    vertices=PlyData.read(str(path))['vertex']
    actual=np.column_stack([vertices[k] for k in ('x','y','z')])
    assert np.array_equal(actual,xyz)


def test_replica_gt_export_cannot_read_labels_before_prediction_freeze(tmp_path):
    from static_ovmap.benchmark_completion_v2.replica_io import export_gt
    with pytest.raises(ValueError,match='prediction freeze'):
        export_gt(dict(output_root=str(tmp_path),asset_config={}), 'room0', {}, tmp_path/'gt')


def test_replica_author_agreement_requires_actual_frozen_predictions(tmp_path):
    from static_ovmap.benchmark_completion_v2.replica_validation import verify_scene
    from static_ovmap.benchmark_completion_v2.common import StageFailure
    with pytest.raises(StageFailure,match='frozen predictions'):
        verify_scene(dict(output_root=str(tmp_path),asset_config={}), 'room0')
