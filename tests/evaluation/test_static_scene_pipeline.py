import hashlib
import json

import pytest

from scripts.evaluation.run_static_replica_scene import verify_native_manifest


def test_only_complete_exact_frame_native_results_enter_evaluation(tmp_path):
    f = tmp_path/'asset'; f.write_bytes(b'bound native result')
    artifact = {'path': str(f), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
    x = {'status': 'PASS', 'scene': 'office0', 'dataset': 'replica',
         'frame_ids': list(range(0, 2000, 10)), 'audit': {'status': 'PASS', 'frame_count': 200},
         'artifacts': {k: artifact for k in ('instance_mesh', 'semantic_features', 'instance_color_log')}}
    p = tmp_path/'manifest.json'; p.write_text(json.dumps(x))
    assert verify_native_manifest(p)['scene'] == 'office0'
    x['frame_ids'][-1] = 2000; p.write_text(json.dumps(x))
    with pytest.raises(ValueError, match='frame protocol'):
        verify_native_manifest(p)
