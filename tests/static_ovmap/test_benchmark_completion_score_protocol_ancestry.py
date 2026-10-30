"""A semantic extension may reuse a numerically compatible frozen prediction."""
from copy import deepcopy

import numpy as np
import pytest

from static_ovmap.benchmark_completion_v2.common import (
    digest_file, save_arrays, source_identity, write,
)


@pytest.mark.parametrize('variant,changed_role,allowed', [
    ('U00', True, True), ('U01', False, True), ('U01', True, False),
])
def test_full_score_export_checks_protocol_ancestry_and_only_the_roles_it_consumes(
        tmp_path, variant, changed_role, allowed):
    from static_ovmap.benchmark_completion_v2.score_artifacts import prepare
    from static_ovmap.benchmark_completion_v2.standard_export import save_manifest, score_F
    scope = dict(text_ids=[1, 2], text_names=['a', 'b'], scenes=['room0'],
                 mapping=dict(base_mask=[True, False]), full_AC_available=True)
    old = write(tmp_path / 'history/old_protocol.json', dict(scopes=dict(replica48=scope)))
    newer = deepcopy(old)
    if changed_role:
        newer['scopes']['replica48']['mapping']['base_mask'] = [False, True]
    newer['semantic_refresh'] = dict(producer=source_identity(('semantic_refresh',)),
        previous_protocol=dict(path=str(tmp_path / 'history/old_protocol.json'),
                               identity=old['identity'], sha256=digest_file(tmp_path / 'history/old_protocol.json')))
    write(tmp_path / 'protocol_lock.json', newer)
    arrays = tmp_path / 'F.npz'
    text = np.eye(2, 768)
    save_arrays(arrays, owners=[7], valid_ids=[1, 2], available=[True],
        text=text, b=text[:1], z=np.tile(text[1:2][None], (2, 1, 1)))
    f = write(tmp_path / 'benchmark/F/replica48/room0/receipt.json', dict(bundle='bundle',
        arrays=dict(path=str(arrays), sha256=digest_file(arrays))))
    raw, calibrated = score_F(text[0], text[1], text,
                              scope['mapping']['base_mask'], calibrated=variant == 'U01')
    label = int(np.argmax(calibrated if variant == 'U01' else raw)) + 1
    folder = tmp_path / 'benchmark/predictions/replica48/room0'
    prediction = save_manifest(folder / variant, scene='room0', benchmark='replica48', method=variant,
        seed=17, owners=[7], indptr=[0, 1], indices=[0], labels=[label], target_points=1,
        valid_instance_ids=[1, 2], provenance=dict(F=f['identity'], protocol=old['identity'], bundle='bundle', NQ=None))
    write(folder / 'receipt.json', dict(GLOBAL=True, benchmark='replica48', scene='room0', bundle='bundle',
        methods={variant + '_seed17': dict(path=str(folder / variant / 'manifest.json'), identity=prediction['identity'])}))
    binding = dict(output_root=str(tmp_path))
    replay = dict(standard=dict(replica48=dict(temperatures=dict(F=1., N=1., Q=1.))))
    if allowed:
        result = prepare(binding, replay)
        assert result['completed_scene_cells'] == 1
    else:
        with pytest.raises(ValueError, match='scope|role'):
            prepare(binding, replay)
