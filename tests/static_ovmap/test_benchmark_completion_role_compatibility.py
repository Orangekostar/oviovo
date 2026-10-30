"""Retained baseline evidence is compatible only with an identical AP scope."""
from copy import deepcopy

import pytest

from test_benchmark_completion_open3dis_replica_import import candidate, fixture
from static_ovmap.benchmark_completion_v2.common import digest_file, source_identity, verified, write


@pytest.mark.parametrize('changed', ['roles_only', 'minimum_region', 'official_source'])
def test_completed_baseline_keeps_its_original_certificate_across_only_a_role_extension(tmp_path, changed):
    binding, _, expected = fixture(tmp_path)
    path = tmp_path / 'protocol_lock.json'
    old = verified(path)
    archived = tmp_path / 'history/protocol_lock.json'
    write(archived, old)
    new = deepcopy(old)
    new['scopes']['replica48']['mapping'] = dict(base_mask=[True] * 48)
    new['scopes']['replica48']['full_AC_available'] = True
    if changed == 'minimum_region':
        new['scopes']['replica48']['minimum_region_vertices'] = 999
    elif changed == 'official_source':
        new['official_sources'] = dict(evaluator_replica=dict(sha256='changed'))
    new['semantic_refresh'] = dict(producer=source_identity(('semantic_refresh',)),
        previous_protocol=dict(path=str(archived), identity=old['identity'], sha256=digest_file(archived)))
    write(path, new)
    before = {p: (digest_file(p), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    if changed == 'roles_only':
        assert candidate().pool(binding) == expected
    else:
        with pytest.raises(ValueError, match='scope|source|protocol'):
            candidate().pool(binding)
    assert before == {p: (digest_file(p), p.stat().st_mtime_ns) for p in before}
