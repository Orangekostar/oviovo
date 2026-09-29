import json

import pytest

from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.paired_evidence_study.path_map import apply_path_map


def test_explicit_relocation_keeps_hash_verification_and_never_overwrites(tmp_path):
    target, original = tmp_path / 'moved', tmp_path / 'old'
    target.mkdir()
    data = target / 'evidence.json'
    data.write_text('{"score":0.1}')
    expected = InputIndex().identity(data)
    mapping = tmp_path / 'mapping.json'
    mapping.write_text(json.dumps({str(original): str(target)}))
    assert apply_path_map(mapping)[0]['alias_created']
    assert InputIndex().identity(original / data.name, expected)['sha256'] == expected['sha256']
    data.write_text('{"score":0.9}')
    with pytest.raises(ValueError):
        InputIndex().identity(original / data.name, expected)
    occupied = tmp_path / 'occupied'
    occupied.mkdir()
    mapping.write_text(json.dumps({str(occupied): str(target)}))
    with pytest.raises(ValueError):
        apply_path_map(mapping)
    assert not occupied.is_symlink()
