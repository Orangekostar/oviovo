import pytest

from src.static_ovmap.native_recording import instrument_original_source


def test_recorder_inserts_only_two_observation_hooks():
    source = (
        '        glo_inst_map = gsm_node.raycastInstancePredictions(\n'
        '            pose, inst_seg, depth_scaled\n'
        '        )\n'
        "            inst_dict[glo_inst_id]['vis_area'].append(overlap_area)\n")
    transformed = instrument_original_source(source)
    assert transformed.count('_static_record_frame(') == 1
    assert transformed.count('_static_record_query(') == 1
    assert 'raycastInstancePredictions' in transformed
    assert '.append(overlap_area)' in transformed


def test_changed_native_interface_is_not_guessed():
    with pytest.raises(ValueError, match='hook'):
        instrument_original_source('different mapper version')
