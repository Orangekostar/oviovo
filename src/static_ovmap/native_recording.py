"""Instrumentation for the historical cached-feature mapper, without method edits."""


def instrument_original_source(source):
    hooks = {
        '        glo_inst_map = gsm_node.raycastInstancePredictions(\n'
        '            pose, inst_seg, depth_scaled\n'
        '        )':
        '\n        _static_record_frame(f_i, glo_inst_map)',
        "            inst_dict[glo_inst_id]['vis_area'].append(overlap_area)":
        '\n            _static_record_query(glo_inst_id, f_i, overlap_area, '
        '(x1, y1, x2, y2), roi_feat, pose, pano_id, glo_inst_mask, pano_mask)',
    }
    for needle, addition in hooks.items():
        if source.count(needle) != 1:
            raise ValueError('original mapper hook is missing or ambiguous')
        source = source.replace(needle, needle + addition)
    return source
