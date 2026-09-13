"""Separate single-semantic-query fallback using independent geometric evidence."""
from .readout import classify


DEFAULT_THRESHOLDS = {'min_geometry_frames': 2, 'min_visible_area_px': 500,
                      'min_depth_valid_ratio': .9, 'min_sharpness_proxy': .5}


def fallback_decision(observations, geometry, thresholds=None):
    config = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    frames = geometry.get('independent_frame_ids', [])
    result = {'accepted': False, 'semantic_query_count': len(observations),
              'geometry_support_frames': len(set(frames)), 'geometry_frame_ids': frames,
              'thresholds': config}
    if len(observations) != 1:
        result['reason'] = 'not_single_semantic_query'
        return result
    obs = observations[0]
    result.update({'visible_area_px': obs.visible_area_px, 'depth_valid_ratio': obs.depth_valid_ratio,
                   'sharpness_proxy': obs.sharpness_proxy, 'source_query_id': obs.source_query_id})
    if len(set(frames)) < config['min_geometry_frames']:
        result['reason'] = 'insufficient_independent_geometry'
    elif obs.depth_valid_ratio is None or obs.sharpness_proxy is None:
        result['reason'] = 'missing_measured_quality'
    elif (obs.visible_area_px < config['min_visible_area_px']
          or obs.depth_valid_ratio < config['min_depth_valid_ratio']
          or obs.sharpness_proxy < config['min_sharpness_proxy']):
        result['reason'] = 'quality_below_threshold'
    else:
        result['accepted'] = True
        result['reason'] = 'single_query_with_independent_geometry_and_measured_context_quality'
    return result


def apply_fallback(bank, original, support, text, valid_ids, space,
                   canonical_features=None, thresholds=None):
    result = dict(original)
    ledger = []
    for instance, observations in bank.items():
        if len(observations) != 1:
            continue
        decision = fallback_decision(observations, support['objects'].get(str(instance), {}), thresholds)
        decision['instance_id'] = instance
        if decision['accepted']:
            predicted = classify(observations, text, valid_ids, space,
                                  canonical_features=canonical_features, min_queries=1)
            result[str(instance)] = predicted
            decision.update({'class_id': predicted['class_id'], 'confidence': predicted['score'],
                             'confidence_type': 'native_canonical_relative_cosine_not_calibrated_probability'})
        ledger.append(decision)
    return result, ledger
