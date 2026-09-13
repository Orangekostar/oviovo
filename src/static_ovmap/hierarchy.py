"""Fixed prediction-only semantic selection for native instance components."""
from dataclasses import replace

from .readout import classify


def component_readout(components, bank, baseline, text, valid_ids, space, canonical):
    result, counts = {}, {}
    for members in components:
        key = str(min(members))
        if len(members) == 1:
            result[key] = baseline.get(key)
            counts[key] = len(result[key].get('selected_query_ids', [])) if result[key] else 0
            continue
        chosen = []
        for owner in sorted(members):
            previous = baseline.get(str(owner))
            if previous is None:
                continue
            wanted = set(previous['selected_query_ids'])
            selected = [obs for obs in bank[owner] if obs.source_query_id in wanted]
            if len(selected) != len(wanted):
                raise ValueError('baseline selected queries missing from immutable native bank')
            # Only this declared component permits reparenting; native records stay immutable.
            # source_query_id still identifies the original owner and query index.
            chosen.extend(replace(obs, instance_id=int(key)) for obs in selected)
        result[key] = classify(chosen, text, valid_ids, space, strategy='all_views',
                                weighting='vis_area', canonical_features=canonical)
        counts[key] = len(chosen)
    return {'observations': result, 'pooled_query_counts': counts,
            'query_rule': 'union_of_member_B0_eligible_selected_queries_original_vis_area',
            'budget_scope': 'up_to_8_per_original_eligible_owner_not_fixed_8_per_merged_object'}
