#!/usr/bin/env python3
"""ROI control on exactly the dense-visible slots, without additional encoding or GT."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.query_scores import score_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    binding = json.loads((args.dense_run/'binding.json').read_text())
    slots = binding['slots']
    data = np.load(args.dense_run/'retained_features.npz', allow_pickle=False)
    text = np.load(args.dense_run/'dense_text.npz', allow_pickle=False)
    c0, c1 = binding['conditions'].index('D0_ROI'), binding['conditions'].index('D1')
    visible = data['observation_valid'][c1]
    if np.any(visible & ~data['observation_valid'][c0]):
        raise ValueError('dense-visible slot lacks its ROI control')
    baseline = json.loads((args.dense_run/'readout/B0.json').read_text())
    predictions = {key: None for key in baseline['observations']}
    for owner in sorted({s['owner'] for s in slots}):
        chosen = [i for i, slot in enumerate(slots) if slot['owner'] == owner and visible[i]]
        if len(chosen) < 2:
            continue
        weights = np.asarray([slots[i]['native_area'] for i in chosen], dtype=np.float64)
        feature = (data['observations'][c0, chosen]*weights[:, None]).sum(axis=0)/(weights.sum()+1e-6)
        space = binding['roi_feature_space_id']
        predictions[str(owner)] = {**score_queries(feature, text['text'], text['valid_ids'], space, space),
            'selected_query_ids': [slots[i]['source_query_id'] for i in chosen],
            'selected_count': len(chosen)}
    dense_doc = json.loads((args.dense_run/'readout/D1.json').read_text())
    for owner, value in predictions.items():
        other = dense_doc['observations'][owner]
        if (value is None) != (other is None) or (value is not None and value['selected_query_ids'] != other['selected_query_ids']):
            raise ValueError('ROI/dense matched budget or eligibility differs')
    document = {**dense_doc, 'condition': 'D0_MATCHED_VISIBLE', 'observations': predictions,
        'readout_seconds': time.perf_counter()-start, 'feature_space_id': binding['roi_feature_space_id'],
        'control': 'same_D1_visible_query_slots_and_min_two_gate; no_extra_image_calls; GT_free'}
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.dense_run/'readout/B0.json', args.output/'B0.json')
    (args.output/'D0_MATCHED_VISIBLE.json').write_text(json.dumps(document, indent=2)+'\n')
    receipt = {'command': sys.argv, 'valid_slots': int(visible.sum()), 'encoder_calls_added': 0,
        'source_sha256': sha256_file(Path(__file__)),
        'input_sha256': {f.name: sha256_file(f) for f in [args.dense_run/'binding.json',
            args.dense_run/'retained_features.npz', args.dense_run/'dense_text.npz']}}
    (args.output/'input_binding.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
