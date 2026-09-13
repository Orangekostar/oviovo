#!/usr/bin/env python3
"""Exercise released GLA/AnyUp weights on one real RGB frame before scene inference."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.gla_backend import GLABackend
from src.static_ovmap.cache_io import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('gla-root', 'dino-root', 'anyup-root', 'weights', 'image', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    start = time.perf_counter()
    model = GLABackend(args.gla_root, args.dino_root, args.anyup_root, args.weights)
    load_seconds = time.perf_counter()-start
    image = Image.open(args.image).convert('RGB')
    timings = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    low, info = model.dense(image)
    torch.cuda.synchronize()
    timings['dense_seconds'] = time.perf_counter()-start
    np.save(args.output/'low_features.npy', low.detach().cpu().numpy())
    stats = {}
    for method in ('bilinear', 'anyup'):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        features = model.upsample(image, low, method)
        torch.cuda.synchronize()
        timings[method+'_seconds'] = time.perf_counter()-start
        if features.shape != (1, 512, image.height, image.width) or not torch.isfinite(features).all():
            raise ValueError('invalid full-resolution released model features')
        stats[method] = {'shape': list(features.shape), 'mean': float(features.mean()),
            'std': float(features.std()), 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
        print(method, timings[method+'_seconds'], stats[method], flush=True)
        del features
    roi = model.roi(image, (0, 0, image.width-1, image.height-1))
    text = model.text_features(['chair', 'table', 'wall'])
    if roi.shape != (512,) or text.shape != (3, 512) or not torch.isfinite(roi).all() or not torch.isfinite(text).all():
        raise ValueError('ROI/text control interface failed')
    second, second_info = model.dense(image)
    difference = float((low-second).abs().max())
    if difference > 1e-5 or second_info['qkv_hook_count_after'] != 0:
        raise ValueError('repeat dense extraction drift or leaked source hooks')
    receipt = {'status': 'PASS_RELEASED_SINGLE_FRAME_INTERFACES', 'command': sys.argv,
        'identity': model.identity, 'feature_space_id': model.space, 'roi_feature_space_id': model.roi_space,
        'input_sha256': sha256_file(args.image), 'load_seconds': load_seconds, 'timings': timings,
        'dense_info': info, 'upsampler_stats': stats, 'repeat_dense_max_abs_difference': difference,
        'ROI_dim': int(roi.numel()), 'text_shape': list(text.shape),
        'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'device': 'cuda:0',
        'anyup_output_size': [image.height, image.width], 'anyup_q_chunk_size': 4096, 'use_natten': False,
        'inference_semantic_scope': 'features_extracted_before_text_queries_no_GT'}
    (args.output/'backend_audit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(receipt['status'], flush=True)


if __name__ == '__main__':
    main()
