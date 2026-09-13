#!/usr/bin/env python3
"""Cache native SigLIP text outputs, with exact vocabulary and model identity."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--semantic-const', type=Path, required=True)
    parser.add_argument('--image-encoder-source', type=Path, required=True)
    parser.add_argument('--dataset', choices=['Replica', 'ScanNet'], default='Replica')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import hashlib
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    constants = runpy.run_path(str(args.semantic_const))
    words = constants['REPLICA_51' if args.dataset == 'Replica' else 'CLASS_LABELS_200']
    ids = list(range(1, 52)) if args.dataset == 'Replica' else list(constants['VALID_CLASS_IDS_200'])
    canonical = ['object', 'things', 'stuff', 'texture']
    identity = {'model': 'google/siglip-large-patch16-384',
                'model_sha256': sha256_file(args.model / 'model.safetensors'),
                'layer': 'get_image_features', 'feature_dim': 1024,
                'image_encoder_source_sha256': sha256_file(args.image_encoder_source),
                'crop': 'native_six_crops_expand_0_0.1_0.2_black_and_rgb_exclusive_slice',
                'crop_fusion': 'mean_of_l2_normalized_crop_features',
                'preprocessor_sha256': sha256_file(args.model / 'preprocessor_config.json')}
    space = 'sha256:' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    start = time.perf_counter()
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True).eval().to(args.device)
    def encode(labels):
        inputs = tokenizer(labels, padding='max_length', max_length=64,
                           return_tensors='pt').to(args.device)
        with torch.no_grad():
            return model.get_text_features(**inputs).cpu().float().numpy()
    text, canon = encode(list(words)), encode(canonical)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(args.device)
    seconds = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, text=text, canonical=canon, valid_ids=np.asarray(ids),
             feature_space_id=np.asarray(space))
    metadata = {'feature_space_id': space, 'feature_space_identity': identity,
                'labels': list(words), 'valid_ids': ids, 'canonical_phrases': canonical,
                'semantic_const_sha256': sha256_file(args.semantic_const),
                'tokenizer_sha256': sha256_file(args.model / 'tokenizer.json'),
                'dataset': args.dataset, 'text_prompt': 'bare_class_name',
                'tokenization': {'padding': 'max_length', 'max_length': 64},
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'text_cache_seconds_with_model_load': seconds, 'device': args.device,
                'additional_image_queries': 0, 'output_sha256': sha256_file(args.output),
                'command': sys.argv}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'shape': list(text.shape),
                      'feature_space_id': space, 'seconds': seconds}))


if __name__ == '__main__':
    main()
