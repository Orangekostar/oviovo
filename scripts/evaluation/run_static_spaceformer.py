#!/usr/bin/env python3
"""Run complete released extra-3D-pretrained segmentation on observed RGB-D only."""
import argparse
import gc
import json
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.attention_runtime import omit_unused_attention_weights, chunk_decoder_queries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('release-root', 'checkpoint', 'siglip2-root', 'native-text-cache', 'cloud', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    p.add_argument('--omit-unused-attention-weights', action='store_true')
    p.add_argument('--attention-query-chunk', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('new output directory required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    sys.path.insert(0, str(args.release_root/'demo'))
    from text_encoder import Siglip2TextEncoder
    from clip_eval import CLIPAlignmentEval
    from labels import PROMPT_TEMPLATES
    from pipeline import make_batch, POST_PROCESSING_CFG
    from postprocessing import apply_post_processing
    import warpconvnet
    from warpconvnet.models.spaceformer import build_spaceformer, load_spaceformer_checkpoint
    device = torch.device('cuda:0')
    sources = [Path(__file__), Path(__file__).resolve().parents[2]/'src/static_ovmap/attention_runtime.py',
               args.checkpoint, args.cloud, args.cloud.parent/'receipt.json',
               args.native_text_cache, args.native_text_cache.with_suffix('.json')]
    sources += sorted((args.release_root/'demo').glob('*.py'))
    sources += [f for f in args.siglip2_root.iterdir() if f.is_file()]
    installed_root = Path(warpconvnet.__file__).parent
    sources += sorted((installed_root/'models/spaceformer').glob('*.py'))
    identity = {'input_source_sha256': {str(f): sha256_file(f) for f in sources},
        'warpconvnet': warpconvnet.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'seed': 0, 'extra_3D_training': True, 'GT_input': False, 'protocol': 'Replica_Room0_200frames_s10',
        'training_scene_exclusion': 'UNVERIFIED_DO_NOT_CLAIM_UNSEEN_SCANNET',
        'text_space': 'SigLIP2_so400m_patch14_224_1152_release_ensemble_normalize_input_false',
        'postprocessing': POST_PROCESSING_CFG, 'precision': args.precision,
        'omit_unused_attention_weights': args.omit_unused_attention_weights,
        'attention_query_chunk': args.attention_query_chunk,
        'precision_scope': 'BF16_runtime_adaptation_not_exact_FP32' if args.precision == 'bf16' else 'released_FP32_forward',
        'command': sys.argv}
    (args.output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    (args.output/'job.json').write_text(json.dumps({'status': 'BUILDING_TEXT'}))
    start = time.perf_counter()
    native_text = np.load(args.native_text_cache, allow_pickle=False)
    labels = json.loads(args.native_text_cache.with_suffix('.json').read_text())['labels']
    valid_ids = native_text['valid_ids']
    torch.cuda.reset_peak_memory_stats()
    encoder = Siglip2TextEncoder(model_id=str(args.siglip2_root), device=str(device))
    text_eval = CLIPAlignmentEval(normalize_input=False)
    text_eval.prepare_target_embedding(labels, encoder, device, prompt_templates=list(PROMPT_TEMPLATES))
    text_features = text_eval.emb_target.detach().cpu().numpy()
    if text_features.shape != (len(valid_ids), 1152) or not np.isfinite(text_features).all():
        raise ValueError('released text dimension/vocabulary/finite check failed')
    np.savez_compressed(args.output/'text.npz', text=text_features, valid_ids=valid_ids)
    torch.cuda.synchronize()
    text_seconds = time.perf_counter()-start
    text_peak = torch.cuda.max_memory_allocated()
    del encoder
    gc.collect(); torch.cuda.empty_cache()
    (args.output/'job.json').write_text(json.dumps({'status': 'LOADING_COMPLETE_SEGMENTOR'}))
    start = time.perf_counter()
    net = build_spaceformer(device=device)
    missing, unexpected = load_spaceformer_checkpoint(net, str(args.checkpoint), strict=True)
    if args.attention_query_chunk:
        attention_modules = chunk_decoder_queries(net, args.attention_query_chunk)
    else:
        attention_modules = omit_unused_attention_weights(net) if args.omit_unused_attention_weights else 0
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-start
    with np.load(args.cloud, allow_pickle=False) as data:
        coord, color = data['coord'], data['color']
    batch = make_batch(coord, color, device)
    cmin, cmax = coord.min(axis=0), coord.max(axis=0)
    shift = np.array([(cmin[0]+cmax[0])/2, (cmin[1]+cmax[1])/2, cmin[2]], dtype=np.float32)
    (args.output/'job.json').write_text(json.dumps({'status': 'FORWARD', 'input_points': len(coord)}))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
        out = net(batch)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter()-start
    forward_peak = torch.cuda.max_memory_allocated()
    output_coord = out['backbone_pc'].coordinates.detach().cpu().numpy().astype(np.float32)+shift
    mask_logits, binary, embeddings = out['mask'][0].T, out['logit'][0], out['clip_feat'][0]
    if mask_logits.shape != (200, len(output_coord)) or embeddings.shape != (200, 1152):
        raise ValueError('complete segmentation output interface or mask coordinates differ')
    if not all(torch.isfinite(x).all() for x in (mask_logits, binary, embeddings)):
        raise ValueError('nonfinite segmentation output')
    # Retain exact float logits for audit; no GT-selected pruning and no raw-input row assumption.
    np.save(args.output/'raw_mask_logits.npy', mask_logits.float().cpu().numpy())
    np.savez_compressed(args.output/'raw_queries.npz', coord=output_coord,
        objectness_logits=binary.float().cpu().numpy(), clip_features=embeddings.float().cpu().numpy(),
        center_shift=shift)
    start = time.perf_counter()
    with torch.inference_mode():
        class_logits = text_eval.predict(embeddings, return_logit=True)
        masks, scores, _, indices = apply_post_processing(mask_logits, binary, mask_threshold=0.,
            point_coords=None, pp_cfg=POST_PROCESSING_CFG, pred_iou=None)
        probs = torch.softmax(class_logits[indices], dim=-1)
        class_probability, class_index = probs.max(dim=1)
        final_scores = scores*class_probability
        order = torch.argsort(final_scores, descending=True)
    torch.cuda.synchronize()
    post_seconds = time.perf_counter()-start
    np.savez_compressed(args.output/'T0.npz', coord=output_coord,
        masks=masks[order].cpu().numpy().astype(bool), scores=final_scores[order].float().cpu().numpy(),
        objectness_mask_scores=scores[order].float().cpu().numpy(), class_probabilities=class_probability[order].float().cpu().numpy(),
        class_ids=valid_ids[class_index[order].cpu().numpy()], query_ids=indices[order].cpu().numpy())
    receipt = {'status': 'COMPLETE_T0_PENDING_EVALUATION', 'text_seconds': text_seconds,
        'model_load_seconds': load_seconds, 'forward_seconds': forward_seconds, 'postprocessing_seconds': post_seconds,
        'text_peak_gpu_bytes': text_peak, 'forward_peak_gpu_bytes': forward_peak,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'input_points': len(coord), 'output_points': len(output_coord), 'processed_instances': len(indices),
        'missing_keys': missing, 'unexpected_keys': unexpected, 'attention_modules_adapted': attention_modules,
        'full_parameter_count': sum(p.numel() for p in net.parameters()),
        'coordinate_alignment': 'returned_backbone_pc_rows_restored_center_shift',
        'cost_scope': 'text_load_encode_model_load_forward_postprocessing_separate; RGBD_fusion_separate_receipt',
        'identity_sha256': sha256_file(args.output/'identity.json'),
        'payload_sha256': {f.name: sha256_file(f) for f in args.output.iterdir() if f.suffix in ('.npz', '.npy')}}
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    (args.output/'job.json').write_text(json.dumps({'status': 'COMPLETE_T0_PENDING_EVALUATION'}))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
