"""Actual frozen FC regions and two matched1000 heads on a new native bundle."""
import argparse
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np

from .common import (verified, write, save_arrays, ConsumedInputs, Resolver,
                     canonical_digest, source_identity)


def run(binding, bundle_path):
    import torch
    from PIL import Image
    from static_ovmap.cvpr_compact.region_worker import FCSession
    from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache
    from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor, signed_mask
    from static_ovmap.cvpr_compact.area_fallback import region_vector, PROTOCOL
    from static_ovmap.module_validation.native_capture import _array_digest
    from static_ovmap.learned_object_readout.features import execution_config
    from .registry import load_models, configure_torch

    configure_torch()
    root = Path(binding['output_root'])
    bundle = verified(bundle_path)
    folder = root / 'benchmark/F' / bundle['benchmark'] / bundle['scene']
    index, resolver = ConsumedInputs(root), Resolver(binding['path_map'])
    index.identity(bundle_path)
    cohort = 'replica8' if bundle['benchmark'] == 'replica48' else 'scannet_cf18'
    template = binding['inherited']['cohorts'][cohort][0]
    context = resolver.rewrite(verified(resolver.resolve(binding['inherited']['scenes'][template]['context']['path'])))
    config = execution_config(binding['inherited'])
    session = FCSession(config, context, index, device='cuda')
    # Replica51 can only be selected into48 through explicit canonical names;
    # never reinterpret the old class IDs as the author48 IDs.
    from .protocol import normalize_name
    from .common import read
    old_names = read(context['config'])['models']['native']['class_names']
    positions = {normalize_name(n): i for i, n in enumerate(old_names)}
    rows = [positions[normalize_name(n)] for n in bundle['text_names']]
    text = session.text[rows].copy()
    if not np.isfinite(text).all() or text.shape != (len(bundle['text_ids']), 768):
        raise ValueError('Actual FC text does not align with the target class names')
    identity = canonical_digest(dict(bundle=bundle['identity'], model=session.model_key,
        text=_array_digest(text), model_lock=bundle['model_lock'],
        producer=source_identity(('fc_worker',))))
    path = folder / 'receipt.json'
    if path.exists():
        old = verified(path)
        if old['input_identity'] != identity:
            raise ValueError('Completed new-scene F input changed')
        index.identity(old['arrays']['path'], old['arrays'])
        return old
    start = time.monotonic()
    cache = ContentCache(binding['inherited']['assets']['read_only_dense_cache_roots'],
        root / 'content_cache/fc', session.model_key, index=index, resolver=resolver)
    groups = {}
    for rid, request in bundle['requests'].items():
        groups.setdefault(request['image_sha256'], []).append(rid)
    features, audits = {}, {}
    stats = dict(physical_image_encodings=0, physical_region_poolings=0,
                 dense_cache_hits=0, region_cache_hits=0, area_fallbacks=0)
    with torch.inference_mode():
        for image_sha, rids in groups.items():
            rgb_record = bundle['requests'][rids[0]]['rgb']
            index.identity(rgb_record['path'], rgb_record)
            rgb = np.asarray(Image.open(rgb_record['path']).convert('RGB')).copy()
            image, size = image_tensor(rgb, 'cuda')
            tensor_key = canonical_digest(dict(model=session.model_key, tensor=_array_digest(image.cpu().numpy())))
            pending = []
            for rid in rids:
                request = bundle['requests'][rid]
                desc = request['input_masks']
                index.identity(desc['path'], desc)
                with np.load(desc['path'], allow_pickle=False) as arrays:
                    mask = arrays['masks'][request['mask_index']].copy()
                if mask.shape != rgb.shape[:2] or mask.dtype != bool or _array_digest(mask) != request['target_mask_sha256']:
                    raise ValueError('New FC mask/image changed before encoding')
                key = canonical_digest(dict(image=tensor_key, mask=_array_digest(mask), pooling=PROTOCOL))
                hit = cache.lookup('regions', key)
                if hit:
                    with np.load(hit['arrays']['path'], allow_pickle=False) as arrays:
                        features[rid] = arrays['feature'].copy()
                    audits[rid] = dict(cache=hit, pooling=hit['pooling_audit'])
                    stats['region_cache_hits'] += 1
                else:
                    pending.append((rid, mask, key))
            if pending:
                session.load_model()
                hit = cache.lookup('dense', tensor_key)
                if hit:
                    with np.load(hit['arrays']['path'], allow_pickle=False) as arrays:
                        dense = torch.from_numpy(arrays['dense'].copy()).cuda()
                    stats['dense_cache_hits'] += 1
                else:
                    dense = session.operators['extract_features_convnext'](SimpleNamespace(clip_model=session.model), image)['clip_vis_dense']
                    if dense.dtype != torch.float32 or not torch.isfinite(dense).all():
                        raise ValueError('Frozen FC dense output is not finite FP32')
                    hit = cache.write('dense', tensor_key, dict(dense=dense.cpu().numpy()),
                        dict(input_tensor_key=tensor_key, image_sha256=image_sha))
                    stats['physical_image_encodings'] += 1
                for rid, mask, key in pending:
                    signed, _, _ = signed_mask(mask, size, image.shape[-2:], dense.shape[-2:], 'cuda')
                    vector, audit = region_vector(session.model, session.operators, dense, signed)
                    features[rid] = vector.cpu().numpy().copy()
                    record = cache.write('regions', key, dict(feature=features[rid]),
                        dict(dense_receipt=hit['receipt_path'], input_tensor_key=tensor_key,
                             pooling_operator=PROTOCOL, pooling_audit=audit))
                    audits[rid] = dict(cache=record, pooling=audit)
                    stats['physical_region_poolings'] += 1
                    stats['area_fallbacks'] += int(audit['fallback'])
                del dense
            print('F_IMAGE', bundle['scene'], len(features), '/', len(bundle['requests']), flush=True)
            del image
    for vector in features.values():
        if vector.shape != (768,) or vector.dtype != np.float32 or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12:
            raise ValueError('A qualified supported FC region has an invalid actual vector')
    del session.model
    torch.cuda.empty_cache()
    heads, weights = load_models(binding)
    locked = verified(root / 'model_lock.json')
    if weights != locked['weights']:
        raise ValueError('New-scene actual MV exports differ from frozen model lock')
    owners = np.asarray([obj['owner'] for obj in bundle['objects']], np.int64)
    b = np.zeros((len(owners), 768), np.float32)
    z = np.zeros((2, len(owners), 768), np.float32)
    available = np.asarray([bool(obj['views']) for obj in bundle['objects']], bool)
    heads_called = 0
    with torch.inference_mode():
        for i, obj in enumerate(bundle['objects']):
            if not obj['views']:
                continue
            vectors = torch.from_numpy(np.stack([features[rid] for rid in obj['views']])).cuda()
            v = len(vectors)
            inputs = dict(fc_vectors=vectors, view_valid=torch.ones(v, device='cuda', dtype=torch.bool),
                frame_ids=torch.tensor([bundle['requests'][rid]['frame_id'] for rid in obj['views']], device='cuda'),
                local_raw=vectors.new_empty(v, 0, 1536), local_unit=vectors.new_empty(v, 0, 768),
                metadata=vectors.new_empty(v, 0, 8), local_valid=torch.empty(v, 0, device='cuda', dtype=torch.bool))
            for si, seed in enumerate((17, 29)):
                out = heads[seed](inputs, None)
                actual_b = out['base'].cpu().numpy()
                z[si, i] = out['embedding'].cpu().numpy()
                if si == 0:
                    b[i] = actual_b
                elif not np.array_equal(actual_b, b[i]):
                    raise ValueError('Two MV heads changed the shared FrozenFCBase anchor')
                heads_called += 1
    if not np.isfinite(z[:, available]).all() or not np.isfinite(b[available]).all():
        raise ValueError('Supported MV output is nonfinite; no fallback is permitted')
    array_path = folder / 'vectors.npz'
    save_arrays(array_path, owners=owners, b=b, z=z, available=available,
                text=text, valid_ids=np.asarray(bundle['text_ids'], np.int64))
    result = write(path, dict(status='DONE', input_identity=identity, bundle=bundle['identity'],
        arrays=index.identity(array_path), model_lock=bundle['model_lock'], weights=weights,
        actual_head_calls=heads_called, logical_anchor_outputs=int(available.sum()),
        zero_F_owners=int((~available).sum()), source_text=context['FC_text'],
        text_renormalized=False, pooling=PROTOCOL, pooling_matches_MV_training_input=True,
        requests=audits, new_training_updates=0, elapsed_seconds=time.monotonic() - start, **stats))
    index.save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    parser.add_argument('--bundle', required=True)
    args = parser.parse_args()
    result = run(verified(args.binding), args.bundle)
    print('F_DONE', result['actual_head_calls'], result['physical_image_encodings'], flush=True)
