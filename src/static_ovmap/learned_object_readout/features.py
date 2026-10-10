"""Frozen FP32 FC grids, separate target sidecars and bounded host caching."""
import ast
from collections import OrderedDict
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from .common import ConsumptionIndex, PathResolver, arrays_record, canonical_digest, read, verified, write


def execution_config(binding):
    cfg = dict(binding['assets']['execution_config'])
    cfg.update(gpu=binding['gpu'], path_map=binding['path_map'],
               gpu_lock=f"/mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-{binding['gpu']}.lock")
    return cfg


class FrozenFC:
    def __init__(self, binding, *, device='cuda'):
        import torch
        from static_ovmap.cvpr_compact.region_worker import FCSession
        from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache
        self.binding, self.root, self.device = binding, Path(binding['output_root']), device
        self.index = ConsumptionIndex(self.root / 'features/verifications.json')
        # Only the existing full text/model context is consumed, not any GT geometry.
        scene = binding['cohorts']['scannet_cf18'][0]
        row = binding['scenes'][scene]['context']
        self.index.identity(row['path'], row)
        context = PathResolver(binding['path_map']).rewrite(verified(row['path']))
        self.session = FCSession(execution_config(binding), context, self.index, device=device)
        self.model = self.session.load_model()
        self.operators, self.model_key = self.session.operators, self.session.model_key
        self.text, self.ids = self.session.text, list(map(int, self.session.ids))
        restored = self.text.astype(np.float32)
        if not np.array_equal(restored.astype(self.text.dtype), self.text):
            raise ValueError('Frozen text does not preserve its numerical values in FP32')
        self.text_tensor = torch.from_numpy(restored.copy()).to(device)
        classes = verified(self.root / 'class_split.json')
        config = PathResolver(binding['path_map']).rewrite(read(context['config']))
        names = config['models']['native']['class_names']
        official = dict(zip(classes['official_ids'], classes['official_names']))
        if set(self.ids) != set(official) or len(names) != len(self.ids):
            raise ValueError('Explicit complete ScanNet200 ID mapping required')
        mapping = [dict(class_id=i, official_name=official[i], frozen_name=n, text_row=j)
                   for j, (i, n) in enumerate(zip(self.ids, names))]
        if any(r['official_name'].strip().casefold() != r['frozen_name'].strip().casefold() for r in mapping):
            raise ValueError('Frozen text and official canonical class names differ')
        write(self.root / 'features/text_mapping.json', dict(status='COMPLETE', classes=mapping,
              text=self.session.text_identity, FP32_lossless=True, normalized_again=False))
        self.base_rows = [j for j, i in enumerate(self.ids) if i in classes['base_ids']]
        self.base_text = self.text_tensor[self.base_rows]
        self.base_targets = {self.ids[j]: k for k, j in enumerate(self.base_rows)}
        source = self.root / 'external/upstream_sources/MaskAdapter/fcclip/fcclip.py'
        self.dense_operator = self.index.identity(source)
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FCCLIP')
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'visual_prediction_forward_convnext_2d')
        function.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        self.project_grid_operator = namespace[function.name]
        # Parent roots are finite and read-only; writes go only to this task.
        roots = list(binding['assets']['read_only_dense_cache_roots'])
        roots += [str(Path(binding['observation_root']) / 'content_cache/fc')]
        self.raw_cache = ContentCache(roots, self.root / 'content_cache/fc', self.model_key,
                                     index=self.index, resolver=PathResolver(binding['path_map']))
        self.images = 0
        self.model.eval().requires_grad_(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def phi(self, value):
        """The pinned frozen vector projection remains differentiable in value."""
        if value.ndim < 2:
            raise ValueError('Raw vectors need an explicit trailing C axis')
        shape = value.shape[:-1]
        pooled = value.reshape(1, -1, value.shape[-1])
        result = self.operators['visual_prediction_forward_convnext'](
            SimpleNamespace(clip_model=self.model), pooled, None)
        return result.reshape(*shape, result.shape[-1])

    def project_grid(self, raw):
        wrapper = SimpleNamespace(backbone=SimpleNamespace(clip_model=self.model))
        return self.project_grid_operator(wrapper, raw)

    def capture(self, frame, *, engineering=False):
        import torch
        from PIL import Image
        from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor
        from static_ovmap.module_validation.native_capture import _array_digest
        self.index.identity(frame['rgb']['path'], frame['rgb'])
        rgb = np.asarray(Image.open(frame['rgb']['path']).convert('RGB')).copy()
        image, resized = image_tensor(rgb, self.device)
        tensor_key = canonical_digest(dict(model=self.model_key, tensor=_array_digest(image.cpu().numpy())))
        feature_key = canonical_digest(dict(raw=tensor_key, projected=self.dense_operator, dtype='FP32'))
        scope = 'engineering' if engineering else 'scientific'
        path = self.root / 'features' / scope / 'grids' / (feature_key + '.json')
        if path.exists():
            record = verified(path)
            self.index.identity(record['arrays']['path'], record['arrays'])
            return record
        begin = time.perf_counter()
        # no_grad creates ordinary tensors; inference_mode tensors cannot train MA.
        with torch.no_grad():
            hit = None if engineering else self.raw_cache.lookup('dense', tensor_key)
            if hit is not None:
                with np.load(hit['arrays']['path'], allow_pickle=False) as values:
                    dense = torch.from_numpy(values['dense'].copy()).to(self.device)
            else:
                dense = self.operators['extract_features_convnext'](SimpleNamespace(clip_model=self.model), image)['clip_vis_dense']
                self.images += 1
            if dense.dtype != torch.float32 or dense.ndim != 4 or not torch.isfinite(dense).all():
                raise ValueError('Invalid frozen raw FP32 FC tensor')
            projected = self.project_grid(dense)
            if projected.dtype != torch.float32 or projected.shape[1] != self.text.shape[1] or projected.shape[1] != 768:
                raise ValueError('Measured FC projection width incompatible with fixed MA protocol')
            if not torch.isfinite(projected).all() or torch.is_inference(dense) or torch.is_inference(projected):
                raise ValueError('Feature cache must contain ordinary finite FP32 tensors')
            torch.cuda.synchronize() if self.device == 'cuda' else None
            elapsed = time.perf_counter()-begin
            arrays = arrays_record(path.with_suffix('.npz'), dict(raw=dense.cpu().numpy(), projected=projected.cpu().numpy()), self.index)
        record = write(path, dict(status='COMPLETE', key=feature_key, arrays=arrays, raw_C=int(dense.shape[1]),
                projected_D=int(projected.shape[1]), raw_hw=list(dense.shape[-2:]), original_hw=list(rgb.shape[:2]),
                resized_hw=list(resized), padded_hw=list(image.shape[-2:]), input_tensor_key=tensor_key,
                physical_model_identity=self.model_key, dense_projection=self.dense_operator,
                image=frame['rgb'], physical_image_encodings=int(hit is None), cache_origin='PARENT' if hit else 'ENCODED',
                extraction_seconds=elapsed, scope=scope, L2_normalized_dense=False))
        estimate = int(dense.numel()+projected.numel()) * 4 * 3904 / 2**30
        if not (self.root / 'features/storage_estimate.json').exists():
            import shutil
            available = shutil.disk_usage(self.root).free / 2**30
            write(self.root / 'features/storage_estimate.json', dict(measured_raw_shape=list(dense.shape),
                  measured_projected_shape=list(projected.shape), uncompressed_max_scientific_GiB=estimate,
                  available_GiB=available, scientific_image_cap=3904))
            if available < estimate + 40:
                raise OSError('BLOCKED_RUNTIME: measured full-cache reserve exceeds task storage')
        return record


class ObjectLoader:
    """Inputs are built only from proposal observations and frozen images."""
    def __init__(self, fc, feature_manifest, *, maximum_GiB=32):
        self.fc, self.frames = fc, feature_manifest['frames']
        self.cache, self.bytes = OrderedDict(), 0
        import psutil
        self.limit = min(float(maximum_GiB)*2**30, psutil.virtual_memory().available/2)

    def _grid(self, key):
        import torch
        if key in self.cache:
            value = self.cache.pop(key); self.cache[key] = value
            return value
        record = self.frames[key]
        self.fc.index.identity(record['arrays']['path'], record['arrays'])
        with np.load(record['arrays']['path'], allow_pickle=False) as arrays:
            value = {k: torch.from_numpy(arrays[k].copy()) for k in ('raw', 'projected')}
        size = sum(v.numel()*v.element_size() for v in value.values())
        while self.cache and self.bytes+size > self.limit:
            _, removed = self.cache.popitem(last=False)
            self.bytes -= sum(v.numel()*v.element_size() for v in removed.values())
        self.cache[key] = value; self.bytes += size
        return value

    def load(self, obj, condition=0, requested_views=8):
        import torch
        from static_ovmap.a7_evidence_upgrade.region_adapter import signed_mask
        from .observations import sample_raw
        self.fc.index.identity(obj['inputs']['path'], obj['inputs'])
        with np.load(obj['inputs']['path'], allow_pickle=False) as arrays:
            allowed = ('masks', 'uv', 'metadata', 'local_valid', 'site_xyz', 'site_available')
            if set(arrays.files) != set(allowed):
                raise ValueError('Predictor inputs contain an undeclared annotation/semantic field')
            masks = arrays['masks'][condition, :requested_views].copy()
            uv = arrays['uv'][condition, :requested_views].copy()
            metadata = arrays['metadata'][condition, :requested_views].copy()
            valid = arrays['local_valid'][condition, :requested_views].copy()
        raws, projected, binaries, local_raw, local_unit, fc_vectors, fallback = [], [], [], [], [], [], []
        from static_ovmap.cvpr_compact.area_fallback import region_vector
        with torch.no_grad():
            for v, frame in enumerate(obj['views'][:requested_views]):
                key = obj['scene']+':'+str(frame['frame_id'])
                record = self.frames[key]; grid = self._grid(key)
                raw = grid['raw'].to(self.fc.device)
                raws.append(raw[0]); projected.append(grid['projected'][0].to(self.fc.device))
                signed, _, _ = signed_mask(masks[v], record['resized_hw'], record['padded_hw'], record['raw_hw'], self.fc.device)
                binaries.append(((signed+1)/2)[0])
                sampled, bounds = sample_raw(raw, uv[v], record['original_hw'], record['resized_hw'], record['padded_hw'])
                visibility = torch.from_numpy(valid[v]).to(self.fc.device) & bounds
                sampled = torch.where(visibility[:, None], sampled, 0)
                # Phi on invisible zero vectors is harmless only when excluded by the valid mask.
                unit = torch.nn.functional.normalize(self.fc.phi(sampled), dim=-1)
                local_raw.append(sampled); local_unit.append(torch.where(visibility[:, None], unit, 0))
                valid[v] = visibility.cpu().numpy()
                vector, audit = region_vector(self.fc.model, self.fc.operators, raw, signed)
                fc_vectors.append(vector); fallback.append(audit)
        inputs = dict(raw=torch.stack(raws), projected=torch.stack(projected), masks=torch.stack(binaries),
                 local_raw=torch.stack(local_raw), local_unit=torch.stack(local_unit),
                 metadata=torch.from_numpy(metadata).to(self.fc.device), local_valid=torch.from_numpy(valid).to(self.fc.device),
                 view_valid=torch.ones(len(raws), dtype=torch.bool, device=self.fc.device),
                 frame_ids=torch.tensor([v['frame_id'] for v in obj['views'][:requested_views]], device=self.fc.device))
        return inputs, torch.stack(fc_vectors), fallback

    def targets(self, obj, condition, requested_views):
        import torch
        self.fc.index.identity(obj['targets']['path'], obj['targets'])
        with np.load(obj['targets']['path'], allow_pickle=False) as arrays:
            membership = arrays['membership'][condition, :requested_views].copy()
            class_id = int(arrays['class_id'])
        return class_id, torch.from_numpy(membership).to(self.fc.device)


def capture_dataset(binding, *, holdout=False, engineering_objects=None):
    import torch
    from static_ovmap.backbone_wave1.runtime import exclusive_lock
    root = Path(binding['output_root'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != binding['gpu']:
        raise ValueError('FC worker requires the single bound CUDA_VISIBLE_DEVICES')
    scope = 'engineering' if engineering_objects is not None else ('holdout' if holdout else 'train-dev')
    path = root / 'features' / (scope+'.json')
    split = verified(root / 'split_manifest.json')
    if engineering_objects is None:
        if holdout:
            verified(root / 'dev_nomination.json'); verified(root / 'repeat_nomination.json')
        roles = ('holdout',) if holdout else ('train', 'dev')
        manifests = [verified(root / 'data/generated' / s / 'manifest.json') for role in roles for s in split['roles'][role]]
        objects = [o for m in manifests for o in m['objects']]
    else:
        objects = engineering_objects
    selected = {}
    for obj in objects:
        for frame in obj['views']:
            key = obj['scene']+':'+str(frame['frame_id'])
            if key in selected and selected[key]['rgb'] != frame['rgb']:
                raise ValueError('One physical frame has inconsistent registered RGB')
            selected[key] = frame
    identity = canonical_digest(dict(binding=binding['identity'], objects=[o['inputs'] for o in objects], scope=scope))
    if path.exists():
        previous = verified(path)
        if previous['input_identity'] != identity:
            raise ValueError('Completed feature pool changed')
        return previous
    started = time.perf_counter()
    with exclusive_lock(execution_config(binding)['gpu_lock']):
        fc = FrozenFC(binding)
        torch.cuda.reset_peak_memory_stats()
        frames = {}
        for k, frame in sorted(selected.items()):
            frames[k] = fc.capture(frame, engineering=engineering_objects is not None)
            if len(frames) % 32 == 0:
                print('FC_CAPTURE', scope, len(frames), '/', len(selected), flush=True)
        result = write(path, dict(status='COMPLETE', input_identity=identity, scope=scope, frames=frames,
                original_objects=len(objects), physical_images=len(frames), physical_encodings_this_invocation=fc.images,
                physical_encodings_persisted=sum(v['physical_image_encodings'] for v in {r['key']: r for r in frames.values()}.values()),
                FC_model=fc.model_key, raw_C=next(iter(frames.values()))['raw_C'], projected_D=768,
                elapsed_seconds=time.perf_counter()-started, model_load_seconds=fc.session.model_load_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        fc.index.write_memo(root / 'features/verifications.json')
    return result
