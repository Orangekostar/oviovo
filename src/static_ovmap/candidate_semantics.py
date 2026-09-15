"""Same-mask native SigLIP reassessment; geometry and historical readout are untouched."""
import hashlib
import json
import time

import numpy as np
from PIL import Image


def array_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode() + repr(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def semantic_cache_key(binding, pixel_mask):
    return hashlib.sha256(json.dumps(dict(binding, pixel_mask_sha256=array_digest(pixel_mask)),
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def native_crops(rgb, mask, bbox):
    """Native OVI-MAP vl_models.py operation, including exclusive upper bounds.

    Adapted from OVI-MAP/OVI-MAP scripts/vl_models.py; upstream license applies.
    No bbox repair, mask dilation, or background replacement beyond native black.
    """
    rgb, mask = np.asarray(rgb), np.asarray(mask)
    if rgb.dtype != np.uint8 or rgb.shape != (*mask.shape, 3) or mask.dtype != bool:
        raise ValueError('expected uint8 RGB and aligned boolean pixel mask')
    x1, y1, x2, y2 = map(int, bbox)
    height, width = mask.shape
    black = rgb.copy()
    black[~mask] = 0
    crops = []
    for layer in range(3):
        xp, yp = int(.1 * layer * (x2-x1)), int(.1 * layer * (y2-y1))
        left, top = max(0, x1-xp), max(0, y1-yp)
        right, bottom = min(width-1, x2+xp), min(height-1, y2+yp)
        if right <= left or bottom <= top:
            raise ValueError('degenerate_native_crop')
        for image in (rgb, black):
            crops.append(Image.fromarray(image[top:bottom, left:right]))
    return crops


def score_views(features, visible_areas, text, valid_ids, current_class,
                image_space, text_space, canonical=None, margin=.02, agreement=2/3):
    if image_space != text_space:
        raise ValueError('cannot mix image/text feature spaces')
    features, text = np.asarray(features), np.asarray(text)
    ids = np.asarray(valid_ids)
    result = {'class_id': int(current_class), 'reason': 'insufficient_views',
              'per_view_cosine': [], 'aggregate_cosine': [], 'canonical_scores': []}
    if not len(features):
        return result
    if (features.ndim != 2 or text.ndim != 2 or features.shape[1] != text.shape[1]
            or len(ids) != len(text) or current_class not in ids
            or not np.isfinite(features).all() or not np.isfinite(text).all()):
        raise ValueError('invalid native feature/text payload')
    weights = np.asarray(visible_areas, dtype=np.float64)
    if weights.shape != (len(features),) or np.any(weights <= 0):
        raise ValueError('invalid visible areas')
    unit_text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-8)
    views = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    fused = np.sum(features * (weights / weights.sum())[:, None], axis=0)
    fused /= max(float(np.linalg.norm(fused)), 1e-8)
    scores, per_view = unit_text @ fused, views @ unit_text.T
    best = int(np.argmax(scores))
    old = int(np.flatnonzero(ids == current_class)[0])
    proposed = int(ids[best])
    advantage = float(scores[best] - scores[old])
    fraction = float(np.mean(np.argmax(per_view, axis=1) == best))
    result.update(per_view_cosine=per_view.tolist(), aggregate_cosine=scores.tolist(),
                  proposed_class=proposed, advantage=advantage, view_agreement=fraction,
                  fused_feature=fused.tolist())
    if canonical is not None:
        canon = np.asarray(canonical)
        if canon.ndim != 2 or canon.shape[1] != text.shape[1]:
            raise ValueError('invalid canonical feature dimension')
        cosine = canon @ fused / np.maximum(np.linalg.norm(canon, axis=1), 1e-8)
        result['canonical_scores'] = (1/(1+np.exp(cosine[:, None]-scores[None, :]))).min(axis=0).tolist()
    if len(features) < 2:
        return result
    if np.count_nonzero(scores == scores[best]) > 1:
        result['reason'] = 'exact_tie'
    elif proposed == current_class:
        result['reason'] = 'unchanged_top1'
    elif fraction < agreement:
        result['reason'] = 'view_disagreement'
    elif advantage < margin:
        result['reason'] = 'insufficient_cosine_advantage'
    else:
        result.update(class_id=proposed, reason='accepted')
    return result


class OfflineNativeEncoder:
    """One local FP32 model; every call is one six-crop forward batch."""
    def __init__(self, model_root, device):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.device = torch, device
        start = time.perf_counter()
        self.model = AutoModel.from_pretrained(model_root, local_files_only=True).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(model_root, use_fast=True, local_files_only=True)
        self.load_seconds = time.perf_counter() - start
        self.batches = self.crop_inputs = 0
        self.encoding_seconds = self.crop_seconds = 0.

    def encode(self, rgb, mask, bbox):
        start = time.perf_counter()
        crops = native_crops(rgb, mask, bbox)
        batch = self.processor(images=crops, return_tensors='pt').to(self.device)
        self.crop_seconds += time.perf_counter() - start
        start = time.perf_counter()
        with self.torch.no_grad():
            features = self.model.get_image_features(**batch)
            features = features / features.norm(dim=-1, keepdim=True)
            feature = features.mean(dim=0).cpu().float().numpy()
        self.encoding_seconds += time.perf_counter() - start
        self.batches += 1
        self.crop_inputs += 6
        if feature.shape != (1024,) or not np.isfinite(feature).all():
            raise ValueError('native encoder returned invalid 1024-D feature')
        return feature


def subset_transition(original_class, final_class, instance_ids):
    old, new = original_class in instance_ids, final_class in instance_ids
    return 'entry' if new and not old else 'exit' if old and not new else 'unchanged'
