"""All-owner GLOBAL MF×AC classification and overlap-preserving mask export."""
from pathlib import Path
import numpy as np

from .common import (verified, write, save_arrays, digest_file, canonical_digest,
                     source_identity, StageFailure)

VARIANTS = ('U00', 'U10', 'U01', 'U11')


def classify_scores(variant, valid_ids, f_raw, f_calibrated, source_loader, temperatures):
    """N/Q accessor is never called for either MF-off cell, including missing F."""
    from static_ovmap.paired_evidence_study.residuals import pool, simple_dependence
    if variant not in VARIANTS:
        raise ValueError('Unknown GLOBAL variant')
    ids = np.asarray(valid_ids)
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or (ids <= 0).any():
        raise ValueError('Prediction text class axis must contain unique positive semantic IDs')
    mf = variant in ('U10', 'U11')
    ac = variant in ('U01', 'U11')
    f = f_calibrated if ac else f_raw
    scores = {'F': f}
    if f is not None and (np.asarray(f).shape != ids.shape or not np.isfinite(f).all()):
        raise ValueError('Nonfinite supported F or invalid class axis')
    if mf:
        for name in ('N', 'Q'):
            value = source_loader(name)
            if value is not None and (np.asarray(value).shape != ids.shape or not np.isfinite(value).all()):
                raise ValueError('Nonfinite supported ' + name + ' or invalid class axis')
            scores[name] = value
        probabilities = pool(scores, temperatures, ('N', 'Q', 'F'))
        probabilities = simple_dependence(scores, temperatures, probabilities, 'PE_D2_GROUPED')
        label = 0 if probabilities is None else int(ids[int(np.argmax(probabilities))])
        if probabilities is not None and (not np.isfinite(probabilities).all() or not np.isclose(np.sum(probabilities), 1)):
            raise ValueError('Supported GLOBAL D2 output is invalid')
    else:
        probabilities = None
        label = 0 if f is None else int(ids[int(np.argmax(f))])
    return dict(label=label, available_sources=[name for name, value in scores.items() if value is not None],
                MF=mf, AC=ac, confidence=1., unknown=label == 0)


def score_F(b, z, text, base_mask, *, calibrated):
    """Use true vectors and original text dtype; subtract gamma before temperature."""
    from static_ovmap.semantic_update_evidence_v1.calibration import angle_cap
    b, z, text = np.asarray(b), np.asarray(z), np.asarray(text)
    if b.shape != (768,) or z.shape != (768,) or text.ndim != 2 or text.shape[1] != 768:
        raise ValueError('Actual F/MV768 dimensions required')
    if not all(np.isfinite(v).all() for v in (b, z, text)):
        raise ValueError('Nonfinite supported F/MV vectors are a runtime error')
    raw = text @ z
    if not calibrated:
        return raw, None
    if base_mask is None:
        raise StageFailure('Full CAP+gamma vocabulary has unresolved training-base roles')
    base = np.asarray(base_mask, bool)
    if base.shape != raw.shape:
        raise ValueError('Gamma mask differs from actual target text axis')
    clipped, _ = angle_cap(b[None], z[None], 10.)
    result = text @ clipped[0]
    result[base] -= .025
    return raw, result


def save_manifest(folder, *, scene, benchmark, method, seed, owners, indptr, indices,
                  labels, target_points, valid_instance_ids, provenance):
    """CSR masks preserve overlaps and unsupported owners; filtering is export-only."""
    folder = Path(folder)
    owners, indptr, indices, labels = (np.asarray(value, np.int64) for value in (owners, indptr, indices, labels))
    if (owners.ndim != 1 or len(owners) != len(set(owners.tolist())) or (owners <= 0).any()
            or labels.shape != owners.shape or indptr.shape != (len(owners) + 1,)
            or indptr[0] != 0 or indptr[-1] != len(indices) or (np.diff(indptr) < 0).any()
            or indices.ndim != 1 or (indices < 0).any() or (indices >= target_points).any()):
        raise ValueError('Immutable all-owner mask/label domain is invalid')
    for start, end in zip(indptr[:-1], indptr[1:], strict=True):
        if len(np.unique(indices[start:end])) != end - start:
            raise ValueError('One proposal mask contains duplicate vertex indices')
    confidence = np.ones(len(owners), np.float64)
    active = np.isin(labels, valid_instance_ids)
    mask_identity = canonical_digest(dict(owners=owners, indptr=indptr,
        index_bytes_sha256=__import__('hashlib').sha256(indices.tobytes()).hexdigest(), target_points=target_points))
    prediction_identity = canonical_digest(dict(masks=mask_identity, labels=labels, confidence=confidence))
    identity = canonical_digest(dict(prediction=prediction_identity, provenance=provenance,
        method=method, seed=seed, producer=source_identity(('standard_export',))))
    receipt = folder / 'manifest.json'
    if receipt.exists():
        previous = verified(receipt)
        if previous['input_identity'] != identity or digest_file(folder / 'masks.npz') != previous['arrays']['sha256']:
            raise ValueError('Completed GLOBAL export changed; preserve before affected repair')
        return previous
    path = folder / 'masks.npz'
    save_arrays(path, owners=owners, indptr=indptr, indices=indices, labels=labels,
                confidence=confidence, exported=active)
    return write(receipt, dict(status='DONE', input_identity=identity, scene=scene, benchmark=benchmark,
        method=method, seed=seed, format='OVMAP_INDEPENDENT_MASK_CSR_V1', arrays=dict(
            path=str(path), bytes=path.stat().st_size, sha256=digest_file(path)),
        prediction_identity=prediction_identity, proposal_identity=mask_identity,
        immutable_owner_count=len(owners), target_points=int(target_points), valid_instance_ids=list(valid_instance_ids),
        exported_instances=int(active.sum()), unknown_owners=int((labels == 0).sum()),
        excluded_stuff_or_invalid_owners=int(((labels != 0) & ~active).sum()),
        overlap_preserved=True, confidence=1., top1_per_owner=True,
        GT_denominator_unchanged=True, GT_labels_used_for_prediction=False,
        scope='GLOBAL_ALL_ORIGINAL_OVI_NATIVE_OWNERS', provenance=provenance))


def official_prediction(manifest):
    """Decode into the exact common evaluator arrays without changing mask overlap."""
    path = Path(manifest['arrays']['path'])
    if digest_file(path) != manifest['arrays']['sha256']:
        raise ValueError('Prediction mask arrays changed')
    with np.load(path, allow_pickle=False) as arrays:
        active = np.flatnonzero(arrays['exported'])
        masks = np.zeros((manifest['target_points'], len(active)), np.uint8)
        for column, i in enumerate(active):
            masks[arrays['indices'][arrays['indptr'][i]:arrays['indptr'][i + 1]], column] = 1
        return dict(pred_masks=masks, pred_classes=arrays['labels'][active].copy(),
                    pred_scores=arrays['confidence'][active].copy())
