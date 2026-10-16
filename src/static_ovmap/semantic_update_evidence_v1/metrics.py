"""Class-macro metrics, with abstentions retained in original denominators."""
import numpy as np

def harmonic(a, b):
    if a is None or b is None:
        return None
    return 2 * a * b / (a + b) if a + b else 0.

def predictions(scores, valid_ids, defined=None):
    scores = np.asarray(scores)
    valid_ids = np.asarray(valid_ids)
    if scores.shape[-1] != len(valid_ids) or len(set(valid_ids.tolist())) != len(valid_ids):
        raise ValueError('Vocabulary alignment differs')
    ok = np.isfinite(scores).all(axis=-1)
    if defined is not None:
        ok &= np.asarray(defined, dtype=bool)
    result = np.zeros(scores.shape[:-1], dtype=np.int64)
    # np.argmax returns the first original row in an exact tie.
    result[ok] = valid_ids[np.argmax(scores[ok], axis=-1)]
    return result

def recognition(target, prediction, base_ids, novel_ids):
    target = np.asarray(target); prediction = np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError('One prediction for every original is required')
    allowed = set(base_ids) | set(novel_ids)
    if any(int(t) not in allowed for t in target):
        raise ValueError('Original target outside declared class roles')
    per_class = {}
    for c in np.unique(target):
        mask = target == c
        per_class[int(c)] = dict(originals=int(mask.sum()), correct=int((prediction[mask] == c).sum()),
                                 accuracy=float((prediction[mask] == c).mean()))
    def group(ids):
        support = [int(c) for c in ids if int(c) in per_class]
        mask = np.isin(target, list(ids))
        return dict(classes=support, originals=int(mask.sum()),
                    macro=float(np.mean([per_class[c]['accuracy'] for c in support])) if support else None,
                    micro=float((target[mask] == prediction[mask]).mean()) if mask.any() else None)
    b, n = group(base_ids), group(novel_ids)
    return dict(B_macro=b['macro'], N_macro=n['macro'], H_macro=harmonic(b['macro'], n['macro']),
                B_micro=b['micro'], N_micro=n['micro'], H_micro=harmonic(b['micro'], n['micro']),
                ordinary_top1=float((target == prediction).mean()) if len(target) else None,
                balanced_top1=float(np.mean([r['accuracy'] for r in per_class.values()])) if per_class else None,
                originals=len(target), abstentions=int((prediction == 0).sum()),
                base_support=b, novel_support=n, per_class=per_class)
