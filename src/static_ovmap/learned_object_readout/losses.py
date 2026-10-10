"""Common base loss and AUX-only annotation supervision sidecars."""
import itertools

import torch
from torch.nn import functional as F


def auxiliary_losses(results, memberships):
    """One original item; at most128 same-site pairs across its variants.

    Pairs never cross clean/corrupt site banks. Sort by site, actual view IDs,
    then variant as a deterministic tie break; unknown membership is ignored.
    """
    if isinstance(results, dict):
        results, memberships = [results], [memberships]
    zero = results[0]['reliability'].sum() * 0
    logits, targets, pairs = [], [], []
    counts = dict(positive_tokens=0, negative_tokens=0, unknown_tokens=0, correspondence_pairs=0)
    for variant, (result, membership) in enumerate(zip(results, memberships)):
        selected = result.get('view_indices')
        if selected is not None:
            membership = membership[selected]
        valid = result['local_valid']
        known = valid & (membership >= 0)
        counts['positive_tokens'] += int((known & (membership == 1)).sum())
        counts['negative_tokens'] += int((known & (membership == 0)).sum())
        counts['unknown_tokens'] += int((valid & (membership < 0)).sum())
        if bool(known.any()):
            logits.append(result['reliability'][known]); targets.append(membership[known].float())
        frame_ids = result.get('frame_ids', torch.arange(valid.shape[0], device=valid.device)).tolist()
        for site in range(valid.shape[1]):
            visible = torch.nonzero(valid[:, site] & (membership[:, site] == 1), as_tuple=False).flatten().tolist()
            for v, w in itertools.combinations(sorted(visible, key=lambda i: frame_ids[i]), 2):
                pairs.append((site, frame_ids[v], frame_ids[w], variant, v, w))
    membership_loss = F.binary_cross_entropy_with_logits(torch.cat(logits), torch.cat(targets)) if logits else zero
    chosen = sorted(pairs)[:128]
    terms = []
    for site, _, _, variant, v, w in chosen:
        h = results[variant]['local_hidden']
        terms.append(1 - F.cosine_similarity(h[v, site], h[w, site], dim=0))
    correspondence = torch.stack(terms).mean() if terms else zero
    counts['correspondence_pairs'] = len(chosen)
    counts.update(membership_loss=float(membership_loss.detach()), correspondence_loss=float(correspondence.detach()))
    return .10 * membership_loss + .05 * correspondence, counts


def base_loss(clean, corrupt, target_index, base_text):
    target = torch.as_tensor([target_index], device=base_text.device, dtype=torch.long)
    ce_clean = F.cross_entropy((clean['embedding'] @ base_text.T / .07)[None], target)
    ce_corrupt = F.cross_entropy((corrupt['embedding'] @ base_text.T / .07)[None], target)
    consistency = 1 - F.cosine_similarity(corrupt['embedding'], clean['embedding'].detach(), dim=0)
    loss = .5*ce_clean + .5*ce_corrupt + .10*consistency
    return loss, dict(ce_clean=float(ce_clean.detach()), ce_corrupt=float(ce_corrupt.detach()),
                      consistency=float(consistency.detach()))
