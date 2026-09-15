"""Ranking-only intervention on finalized COMMON-SOURCE owners."""
import numpy as np


def final_source_ranks(source_owners, active_ids):
    owners = np.asarray(source_owners)
    if owners.ndim != 1 or not np.issubdtype(owners.dtype, np.integer) or np.any(owners < 0):
        raise ValueError('expected one-dimensional nonnegative source owners')
    counts = np.bincount(owners)
    return [int(counts[i+1]) if i+1 < len(counts) else 0 for i in active_ids]
