"""Historical evaluator-export compatibility bridge, never deployment eligibility."""
from pathlib import Path
import re

import numpy as np


def load_historical_registry(manifest,native_projected_owners,historical_semantic):
    manifest=Path(manifest)
    owners=np.asarray(native_projected_owners);semantic=np.asarray(historical_semantic)
    if owners.shape!=semantic.shape or owners.ndim!=1:raise ValueError('historical domains differ')
    ids=[];labels=[];scores=[];areas=[];registry_semantic=np.zeros(len(owners),dtype=np.int64)
    for line in manifest.read_text().splitlines():
        name,label,score=line.split();match=re.fullmatch(r'inst-(\d+)_label-(\d+)\.npy',name)
        if not match or int(label)!=int(match.group(2)):raise ValueError('historical candidate identity missing')
        owner=int(match.group(1));mask=np.load(manifest.parent/name,allow_pickle=False).ravel()
        if owner in ids or mask.dtype!=bool or not np.array_equal(mask,owners==owner):
            raise ValueError('historical emitted mask/owner identity differs')
        if not np.all(semantic[mask]==int(label)):raise ValueError('historical semantic/readout registry differs')
        ids.append(owner);labels.append(int(label));scores.append(score);areas.append(int(mask.sum()))
        registry_semantic[mask]=int(label)
    areas=np.asarray(areas,dtype=np.int64);labels=np.asarray(labels,dtype=np.int64)
    full=np.zeros(len(ids),dtype=float)
    for label in np.unique(labels):
        keep=labels==label;full[keep]=areas[keep]/areas[keep].max()
    if [f'{x:.6f}' for x in full]!=scores:raise ValueError('released original area score cannot be reconstructed exactly')
    return {'owner_ids':np.array(ids,dtype=np.int64),'labels':labels,'scores_serialized':scores,
        'scores_full_precision':full,'original_projected_areas':areas,'registry_semantic':registry_semantic,
        'positive_semantics_outside_emitted_registry':int(((semantic>0)&(registry_semantic==0)).sum())}
