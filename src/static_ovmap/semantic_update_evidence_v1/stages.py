"""Conditional public phase contract and deterministic checkpoint/seed locks.

The observed A-negative transaction activates no B/C numerical producers.
This module never invents missing trained models or acquisition receipts.
"""
from pathlib import Path
from .common import write,verified,DependencyError

def checkpoint_paths(seeds=(17,29),methods=('FREE','RADIUS','EVIDENCE'),steps=(250,500,1000,1500,2000)):
    return [f'training/seed{s}/{m}/model_step{k:04d}.pt' for s in seeds for m in methods for k in steps]

def seed29_lock(seed17):
    return {**seed17,'seed':29,'selection_seed':17,'reselected':False}

def inactive(root,spec,decision):
    if decision['open']:raise DependencyError('Inactive phase dispatch requires a closed actual A gate')
    rows={p:dict(status='NOT_ACTIVATED_BY_A',reason=decision['status'],actual_models=[],actual_new_workers=0)
          for p in spec['phases'] if p.startswith(('B_','C_'))}
    for phase,row in rows.items():write(Path(root)/'phases'/(phase+'.json'),row)
    return rows
