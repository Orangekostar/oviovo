"""Atomic full resume generations; only completed optimizer updates are saved."""
from pathlib import Path
import os
from static_ovmap.learned_object_readout.training import save_checkpoint,restore_checkpoint

def save_resume(directory,head,optimizer,*,seed,method,completed,input_identity,validations,**metadata):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if int(completed)!=completed or completed<0:raise ValueError('Invalid completed update count')
    last=directory/'last.pt';previous=directory/'last.previous.pt'
    if last.exists():os.replace(last,previous)
    save_checkpoint(last,head,optimizer,seed=seed,branch=method,step=int(completed),input_identity=input_identity,validations=validations,**metadata)
    return last

def restore_resume(directory,head,optimizer,*,input_identity):
    directory=Path(directory);last=directory/'last.pt'
    if not last.exists():last=directory/'last.previous.pt'
    return restore_checkpoint(last,head,optimizer,input_identity=input_identity)
