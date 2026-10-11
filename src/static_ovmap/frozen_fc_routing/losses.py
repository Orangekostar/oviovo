"""Inherited base objective and pooled known-token BCE per original item."""
import torch
from torch.nn import functional as F
from static_ovmap.learned_object_readout.losses import base_loss

def item_loss(clean,corrupt,target,text,*,memberships,membership_enabled):
    common,parts=base_loss(clean,corrupt,target,text)
    logits=[];labels=[];counts=dict(positive_tokens=0,negative_tokens=0,unknown_tokens=0,local_observations=0)
    for r,y in zip((clean,corrupt),memberships):
        y=y[r['view_indices']];valid=r['local_valid'];known=valid&(y>=0)
        if bool((valid&~((y==-1)|(y==0)|(y==1))).any()):raise ValueError('Unknown membership code')
        counts['positive_tokens']+=int((known&(y==1)).sum());counts['negative_tokens']+=int((known&(y==0)).sum())
        counts['unknown_tokens']+=int((valid&(y<0)).sum());counts['local_observations']+=int(valid.sum())
        if bool(known.any()):logits.append(r['membership_logits'][known]);labels.append(y[known].float())
    counts['known_tokens']=counts['positive_tokens']+counts['negative_tokens']
    # Disabled logging cannot introduce zero-valued gradients/AdamW updates in membership.
    if logits:
        z=torch.cat(logits);ys=torch.cat(labels)
        bce=F.binary_cross_entropy_with_logits(z if membership_enabled else z.detach(),ys)
    else:bce=clean['membership_logits'].sum()*0 if membership_enabled else common.detach()*0
    loss=common+.1*bce if membership_enabled else common
    parts.update(counts,membership_loss=float(bce.detach()),membership_enabled=membership_enabled,
                 common=float(common.detach()),total=float(loss.detach()))
    return loss,parts
