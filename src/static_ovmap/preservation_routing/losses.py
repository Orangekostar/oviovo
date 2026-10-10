"""Original-item means, with targets confined to the training loss API."""
import itertools
import torch
from torch.nn import functional as F
from static_ovmap.learned_object_readout.losses import base_loss

def preservation(clean,corrupt,teacher,correct):
    return float(bool(correct))*(.5*(1-F.cosine_similarity(clean,teacher.detach(),dim=0))+
                                 .5*(1-F.cosine_similarity(corrupt,teacher.detach(),dim=0)))

def auxiliary(results,targets,membership_enabled,correspondence_enabled):
    zero=results[0]['membership_logits'].sum()*0
    logits=[];labels=[];pairs=[]
    counts=dict(positive_tokens=0,negative_tokens=0,unknown_tokens=0,correspondence_pairs=0)
    for variant,(r,target) in enumerate(zip(results,targets)):
        target=target[r['view_indices']];valid=r['local_valid'];known=valid&(target>=0)
        counts['positive_tokens']+=int((known&(target==1)).sum())
        counts['negative_tokens']+=int((known&(target==0)).sum())
        counts['unknown_tokens']+=int((valid&(target<0)).sum())
        if known.any():logits.append(r['membership_logits'][known]);labels.append(target[known].float())
        frames=r['frame_ids'].tolist()
        if correspondence_enabled:
            for site in range(valid.shape[1]):
                visible=torch.nonzero(valid[:,site]&(target[:,site]==1),as_tuple=False).flatten().tolist()
                for v,w in itertools.combinations(sorted(visible,key=lambda i:frames[i]),2):
                    pairs.append((site,frames[v],frames[w],variant,v,w))
    bce=F.binary_cross_entropy_with_logits(torch.cat(logits),torch.cat(labels)) if logits else zero
    selected=sorted(pairs)[:128]
    terms=[1-F.cosine_similarity(results[t]['local_hidden'][v,s],results[t]['local_hidden'][w,s],dim=0)
           for s,_,_,t,v,w in selected]
    corr=torch.stack(terms).mean() if terms else zero
    counts.update(correspondence_pairs=len(selected),membership_loss=float(bce.detach()),correspondence_loss=float(corr.detach()))
    return (.1*bce if membership_enabled else zero)+(.05*corr if correspondence_enabled else zero),counts

def item_loss(clean,corrupt,target,text,teacher,teacher_correct,keep,*,memberships=None,config=None):
    common,components=base_loss(clean,corrupt,target,text)
    retain=preservation(clean['embedding'],corrupt['embedding'],teacher,teacher_correct)
    loss=common+(retain if keep else 0)
    components.update(common=float(common.detach()),preservation=float(retain.detach()),
                      teacher_correct=int(teacher_correct),teacher_items=1,preservation_enabled=keep)
    if memberships is not None:
        extra,counts=auxiliary([clean,corrupt],memberships,config['membership_loss'],config['correspondence_loss'])
        loss=loss+extra;components.update(counts)
    components['total']=float(loss.detach())
    return loss,components
