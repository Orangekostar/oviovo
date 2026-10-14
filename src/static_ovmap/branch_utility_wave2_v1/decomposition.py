"""Compose a supplied base with base-independent native local vectors/routes."""
import torch
from static_ovmap.frozen_fc_routing.models import correct

def compose(base,result,eta=.5):
    if not result.get('group_count',0):return base
    return correct(base,result['group_vectors']-base,result['alpha'],eta)[0]

def local_statistics(result):
    base=result['base']
    if not result.get('group_count',0):return 0.,torch.zeros_like(base),1.,0
    a=result['alpha'];return float(a.sum(-1).mean()),(a@result['group_vectors']).mean(0),float(result['null'].mean()),int(result['group_count'])

def effects(h00,h10,h01,h11):
    return dict(base=.5*((h10-h00)+(h11-h01)),local=.5*((h01-h00)+(h11-h10)),interaction=h11-h10-h01+h00,total=h11-h00)
