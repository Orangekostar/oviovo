"""Frozen unrounded screening choice and strict full-regression gates."""
import numpy as np
from .common import METRICS


def finite(*values):
    return all(v is not None and np.isfinite(v) for v in values)


def screen_gate(candidate,control,g1,*,changed,corrections_net,gt50_net,eps=1e-10):
    if not changed:return False
    delta=[]
    for cohort in ('replica_probe2','cf_probe2'):
        c,b,g=candidate[cohort],control[cohort],g1[cohort]
        if not finite(c['apall'],c['ap50'],c['miou'],b['apall'],g['apall'],g['ap50'],g['miou']):return False
        d=c['apall']-b['apall'];delta.append(d)
        if d<-.001-eps or c['apall']<g['apall']-.001-eps or c['ap50']<g['ap50']-.001-eps or c['miou']<g['miou']-.002-eps:return False
    return float(np.mean(delta))>=.0005-eps and (corrections_net>=1 or gt50_net>=1)


def final_gate(candidate,g1,d2,eps=1e-10):
    for cohort in ('replica8','scannet_cf18'):
        for metric in METRICS:
            x,y=candidate[cohort][metric],g1[cohort][metric]
            if not finite(x,y) or x<y-eps:return False
    c,b=candidate['scannet_cf18'],d2['scannet_cf18']
    return finite(c['apall'],c['ap50'],b['apall'],b['ap50']) and c['apall']>b['apall']+eps and c['ap50']>=b['ap50']-eps


def choose_screen(pools,methods,costs):
    ranks={}
    for order,method in enumerate(methods):
        delta=[];iou=[]
        for cohort in ('replica_probe2','cf_probe2'):
            c,g=pools[cohort][method]['metrics'],pools[cohort]['DQ01_G1']['metrics']
            if not finite(c['apall'],c['miou'],g['apall'],g['miou']):raise ValueError('screen choice requires finite metrics')
            delta.append(c['apall']-g['apall']);iou.append(c['miou']-g['miou'])
        ranks[method]=(min(delta),float(np.mean(delta)),float(np.mean(iou)),
            -costs[method]['logical_FULL_reads'],-costs[method]['logical_probe_heads'],-order)
    chosen=max(methods,key=lambda m:ranks[m]);return chosen,ranks
