"""TRAIN-only GT-best utility diagnosis; preserve all role coverage."""
from pathlib import Path
import numpy as np
from .common import verified,write
def run(root):
    root=Path(root);capture=verified(root/'capture.json');rows=[];actual={};coverage={}
    for entry in capture['units']:
        unit=verified(entry['path']);obj=unit['object'];actual.setdefault(unit['role'],{}).setdefault(obj['key'],unit['actual_views'])
        if unit['role']!='train':continue
        for g in unit['groups']:
            if g['prefix'] not in (4,8):continue
            actions=[unit['rows'][i] for i in g['indices']];keep=actions[0];best_ce=max(actions,key=lambda a:a['utility']);best_top=max(actions,key=lambda a:(int(a['correct']),a['margin']))
            rows.append(dict(role=unit['role'],key=obj['key'],family=obj['family'],condition=unit['condition_name'],seed=g['seed'],prefix=g['prefix'],actual_views=keep['actual_views'],logical_actions=len(actions),proved_noops=sum(bool(a.get('noop_proof')) for a in actions),best_CE_utility=best_ce['utility'],best_CE_kind=best_ce['kind'],best_CE_top1_correct=best_ce['correct'],best_Top1_correct=best_top['correct'],best_Top1_kind=best_top['kind'],KEEP_correct=keep['correct']))
    for role,objects in actual.items():coverage[role]=dict(originals=len(objects),actual_n4=sum(v['4']==4 for v in objects.values()),actual_n8=sum(v['8']==8 for v in objects.values()),short_banks=sum(v['8']<8 for v in objects.values()),denominator='ALL_ORIGINALS')
    return write(root/'action_bank.json',dict(status='COMPLETE',units=capture['units'],coverage=coverage,diagnostic_GT_best_rows=rows,GT_best_is_not_deployable=True,best_CE_is_not_best_Top1=True,utility_competition=160,utility_temperature=.07,diagnostic_competition=200,new_native_forwards=0))
