"""Rebuild complete ordered-pool metrics from published compact PR/confusions."""
from pathlib import Path
import gzip
import inspect
import json
import numpy as np
from .common import REPO,METRICS,canonical_digest,unit_identity,cached_record,write,plain

def run(root,records,compact,folder):
    from .reporting import _state_ap
    from static_ovmap.m2_reviewer_study.evaluation import semantic_summary
    folder=Path(folder);path=folder/'pooled_PR_parity.json'
    key=unit_identity('compact_pooled_PR',dict(compact=compact,kernel=canonical_digest(inspect.getsource(_state_ap))),producers=('common','pr_audit'))
    old=cached_record(path,key)
    if old:return old
    scenes={};results=[]
    for descriptor in compact:
        file=REPO/descriptor['path'];records.index.identity(file,descriptor)
        with gzip.open(file,'rt') as stream:document=json.load(stream)
        scenes[descriptor['scene']]=dict(rows={r['method']:r for r in document['rows']},scores={r['score_identity']:r for r in document['scores']})
    calls=0
    for cohort,ordered in records.binding['cohorts'].items():
        for method,row in scenes[ordered[0]]['rows'].items():
            if row.get('status')=='MISSING':continue
            saved=[scenes[scene]['scores'][scenes[scene]['rows'][method]['score_identity']] for scene in ordered]
            banks=[{tuple(s[k] for k in ('distance_index','class_index','overlap_index')):s for s in score['PR_sufficient_statistics']} for score in saved]
            if any(set(bank)!=set(banks[0]) for bank in banks):raise ValueError('Compact pool class/IoU support differs')
            namespace=records.released(records.score(ordered[0],method))
            ap=np.full((len(namespace['dist_threshes']),len(namespace['CLASS_LABELS']),len(namespace['overlaps'])),np.nan)
            for state_key in banks[0]:
                states=[bank[state_key] for bank in banks]
                if len({s['class_label'] for s in states})!=1:raise ValueError('Compact pool class order differs')
                combined=dict(y_true=[v for s in states for v in s['y_true']],y_score=[v for s in states for v in s['y_score']],hard_false_negatives=sum(s['hard_false_negatives'] for s in states),has_gt=any(s['has_gt'] for s in states),has_pred=any(s['has_pred'] for s in states))
                value=_state_ap(combined);ap[state_key]=np.nan if value is None else value
            averages=plain(namespace['compute_averages'](ap));calls+=1
            matrix=sum((np.asarray(score['confusion'],np.int64) for score in saved),start=np.zeros_like(np.asarray(saved[0]['confusion'],np.int64)))
            metrics=dict(apall=averages['all_ap'],ap50=averages['all_ap_50%'],ap25=averages['all_ap_25%'],**semantic_summary(matrix))
            pool=records.pool(cohort,method)
            for metric in METRICS:
                actual=metrics[metric];expected=pool['metrics'][metric]
                if (actual is None)!=(expected is None) or actual is not None and abs(actual-expected)>1e-12:raise ValueError('Published compact ordered-pool metric parity differs: '+cohort+' '+method+' '+metric)
            results.append(dict(cohort=cohort,method=method,ordered_scenes=ordered,pool_identity=pool['identity'],metrics=metrics,all_five_parity=True))
    return write(path,dict(status='COMPLETE',input_identity=key,rows=results,absolute_tolerance=1e-12,original_compute_averages_calls=calls,new_scene_assign_calls=0,new_evaluate_matches_calls=0,new_head_calls=0,scope='PUBLISHED_COMPACT_PR_AND_SUMMED_CONFUSION_REBUILD_COMPLETE_ORDERED_POOLS;GEOMETRY_AND_MATCHER_REEXECUTION_REQUIRE_ORIGINAL_INPUTS'))
