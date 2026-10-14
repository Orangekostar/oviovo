"""Exact finite method lookup and original matches-only ordered aggregation."""
from pathlib import Path
import contextlib
import warnings
import numpy as np
from .common import (HISTORICAL,METRICS,verified,read_document,PathResolver,DependencyError,plain)
from static_ovmap.module_validation.scannet_study import load_prediction,project_values
from static_ovmap.released_loader import load_released_module
from static_ovmap.preservation_routing.scorer_compat import adapt
from static_ovmap.cvpr_compact.protocol import validate_overlaps
from static_ovmap.cvpr_compact.evaluation import per_class_metrics,fraction_metrics
from static_ovmap.m2_reviewer_study.evaluation import semantic_summary

class Records:
    def __init__(self,root,binding,index):
        self.root=Path(root);self.binding=binding;self.index=index;self.resolver=PathResolver(binding['path_map'])
        self.inventory=verified(self.root/'inventory.json');self.main=verified(self.root/'main_score.json') if (self.root/'main_score.json').exists() else dict(rows=[],pools={})
        self.predictions=verified(self.root/'main_predict.json') if (self.root/'main_predict.json').exists() else dict(records=[])
        self.cache={};self.matches={};self.namespaces={}

    def row(self,scene,method):
        native=HISTORICAL.get(method,method)
        rows=self.inventory['historical_rows']+self.inventory['FG_rows']+self.main['rows']
        result=next((r for r in rows if r['scene']==scene and r['method']==native and r['rank_mode']=='OFFICIAL_CURRENT_CLASS'),None)
        if result is None:raise DependencyError('Missing complete method scene: '+scene+' '+method)
        return self.resolver.rewrite(result)

    def pool(self,cohort,method):
        if method in HISTORICAL:bank=self.inventory['historical_pools']
        elif method.startswith('E') or method.startswith('J'):bank=self.main['pools']
        else:bank=self.inventory['FG_pools']
        result=bank.get(cohort,{}).get(method)
        if result is None:raise DependencyError('Missing full ordered pool: '+cohort+' '+method)
        return self.resolver.rewrite(result)

    def manifest(self,scene,method):
        if method in HISTORICAL:
            path=Path(self.binding['area_root'])/'predictions'/scene/'receipt.json'
            return self.resolver.rewrite(read_document(path,self.index))['predictions'][HISTORICAL[method]]
        if method.startswith('E') or method.startswith('J'):
            row=next((r for r in self.predictions['records'] if r['scene']==scene and r['method']==method),None)
            if row is None:raise DependencyError('Missing locked prediction: '+scene+' '+method)
            return row['manifest']
        path=Path(self.binding['fg_parent_root'])/'predictions'/scene/'receipt.json'
        return self.resolver.rewrite(read_document(path,self.index))['methods'][method]['manifest']

    def payload(self,scene,method):
        key=(scene,method)
        if key not in self.cache:
            path=self.manifest(scene,method);m=read_document(path,self.index,sealed=False)
            self.index.identity(Path(path).parent/m['arrays']['path'],m['arrays'])
            self.cache[key]=load_prediction(path)
        return self.cache[key]

    def score(self,scene,method):
        row=self.row(scene,method);result=self.resolver.rewrite(read_document(row['evaluation_receipt'],self.index,sealed=False))
        if result['identity']!=row['evaluation_identity']:raise ValueError('Original score identity differs')
        return result

    def projected(self,scene,method,score=None):
        score=score or self.score(scene,method);descriptor=score['context']['projection_arrays'];self.index.identity(descriptor['path'],descriptor)
        with np.load(descriptor['path'],allow_pickle=False) as arrays:owners=project_values(self.payload(scene,method).owner_ids,arrays['nearest'],arrays['matched'])
        return owners

    def targets(self,score):
        descriptor=score['context']['gt'];self.index.identity(descriptor['path'],descriptor)
        return np.load(descriptor['path'],allow_pickle=False)

    def released(self,score):
        context=score['context'];key=(context['evaluator']['sha256'],context['dataset'])
        if key not in self.namespaces:
            self.index.identity(context['evaluator']['path'],context['evaluator']);n=load_released_module(context['evaluator']['path'])
            n['init']('Replica' if context['dataset']=='Replica' else 'Scannet200');adapt(n);validate_overlaps(n['overlaps'])
            if n['overlaps'].tolist()!=context['runtime_overlaps']:raise ValueError('Released overlap order differs')
            self.namespaces[key]=n
        return self.namespaces[key]

    def match_input(self,scene,method):
        score=self.score(scene,method);path=Path(score['manifest']).with_name('matches.json.gz')
        if path not in self.matches:self.matches[path]=read_document(path,self.index,sealed=False)
        return score,self.matches[path]

    def aggregate(self,scenes,method):
        from .stability import namespace_matches
        matches={};matrix=None;first=None
        for slot,scene in enumerate(scenes):
            score,saved=self.match_input(scene,method);first=first or score
            if any(score['context'][k]!=first['context'][k] for k in ('valid_ids','class_names','runtime_overlaps')):raise ValueError('Ordered class/IoU context differs')
            matches.update(namespace_matches(saved,f'{slot}:{scene}'))
            m=np.asarray(score['confusion'],np.int64);matrix=m.copy() if matrix is None else matrix+m
        if first is None:raise DependencyError('No complete ordered matches')
        namespace=self.released(first)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);ap=namespace['evaluate_matches'](matches);averages=plain(namespace['compute_averages'](ap))
        metrics=dict(apall=averages['all_ap'],ap50=averages['all_ap_50%'],ap25=averages['all_ap_25%'],**semantic_summary(matrix))
        classes=per_class_metrics(averages,matrix,first['context']['valid_ids'],first['context']['class_names'],namespace['VALID_CLASS_IDS'],namespace['CLASS_LABELS'])
        return dict(status='COMPLETE',metrics=fraction_metrics(metrics),per_class=classes,confusion=matrix.tolist(),valid_GT_instance_classes=int(np.isfinite(ap[0,:,0]).sum()),matches_only_evaluate_calls=1,ordered_scenes=list(scenes),aggregation='ORIGINAL_RELEASED_MATCHES_ONLY_ORDERED_POOL')

def delta(after,before):
    return {m:None if after.get(m) is None or before.get(m) is None else after[m]-before[m] for m in METRICS}

def aggregate_scores(scores,index):
    """Original evaluation kernel; input scenes may repeat with independent names."""
    from .stability import namespace_matches
    if not scores:raise DependencyError('Missing complete ordered scoring inputs')
    first=scores[0];context=first['context'];index.identity(context['evaluator']['path'],context['evaluator'])
    namespace=load_released_module(context['evaluator']['path']);namespace['init']('Replica' if context['dataset']=='Replica' else 'Scannet200');adapt(namespace);validate_overlaps(namespace['overlaps'])
    matches={};matrix=np.zeros(np.asarray(first['confusion']).shape,np.int64)
    for slot,score in enumerate(scores):
        if any(score['context'][k]!=context[k] for k in ('valid_ids','class_names','runtime_overlaps')):raise ValueError('Ordered score context differs')
        saved=read_document(Path(score['manifest']).with_name('matches.json.gz'),index,sealed=False)
        matches.update(namespace_matches(saved,str(slot)));matrix+=np.asarray(score['confusion'],np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);ap=namespace['evaluate_matches'](matches);averages=plain(namespace['compute_averages'](ap))
    metrics=dict(apall=averages['all_ap'],ap50=averages['all_ap_50%'],ap25=averages['all_ap_25%'],**semantic_summary(matrix))
    return dict(metrics=fraction_metrics(metrics),per_class=per_class_metrics(averages,matrix,context['valid_ids'],context['class_names'],namespace['VALID_CLASS_IDS'],namespace['CLASS_LABELS']),confusion=matrix.tolist(),ordered_inputs=[s['identity'] for s in scores],matches_only_evaluate_calls=1)
