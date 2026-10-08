"""Actual partition registry keys and exact complete probe-scene ordering."""

from static_ovmap.module_validation.native_capture import _array_digest


def partition_registry_key(owners):
    return _array_digest(owners)


def ordered_rows(rows,scenes,method):
    selected=[r for r in rows if r['method']==method]
    names=[r['scene'] for r in selected]
    if len(names)!=len(scenes) or len(set(names))!=len(names) or set(names)!=set(scenes) or any(r['status']!='COMPLETE' for r in selected):
        raise ValueError('pool requires every exact scene once with complete actual scoring')
    table={r['scene']:r for r in selected};return [table[s] for s in scenes]


def baseline_parity(fresh,parent,index):
    """Compare actual released events/states; only mask-directory roots differ."""
    import gzip,json
    from pathlib import Path
    from static_ovmap.cvpr_compact.evaluation import fraction_metrics
    import numpy as np
    def trace(score):
        path=Path(score['manifest']).with_name('trace.json.gz');index.identity(path)
        with gzip.open(path,'rt') as stream:value=json.load(stream)
        paths={'candidate_file','previous_score_owner','tp_score_owner','fp_score_owner'}
        events=[{k:(Path(v).name if k in paths and isinstance(v,str) else v)
                 for k,v in event.items()} for event in value['events']]
        return json.dumps({'events':events,'states':value['states']},sort_keys=True)
    checks={'official_view':fresh['view']==parent['view'],
            'confusion':np.array_equal(fresh['confusion'],parent['confusion']),
            'five_metrics':fraction_metrics(fresh['metrics'])==fraction_metrics(parent['metrics']),
            'released_events_and_terminal_states':trace(fresh)==trace(parent)}
    if not all(checks.values()):raise ValueError('fresh complete-map G1 baseline parity failed: '+str(checks))
    return checks


def preflight_parity(binding):
    import contextlib
    from pathlib import Path
    from static_ovmap.composition_study.object_evidence import owner_labels
    from static_ovmap.source_preserving_update.evaluation import partition_evaluator
    from .binding import load_scene
    from .common import canonical_digest,read,verified,write
    from .runner import require_freeze
    require_freeze(binding);root=Path(binding['output_root']);lock=verified(root/'predictions/summary.json')
    if lock['status']!='ALL_PREDICTIONS_LOCKED' or set(lock['scenes'])!=set(binding['scenes']):raise ValueError('global lock required before baseline annotations')
    path=root/'evaluation/baseline_parity.json';index=load_scene(binding,next(iter(binding['scenes']))).index
    key=canonical_digest({'global_predictions':lock['identity'],'operator':index.identity(__file__)})
    if path.exists():
        prior=verified(path)
        if prior['input_identity']!=key:raise ValueError('baseline parity dependency changed')
        return prior
    rows=[]
    for cohort,names in binding['cohorts'].items():
        scene=names[0];inputs=load_scene(binding,scene);payload=inputs.g1
        folder=root/'evaluation'/scene;folder.mkdir(parents=True,exist_ok=True)
        before=set(map(str,folder.glob('partitions/*/evaluation_cache/*/receipt.json')))
        evaluator=partition_evaluator(inputs,payload,binding,folder)
        with (folder/'G1_PARITY_PREFLIGHT.log').open('a') as stream,contextlib.redirect_stdout(stream):
            measured=evaluator.evaluate(owner_labels(payload),'SV00_G1','OFFICIAL_CURRENT_CLASS',payload.prediction_key)
        parent_path=binding['baseline_rows']['SU01_G1'][scene]['evaluation_receipt']
        rows.append({'cohort':cohort,'scene':scene,'fresh_receipt':measured['evaluation_receipt'],'parent_receipt':parent_path,
            'checks':baseline_parity(read(measured['evaluation_receipt']),read(parent_path),inputs.index),
            'physical_new_scorer_call':measured['evaluation_receipt'] not in before})
    return write(path,{'status':'BOTH_COHORT_BASELINES_VERIFIED_BEFORE_FINAL_SCORING','input_identity':key,'rows':rows})


def evaluate_scene(binding,scene):
    import contextlib,time
    from pathlib import Path
    import numpy as np
    from static_ovmap.composition_study.object_evidence import owner_labels
    from static_ovmap.cvpr_compact.evaluation import equivalent_scoring_input,fraction_metrics,trace_class_metrics
    from static_ovmap.cvpr_compact.protocol import validate_overlaps
    from static_ovmap.m2_reviewer_study.evaluation import official_view
    from static_ovmap.module_validation.scannet_study import load_prediction
    from static_ovmap.recovery_wave2.evaluation import scorer_context
    from static_ovmap.source_preserving_update.evaluation import partition_evaluator
    from .binding import load_scene
    from .common import METHODS,canonical_digest,verified,read,write
    from .runner import require_freeze
    require_freeze(binding);root=Path(binding['output_root'])
    summary=verified(root/'predictions/summary.json')
    if summary['status']!='ALL_PREDICTIONS_LOCKED' or set(summary['scenes'])!=set(binding['scenes']):
        raise ValueError('all four complete predictions must be locked before annotation access')
    lock=verified(root/'predictions'/scene/'receipt.json');inputs=load_scene(binding,scene)
    if lock['identity']!=summary['scenes'][scene]:raise ValueError('global prediction lock changed')
    dest=root/'evaluation'/scene/'receipt.json';index=inputs.index
    key=canonical_digest({'predictions':lock['identity'],'operator':index.identity(__file__),
        'adapter':index.identity(Path(__file__).parents[1]/'m2_reviewer_study/evaluation.py')})
    if dest.exists():
        prior=verified(dest)
        if prior['input_identity']!=key:raise ValueError('scoring dependency changed; invalidate affected descendants')
        for record in prior['outputs']:index.identity(record['path'],record)
        return prior
    begin=time.perf_counter();dest.parent.mkdir(parents=True,exist_ok=True)
    registries={};aliases={};rows=[];parity=None;outputs=[];checks={}
    first=scene in [names[0] for names in binding['cohorts'].values()]
    before=set(map(str,(dest.parent/'partitions').glob('*/evaluation_cache/*/receipt.json')))
    for method in (*METHODS,'REF_D2'):
        item=lock['methods'][method];payload=load_prediction(item['manifest'])
        if payload.prediction_key!=item['prediction_key'] or payload.geometry!=inputs.g1.geometry:
            raise ValueError('actual prediction identity changed')
        labels=owner_labels(payload);partition=partition_registry_key(payload.owner_ids)
        if partition not in registries:registries[partition]=partition_evaluator(inputs,payload,binding,dest.parent)
        evaluator=registries[partition];validate_overlaps(evaluator.namespace['overlaps'])
        if evaluator.minimum!=100 or set(evaluator.masks)!=set(labels):raise ValueError('minimum or whole-map registry changed')
        view={str(o):v for o,v in official_view(evaluator.owners,labels,evaluator.minimum).items()}
        ranks=dict(payload.instance_ranks)
        if set(ranks)!=set(labels) or any(f'{ranks[o]:.6f}'!=view.get(str(o),{'rank':'0.000000'})['rank'] for o in labels):
            raise ValueError('locked target-space current-class ranks differ')
        identity=canonical_digest({'partition':partition,'semantics':_array_digest(payload.semantic_labels),
            'ranks':payload.instance_ranks,'view':view,'context':scorer_context(evaluator.context)})
        row=None;kind='RELEASED_SCORER';fresh_required=method=='SV00_G1' and first
        if not fresh_required:
            if identity in aliases:row=aliases[identity];kind='EXACT_LOCAL_PARTITION_CLASSES_RANKS_CONTEXT_ALIAS'
            else:
                for baseline,reference in [('SU01_G1',inputs.g1),('SU00_D2',inputs.d2)]:
                    candidate=binding['baseline_rows'][baseline][scene];score=read(candidate['evaluation_receipt'])
                    if equivalent_scoring_input(payload,reference,evaluator.context,score['context'],view,score['view']):
                        row=candidate;kind='EXACT_PARENT_PARTITION_CLASSES_RANKS_CONTEXT_ALIAS';break
        if row is None:
            with (dest.parent/(method+'.log')).open('a') as stream,contextlib.redirect_stdout(stream):
                row=evaluator.evaluate(labels,method,'OFFICIAL_CURRENT_CLASS',payload.prediction_key)
        score=read(row['evaluation_receipt'])
        if score['status']!='COMPLETE' or score['view']!=view or scorer_context(score['context'])!=scorer_context(evaluator.context):
            raise ValueError('scorer context or actual view mismatch')
        if fresh_required:
            parent=read(binding['baseline_rows']['SU01_G1'][scene]['evaluation_receipt'])
            parity={'scene':scene,'fresh_receipt':row['evaluation_receipt'],'parent_receipt':binding['baseline_rows']['SU01_G1'][scene]['evaluation_receipt'],
                    'checks':baseline_parity(score,parent,index),'mask_root_normalization_only':True}
        matrix=np.asarray(score['confusion'],np.int64);valid_count=int(np.isin(evaluator.targets['gt_semantic'],evaluator.ids).sum())
        if matrix.shape!=(len(evaluator.ids)+1,)*2 or int(matrix.sum())!=valid_count:
            raise ValueError('whole-map semantic confusion omitted target errors')
        record=write(dest.parent/'rows'/(method+'.json'),{**row,'scene':scene,'method':method,
            'cohort':binding['scenes'][scene]['cohort'],'dataset':inputs.data['dataset'],
            'metrics':fraction_metrics(score['metrics']),'metric_unit':'FRACTION','rank_mode':'OFFICIAL_CURRENT_CLASS',
            'reuse_kind':kind,'scoring_input_identity':identity,'prediction_identity':payload.prediction_key,
            'owner_partition_digest':partition,'per_class':trace_class_metrics(score,index=index),
            'physical_scorer_new_call':(kind=='RELEASED_SCORER' and str(Path(row['evaluation_receipt'])) not in before)
                or (fresh_required and next(r for r in verified(root/'evaluation/baseline_parity.json')['rows'] if r['scene']==scene)['physical_new_scorer_call']),
            'runtime_overlaps':score['context']['runtime_overlaps']})
        rows.append(record);aliases[identity]=record
        checks[method]={'source_positive_owners':sorted(labels),'mask_registry_owners':sorted(evaluator.masks),
            'semantic_valid_targets':valid_count,'confusion_sum':int(matrix.sum()),'unknown_class_owners':[o for o,l in labels.items() if l==0]}
        for p in [Path(row['evaluation_receipt']),Path(score['manifest']),
                  Path(score['manifest']).with_name('matches.json.gz'),Path(score['manifest']).with_name('trace.json.gz')]:outputs.append(index.identity(p))
    result=write(dest,{'status':'COMPLETE','scene':scene,'rows':rows,'row_count':len(rows),'input_identity':key,
        'registry_checks':checks,'outputs':outputs,'baseline_parity':parity,'elapsed_seconds':time.perf_counter()-begin,
        'all_four_predictions_locked_before_annotations':True,'single_partition_for_AP_and_mIoU':True})
    index.write_memo(root/'inputs'/scene/'verifications.json');print('SCORED',scene,len(rows),'conditions',flush=True);return result


def evaluate(binding):
    import contextlib
    from pathlib import Path
    from static_ovmap.cvpr_compact.evaluation import fraction_metrics,released_pool_with_classes
    from static_ovmap.released_loader import load_released_module
    from .common import METHODS,ConsumptionIndex,read,write
    root=Path(binding['output_root']);preflight_parity(binding)
    receipts=[evaluate_scene(binding,s) for s in binding['scenes']]
    rows=[r for receipt in receipts for r in receipt['rows']];pools={};index=ConsumptionIndex(root/'evaluation/pool_verifications.json')
    for cohort,names in binding['cohorts'].items():
        score=read(binding['baseline_rows']['SU01_G1'][names[0]]['evaluation_receipt'])
        namespace=load_released_module(score['context']['evaluator']['path']);namespace['init']('Replica' if cohort=='replica_probe2' else 'Scannet200')
        output=root/'pools'/cohort;output.mkdir(parents=True,exist_ok=True);pools[cohort]={}
        for method in (*METHODS,'REF_D2'):
            subset=ordered_rows([r for r in rows if r['scene'] in names],names,method)
            ordered=[r['evaluation_identity'] for r in subset]
            alias=next((p for p in pools[cohort].values() if p['ordered_inputs']==ordered),None)
            if alias is not None and not (output/(method+'_per_class.json')).exists():
                # The shared released pool cache is keyed by ordered receipts.
                # Its captured per-class values are equally valid for this exact alias.
                write(output/(method+'_per_class.json'),{**read(alias['per_class_receipt']),'method':method})
            with (output/(method+'.log')).open('a') as stream,contextlib.redirect_stdout(stream):
                pooled,classes=released_pool_with_classes(output,subset,namespace,names,method,'OFFICIAL_CURRENT_CLASS',index)
            pools[cohort][method]=write(output/(method+'.json'),{**pooled,'method':method,'cohort':cohort,
                'metrics':fraction_metrics(pooled['metrics']),'metric_unit':'FRACTION',
                'per_class_receipt':str(output/(method+'_per_class.json')),'per_class_identity':classes['identity'],
                'exact_subset_recomputed':True,'parent_full_cohort_metrics_reused':False,
                'exact_ordered_local_alias':alias['method'] if alias else None})
    if len(rows)!=28 or sum(map(len,pools.values()))!=14:raise ValueError('incomplete fixed subset scoring')
    result=write(root/'result_store.json',{'status':'SCIENCE_COMPLETE','binding_identity':binding['identity'],
        'cohorts':binding['cohorts'],'scene_method_coverage':len(rows),'subset_pool_coverage':14,
        'scene_metrics':rows,'pooled_metrics':pools,'baseline_parity':[r['baseline_parity'] for r in receipts if r['baseline_parity']],
        'logical_rows':28,'physical_new_scene_scorer_calls':sum(r['physical_scorer_new_call'] for r in rows),
        'distinct_scene_scoring_receipts':len({r['evaluation_receipt'] for r in rows}),
        'previously_exposed':True,'independent_confirmation':False,'deployment':'N0_UNCHANGED'})
    index.write_memo(root/'evaluation/pool_verifications.json');return result
