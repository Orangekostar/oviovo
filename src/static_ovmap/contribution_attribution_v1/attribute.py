"""Full positive-owner error paths and five-metric factorial attribution."""
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .common import (ConsumptionIndex,verified,read_document,write,unit_identity,cached_record,family,METRICS,DependencyError)
from .binding import load_binding
from .records import Records,delta
from .history import factorial
from .inventory import scene_sources
from .sources import source_probabilities,combine
from .prediction import _J_readouts
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.branch_utility_wave2_v1.conveyor import geometry_matches
from static_ovmap.paired_evidence_study.residuals import pool,simple_dependence

def path_bin(match,old,new,new_f):
    truth=match['GT_class'];iou=match['IoU']
    if truth is None or iou is None:return 'NO_VALID_GT_REFERENCE'
    if iou<=.25:return 'GEOMETRY_BELOW_025'
    if iou<=.5:return 'GEOMETRY_025_TO_050'
    if old==truth and new!=truth:return 'HARMED'
    if old!=truth and new==truth:return 'CORRECTED'
    if new_f==truth and new!=truth:return 'F_CORRECT_FUSION_WRONG'
    if new!=truth:return 'FINAL_STILL_WRONG'
    return 'REMAINED_CORRECT'

def probability_measures(p,ids,truth):
    if p is None or truth not in ids:return dict(NLL=None,Brier=None,margin=None,correct=None)
    p=np.asarray(p,np.float64);target=ids.index(truth);other=p.copy();other[target]=-np.inf;one=np.zeros_like(p);one[target]=1
    return dict(NLL=float(-np.log(max(float(p[target]),1e-12))),Brier=float(np.sum((p-one)**2)),margin=float(p[target]-other.max()),correct=int(ids[int(p.argmax())])==truth)

def _scope_effects(cells,axis):
    out={}
    for m in METRICS:
        values=[c['metrics'][m] for c in cells]
        out[m]=factorial(*values) if all(v is not None for v in values) else None
    return dict(cells=cells,effects=out,axis=axis)

def _gain(records,binding):
    result=[];calls=0
    definitions=[('ER',None,('A0','A1','A2','A3'),'E_R')]
    definitions += [('S_FUSION',seed,('J0_FC8_FONLY',namej,'FG0_FC8',name),'S_FUSION') for seed,namej,name in [(17,'J1_FG2_S17_FONLY','FG2_SURFACE'),(29,'J2_FG2_S29_FONLY','S29_FG2_SURFACE')]]
    for cohort,scenes in binding['cohorts'].items():
        for kind,seed,methods,axis in definitions:
            cells=[dict(method=m,metrics=records.pool(cohort,m)['metrics']) for m in methods]
            pooled=_scope_effects(cells,axis)
            classes=[]
            for m in methods:
                row=records.pool(cohort,m);path=row.get('per_class_receipt')
                if path:
                    details=read_document(path,records.index);classrows=details.get('classes',details.get('rows',details.get('per_class')))
                else:
                    aggregate=records.aggregate(scenes,m);calls+=1;classrows=aggregate['per_class']
                if classrows is None:raise DependencyError('Missing per-class pool '+cohort+' '+m)
                classes.append({r['class_id']:r for r in classrows})
            perclass=[]
            for label in classes[0]:
                metrics=[]
                for c in classes:
                    r=c[label];metrics.append(dict(metrics={m:r[m] if m.startswith('ap') else r['iou' if m=='miou' else 'accuracy'] for m in METRICS}))
                perclass.append(dict(class_id=label,class_name=classes[0][label]['class_name'],**_scope_effects(metrics,axis)))
            familyrows=[]
            for f in sorted({family(s) for s in scenes}):
                ordered=[s for s in scenes if family(s)==f];values=[dict(method=m,**records.aggregate(ordered,m)) for m in methods];calls+=4
                familyrows.append(dict(family=f,scenes=ordered,**_scope_effects(values,axis)))
            scenerows=[dict(scene=s,**_scope_effects([dict(method=m,metrics=records.row(s,m)['metrics']) for m in methods],axis)) for s in scenes]
            result.append(dict(cohort=cohort,kind=kind,seed=seed,methods=methods,pooled=pooled,per_class=perclass,per_family=familyrows,per_scene=scenerows,
                               scope='S_FIXED_ELIGIBLE_G1_DOMAIN' if seed else 'HISTORICAL_FIXED_SURFACE_E_R_SYSTEM'))
    return result,calls

def run(root):
    from . import recovery
    from .lineage import verify as verify_lineage
    from .recovery import official_entries
    root=Path(root);binding=load_binding(root);index=ConsumptionIndex();records=Records(root,binding,index)
    recovered=recovery.run(root);main=verified(root/'main_score.json');surface=verified(root/'surface.json')
    lineage=verify_lineage(root,records)
    key=unit_identity('attribute',dict(main=main['identity'],inventory=records.inventory['identity'],surface=surface['identity'],recovery=recovered['identity'],lineage=lineage['identity']),
                      producers=('common','records','recovery','history','attribute','lineage'),cohorts=binding['cohorts'])
    previous=cached_record(root/'attribute.json',key)
    if previous:return previous
    srows=[];erows=[];source_rows=[];availability=[];denominators=[];d2proof=[]
    for cohort,scenes in binding['cohorts'].items():
        for scene in scenes:
            data,sources=scene_sources(binding,scene,'REGRESSION',index);ids=list(map(int,sources['N']['valid_ids']));readout=verified(root/'main_predict'/scene/'readouts.json')
            a0=records.payload(scene,'A0');labels0=owner_labels(a0);score=records.score(scene,'A0');gt=records.targets(score);owners=records.projected(scene,'A0',score);matches=geometry_matches(owners,gt,score['context']['valid_ids'])
            outputs={m:owner_labels(records.payload(scene,m)) for m in ('A0','A1','E0','E1','E2','E3','E4','E5','E6','E7','E8') if m!='E8' or verified(root/'weight_lock.json')['weight'] is not None}
            official={}
            for method in ('A0','A1','E6','E7','FG0_FC8','FG2_SURFACE','S29_FG2_SURFACE'):
                scored=records.score(scene,method);trace=read_document(Path(scored['manifest']).with_name('trace.json.gz'),index,sealed=False)
                entries=official_entries(trace,.5)
                official[method]=dict(TP=set(entries['definite_TP_owners']),FP=Counter(entries['definite_FP_score_entry_owners']))
            def events(before,after,owner):
                return dict(official_event_scope='OFFICIAL_IOU50_NONEXCLUSIVE_FLAGS_NOT_EXCLUSIVE_OWNER_PARTITION',
                            old_definite_TP=owner in official[before]['TP'],new_definite_TP=owner in official[after]['TP'],
                            old_definite_FP_score_entries=official[before]['FP'][owner],new_definite_FP_score_entries=official[after]['FP'][owner])
            for method in [m for m in outputs if m.startswith('E')]:
                if not np.array_equal(records.payload(scene,method).owner_ids,a0.owner_ids):raise ValueError('E changed geometry/owner support')
            pattern_counts=Counter()
            for owner in sorted(labels0):
                available=readout['source_availability'][str(owner)];pattern=''.join(str(int(available[n])) for n in ('N','Q','F'));pattern_counts[pattern]+=1
                probabilities=source_probabilities(sources,data['temperatures'],ids,owner);tops={n:None if p is None else ids[int(p.argmax())] for n,p in probabilities.items()}
                match=matches.get(owner,dict(GT_class=None,GT_instance=None,IoU=None,matched_IoU50=False,owner_points=0))
                stratum='ALL_AGREE' if len(set(tops.values()))==1 else 'NQ_AGREE_F_DIFF' if tops['N']==tops['Q'] and tops['N']!=tops['F'] else 'OTHER_DISAGREEMENT'
                measures={n:probability_measures(p,ids,match['GT_class']) for n,p in probabilities.items()}
                oldp=readout['methods']['E6'][str(owner)]['probabilities'];newp=readout['methods']['E7'][str(owner)]['probabilities'];old_measure=probability_measures(oldp,ids,match['GT_class']);new_measure=probability_measures(newp,ids,match['GT_class'])
                if pattern=='111':
                    source_rows.append(dict(cohort=cohort,family=family(scene),scene=scene,owner=owner,stratum=stratum,comparison_support='ALL_THREE_AVAILABLE',**match,
                                            source_measures=measures,NQ_common_wrong=tops['N']==tops['Q'] and tops['N']!=match['GT_class'],F_corrects_NQ=tops['F']==match['GT_class'] and tops['N']==tops['Q'] and tops['N']!=match['GT_class'],
                                            fusion_corrected=match['matched_IoU50'] and not old_measure['correct'] and bool(new_measure['correct']),fusion_harmed=match['matched_IoU50'] and bool(old_measure['correct']) and not new_measure['correct'],E6=old_measure,E7=new_measure))
                for module,before,after in [('E_SOURCE','A0','A1'),('E_GROUP','E6','E7')]:
                    truth=match['GT_class'];old=outputs[before][owner];new=outputs[after][owner]
                    erows.append(dict(module=module,cohort=cohort,family=family(scene),scene=scene,owner=owner,eligible=True,availability_pattern=pattern,stratum=stratum,old_label=old,new_label=new,F_top1=tops['F'],
                                      path=path_bin(match,old,new,tops['F']),final_label_changed=old!=new,margin_change=None if module=='E_SOURCE' or old_measure['margin'] is None or new_measure['margin'] is None else new_measure['margin']-old_measure['margin'],
                                      margin_pair='E6_E7' if module=='E_GROUP' else 'NOT_COMPARABLE_NATIVE_RETAINED_SCORE',**events(before,after,owner),**match))
            for pattern in (format(i,'03b') for i in range(8)):
                members=[o for o in labels0 if ''.join(str(int(readout['source_availability'][str(o)][n])) for n in ('N','Q','F'))==pattern]
                availability.append(dict(cohort=cohort,scene=scene,pattern=pattern,owner_count=len(members),point_count=sum(matches.get(o,{}).get('owner_points',0) for o in members),
                                         methods={m:dict(unknown_owners=sum(outputs[m][o]==0 for o in members),unknown_points=sum(matches.get(o,{}).get('owner_points',0) for o in members if outputs[m][o]==0)) for m in outputs if m.startswith('E')}))
            base,eligible,jids,bank,lock=_J_readouts(binding,scene,index);eligible=set(eligible);base_labels=owner_labels(base);base_score=records.score(scene,'FG0_FC8');g1owners=records.projected(scene,'FG0_FC8',base_score);g1matches=geometry_matches(g1owners,gt,base_score['context']['valid_ids'])
            if list(map(int,jids))!=ids or base_score['context']['gt']['sha256']!=score['context']['gt']['sha256']:raise ValueError('S and E vocabulary or GT differs')
            for seed,jname,name in [(17,'J1_FG2_S17_FONLY','FG2_SURFACE'),(29,'J2_FG2_S29_FONLY','S29_FG2_SURFACE')]:
                new_payload=records.payload(scene,name);new_labels=owner_labels(new_payload)
                if not np.array_equal(base.owner_ids,new_payload.owner_ids) or set(new_labels)!=set(base_labels):raise ValueError('S fixed G1 domain differs')
                for owner in sorted(base_labels):
                    match=g1matches.get(owner,dict(GT_class=None,GT_instance=None,IoU=None,matched_IoU50=False,owner_points=0));old=base_labels[owner];new=new_labels[owner];truth=match['GT_class']
                    if owner not in eligible:
                        if new!=old:raise ValueError('Noneligible S owner label changed')
                        oldf=newf=None;oldmeasure=newmeasure=dict(margin=None)
                    else:
                        oldv=bank['J0_FC8_FONLY'][owner];newv=bank[jname][owner];oldf=int(jids[int(oldv.argmax())]);newf=int(jids[int(newv.argmax())])
                        probs=source_probabilities(sources,data['temperatures'],ids,owner);temps=data['temperatures'];oldp=combine(dict(probs,F=softmax(oldv.astype(np.float64)/temps['F'])),('N','Q','F'),'PARENT_D2');newp=combine(dict(probs,F=softmax(newv.astype(np.float64)/temps['F'])),('N','Q','F'),'PARENT_D2')
                        source={n:None if probs[n] is None else np.asarray(sources[n]['objects'][str(owner)]['scores'],np.float64) for n in ('N','Q','F')}
                        for vector,label in [(oldv,old),(newv,new)]:
                            source['F']=vector;p=simple_dependence(source,temps,pool(source,temps,('N','Q','F')),'PE_D2_GROUPED')
                            if p is None or int(jids[int(p.argmax())])!=label:raise ValueError('Existing D2 endpoint differs from exact eligible new-F replacement')
                        oldmeasure=probability_measures(oldp,ids,truth);newmeasure=probability_measures(newp,ids,truth)
                    srows.append(dict(module='S',cohort=cohort,family=family(scene),scene=scene,owner=owner,seed=seed,eligible=owner in eligible,old_label=old,new_label=new,old_F_top1=oldf,new_F_top1=newf,
                                     path=path_bin(match,old,new,newf) if owner in eligible else 'NOT_ELIGIBLE',F_wrong_to_right=match['matched_IoU50'] and oldf!=truth and newf==truth,F_right_to_wrong=match['matched_IoU50'] and oldf==truth and newf!=truth,final_label_changed=old!=new,
                                     margin_change=None if oldmeasure['margin'] is None or newmeasure['margin'] is None else newmeasure['margin']-oldmeasure['margin'],**events('FG0_FC8',name,owner),**match))
                d2proof.append(dict(cohort=cohort,scene=scene,seed=seed,D2_method=name,exact_source_operator_and_labels=True,fixed_noneligible_labels=True,eligible_count=len(eligible),GT_used_for_proof=False))
            denominators.append(dict(cohort=cohort,scene=scene,E_all_owners=len(labels0),E_evaluable_owners=sum(m['owner_points']>=100 for m in matches.values()),E_point_coverage=sum(m['owner_points'] for m in matches.values()),
                                     S_all_owners=len(base_labels),S_evaluable_owners=sum(m['owner_points']>=100 for m in g1matches.values()),S_eligible_owners=len(eligible),S_eligible_evaluable=sum(g1matches.get(o,{}).get('owner_points',0)>=100 for o in eligible),S_point_coverage=sum(m['owner_points'] for m in g1matches.values()),S_eligible_point_coverage=sum(g1matches.get(o,{}).get('owner_points',0) for o in eligible),target_points=len(gt)))
            print('ATTRIBUTION_SCENE',scene,flush=True);records.cache.clear()
    gains,calls=_gain(records,binding);summaries=[]
    for cohort in binding['cohorts']:
        for stratum in ('ALL_AGREE','NQ_AGREE_F_DIFF','OTHER_DISAGREEMENT','ALL'):
            supported=[r for r in source_rows if r['cohort']==cohort and r['matched_IoU50'] and (stratum=='ALL' or r['stratum']==stratum)]
            summaries.append(dict(cohort=cohort,stratum=stratum,objects=len(supported),families=len({r['family'] for r in supported}),
                                   source_means={n:{m:float(np.mean([r['source_measures'][n][m] for r in supported])) if supported else None for m in ('correct','NLL','Brier','margin')} for n in ('N','Q','F')},
                                   NQ_common_errors=sum(r['NQ_common_wrong'] for r in supported),F_corrects_NQ=sum(r['F_corrects_NQ'] for r in supported),fusion_corrected=sum(r['fusion_corrected'] for r in supported),fusion_harmed=sum(r['fusion_harmed'] for r in supported)))
    index.write_memo(root/'attribute/verifications.json')
    gain_record=write(root/'gain_attribution.json',dict(status='COMPLETE',rows=gains,matches_only_pool_calls=calls,metric_unit='FRACTION',mean_marginals_close_total=True,not_natural_causal_shares=True))
    return write(root/'attribute.json',dict(status='COMPLETE' if main['status']=='COMPLETE' and recovered['status']=='COMPLETE' else 'PARTIAL_DEPENDENCY',input_identity=key,S_owner_rows=srows,E_owner_rows=erows,source_owner_rows=source_rows,
                 source_summaries=summaries,availability_patterns=availability,denominators=denominators,D2_J_alias_proof=d2proof,gain_attribution= gain_record['identity'],matches_only_pool_calls=calls,NLL_clamp=1e-12,
                 Brier_definition='SUM_OVER_FULL_VOCABULARY_NOT_DIVIDED_BY_CLASSES',source_support='SAME_ALL_THREE_AVAILABLE_IoU>0.5_FOR_CORRECTNESS',descriptive_matching='MAX_GEOMETRIC_IOU_MULTI_TO_ONE_NOT_OFFICIAL_TP',new_model_forwards=0,new_scene_assign_calls=0))
