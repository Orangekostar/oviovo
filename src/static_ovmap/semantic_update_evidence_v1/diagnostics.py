"""Exposed CONF12 diagnostics and seed17-only leave-one-family-out calibration."""
from pathlib import Path
import numpy as np
from .common import read, verified, write, gzip_json, save_arrays, SEEDS, METHODS, CONDITIONS
from .readers import column_index
from .metrics import recognition, predictions
from .calibration import recipes, candidate_scores, best_candidate, gate

def rank_record(s, target, ids, novel_ids, base_ids, defined=True):
    if not defined or not np.isfinite(s).all():return dict(defined=False,full_rank=None,novel_rank=None,
            top_all_base=None,top_novel_correct=None,strongest_competitor_base=None)
    order=np.argsort(-s,kind='stable');true=int(np.flatnonzero(ids==target)[0])
    novel=np.flatnonzero(np.isin(ids,novel_ids));nr=novel[np.argsort(-s[novel],kind='stable')]
    competitor=next(i for i in order if i!=true)
    return dict(defined=True,full_rank=int(np.flatnonzero(order==true)[0])+1,
                novel_rank=int(np.flatnonzero(nr==true)[0])+1,
                top_all_base=int(ids[order[0]]) in base_ids,
                top_novel_correct=int(ids[nr[0]])==target,
                strongest_competitor_base=int(ids[competitor]) in base_ids)

def run(binding):
    root=Path(binding['output_root']);meta=verified(root/'A/import.json')
    with np.load(root/'A/import.npz',allow_pickle=False) as a:
        scores=a['scores'];vectors=a['embeddings'];defined=a['defined'];ids=a['valid_ids'];text=a['text']
    objects=meta['objects'];target=np.array([o['class_id'] for o in objects]);families=np.array([o['family'] for o in objects])
    keys=[o['key'] for o in objects];base=meta['class_split']['base_ids'];novel=meta['class_split']['heldout_ids']
    columns=meta['columns'];rows=[];aux=[];main={};indices={};bs=column_index(columns,'B_FC',None)
    b=scores[:,bs,0];bz=vectors[:,bs,0]
    def metric(p,mask=None):
        mask=np.ones(len(target),bool) if mask is None else mask
        return recognition(target[mask],p[mask],base,novel)
    baseline=predictions(b,ids,defined[:,bs,0])
    for method in METHODS:
        for seed in ((None,) if method=='B_FC' else SEEDS):
            ci=column_index(columns,method,seed);p=predictions(scores[:,ci,0],ids,defined[:,ci,0])
            main[(method,seed)]=p;indices[(method,seed)]=ci
            rows.append(dict(method=method,seed=seed,profile='selected_static' if method=='B_FC' else 'matched1000',
                             prefix=8,condition='clean',metrics=metric(p),
                             supported_only=metric(p,np.array([o['supported'] for o in objects]))))
    for ci,column in enumerate(columns):
        for condition,name in enumerate(CONDITIONS):
            aux.append(dict(column=column,condition=name,metrics=metric(predictions(scores[:,ci,condition],ids,defined[:,ci,condition]))))
    ranks=[];errors=[];angles=[];fix_harm=[]
    for method in METHODS[1:]:
        for seed in SEEDS:
            ci=indices[method,seed];p=main[method,seed];good=(target==p);fc_good=(target==baseline)
            for group,mask in [('all',np.ones(len(target),bool)),('base',np.isin(target,base)),('novel',np.isin(target,novel))]:
                fix_harm.append(dict(method=method,seed=seed,group=group,originals=int(mask.sum()),
                                     corrected=int((mask&good&~fc_good).sum()),harmed=int((mask&~good&fc_good).sum()),
                                     both_correct=int((mask&good&fc_good).sum()),both_wrong=int((mask&~good&~fc_good).sum())))
            for oi,o in enumerate(objects):
                ok=bool(defined[oi,ci,0] and defined[oi,bs,0])
                theta=float(np.rad2deg(np.arccos(np.clip(np.dot(vectors[oi,ci,0].astype(np.float64),bz[oi].astype(np.float64)),-1,1)))) if ok else None
                angles.append(dict(key=o['key'],family=o['family'],base=o['base'],method=method,seed=seed,degrees=theta,defined=ok))
                errors.append(dict(key=o['key'],family=o['family'],target=int(target[oi]),base=o['base'],method=method,seed=seed,
                                   defined=bool(defined[oi,ci,0]),prediction=int(p[oi]),FC_prediction=int(baseline[oi]),
                                   corrected=bool(good[oi] and not fc_good[oi]),harmed=bool(fc_good[oi] and not good[oi]),angle_degrees=theta))
                if not o['base']:
                    r=rank_record(scores[oi,ci,0],target[oi],ids,novel,base,bool(defined[oi,ci,0]))
                    fc=rank_record(b[oi],target[oi],ids,novel,base,bool(defined[oi,bs,0]))
                    ranks.append(dict(key=o['key'],method=method,seed=seed,**r,FC=fc,
                                      full_rank_change=r['full_rank']-fc['full_rank'] if r['defined'] and fc['defined'] else None,
                                      novel_rank_change=r['novel_rank']-fc['novel_rank'] if r['defined'] and fc['defined'] else None))
    candidates=recipes(binding['task_specification']['A']);allpred=[];candidate_rows=[];aliases=[];flags_total=0
    base_mask=np.isin(ids,base);cap_cache={};score_aliases={};prediction_aliases={}
    # Logical candidates remain 110, even when their actual arrays coincide.
    for seed in SEEDS:
        ci=indices['MV_ATTN',seed];seedpred=[]
        for recipe in candidates:
            zero=dict(recipe,gamma=0.)
            cachekey=(seed,recipe['kind'],recipe['alpha'],recipe['angle_degrees'])
            if cachekey not in cap_cache:
                cap_cache[cachekey]=candidate_scores(zero,b,scores[:,ci,0],bz,vectors[:,ci,0],text,base_mask)
            s,flags=cap_cache[cachekey];s=s.copy();s[:,base_mask]-=recipe['gamma']
            candidate_defined=(defined[:,bs,0] if recipe['kind']=='MIX' and recipe['alpha']==0
                               else defined[:,ci,0] if recipe['kind']=='MIX' and recipe['alpha']==1
                               else defined[:,bs,0]&defined[:,ci,0])
            p=predictions(s,ids,candidate_defined);seedpred.append(p);flags_total+=int(flags.sum())
            candidate_rows.append(dict(seed=seed,recipe=recipe,metrics=metric(p),antipodal_fallbacks=int(flags.sum())))
            # Bytes include shape/dtype by construction, and equality is verified after hash lookup.
            from hashlib import sha256
            sh=sha256(s.tobytes()).hexdigest();ph=sha256(p.tobytes()).hexdigest()
            sk=(seed,sh);pk=(seed,ph)
            score_ref=score_aliases.get(sk);pred_ref=prediction_aliases.get(pk)
            if score_ref is not None and not np.array_equal(s,score_ref[1],equal_nan=True):raise ValueError('Score digest collision')
            if pred_ref is not None and not np.array_equal(p,pred_ref[1]):raise ValueError('Prediction digest collision')
            aliases.append(dict(seed=seed,candidate_id=recipe['candidate_id'],score_SHA256=sh,prediction_SHA256=ph,
                                exact_score_alias=score_ref[0] if score_ref else None,
                                exact_prediction_alias=pred_ref[0] if pred_ref else None))
            score_aliases.setdefault(sk,(recipe['candidate_id'],s));prediction_aliases.setdefault(pk,(recipe['candidate_id'],p))
        allpred.append(np.stack(seedpred))
    allpred=np.stack(allpred);oof=np.zeros((2,len(target)),np.int64);folds=[];family_rows=[]
    for f in sorted(set(families)):
        hold=families==f;train=~hold
        trainrows=[dict(recipe=r,metrics=metric(allpred[0,i],train)) for i,r in enumerate(candidates)]
        winner=best_candidate(trainrows);ri=next(i for i,r in enumerate(candidates) if r['candidate_id']==winner['recipe']['candidate_id'])
        folds.append(dict(held_family=f,selection_seed=17,applied_seeds=list(SEEDS),
                          training_originals=int(train.sum()),held_originals=int(hold.sum()),
                          recipe=winner['recipe'],training_metrics=winner['metrics'],
                          training_candidate_metrics=trainrows,held_keys=[keys[i] for i in np.flatnonzero(hold)]))
        for si,seed in enumerate(SEEDS):
            oof[si,hold]=allpred[si,ri,hold];simple=metric(oof[si],hold)
            for method in METHODS:
                p=baseline if method=='B_FC' else main[method,seed]
                m=metric(p,hold);delta=m['H_macro']-simple['H_macro'] if m['H_macro'] is not None and simple['H_macro'] is not None else None
                family_rows.append(dict(family=f,method=method,seed=seed,metrics=m,simple_metrics=simple,delta_H_macro=delta,
                                        class_support_differs_across_families=True))
    oofrows=[dict(seed=seed,method='LOFO_SIMPLE',metrics=metric(oof[si])) for si,seed in enumerate(SEEDS)]
    comparisons={}
    for method in ('S_LL','G_LL'):
        comparisons[method]=[]
        for si,seed in enumerate(SEEDS):
            m=metric(main[method,seed]);simple=oofrows[si]['metrics']
            fam=[r['delta_H_macro'] for r in family_rows if r['method']==method and r['seed']==seed]
            comparisons[method].append(dict(seed=seed,delta_H_macro=m['H_macro']-simple['H_macro'] if m['H_macro'] is not None and simple['H_macro'] is not None else None,
                 positive_families=sum(x is not None and x>0 for x in fam),defined_families=sum(x is not None for x in fam),
                 method_metrics=m,simple_metrics=simple))
    decision=gate(comparisons)
    # Explicit original-to-fold map makes OOF recomputation independent of row order.
    gzip_json(root/'A/crossfit.json.gz',dict(status='COMPLETE',folds=folds,family_rows=family_rows,OOF_metrics=oofrows,
          original_keys=keys,OOF_predictions=oof,logical_candidates=110,seed_selection='17_ONLY',candidate_recipes=candidates))
    gzip_json(root/'A/candidates.json.gz',dict(status='COMPLETE',rows=candidate_rows,aliases=aliases,
          logical_candidates=110,seed_realizations=220,unique_prediction_arrays=len(prediction_aliases),
          unique_score_arrays=len(score_aliases),alias_requires_actual_array_equality=True))
    gzip_json(root/'A/error_paths.json.gz',dict(status='COMPLETE',rows=errors,fix_harm=fix_harm))
    gzip_json(root/'A/ranks.json.gz',dict(status='COMPLETE',rows=ranks,novel_only_is_diagnostic=True))
    gzip_json(root/'A/angles.json.gz',dict(status='COMPLETE',rows=angles,vector_source='ORIGINAL_OUT_NPZ'))
    gzip_json(root/'A/auxiliary.json.gz',dict(status='COMPLETE',rows=aux,used_for_gate=False))
    save_arrays(root/'A/predictions.npz',candidate_predictions=allpred,OOF_predictions=oof,targets=target,
                original_keys=np.asarray(keys),families=families,valid_ids=ids)
    write(root/'A/decision.json',dict(decision,source_identity=binding['source_identity'],input_identity=meta['identity']))
    return write(root/'A/summary.json',dict(status='COMPLETE',domain='A_EXPLORE_EXPOSED',original_objects=len(objects),
          base_objects=meta['base_objects'],novel_objects=meta['novel_objects'],abstention_originals=meta['abstention_originals'],
          main_rows=rows,LOFO_rows=oofrows,decision=decision,fix_harm=fix_harm,
          original_score_reconstruction_max_absolute_error=meta['original_score_reconstruction_max_absolute_error'],
          computational_work=dict(new_head_calls=0,new_GPU_inferences=0,new_training_updates=0,new_RGB_encodings=0,
              CPU_candidate_predictions=220,CPU_CAP_text_matmuls=sum(k[1]=='CAP' for k in cap_cache),logical_CAP_realizations=110,
              original_NPZ_imports=len(objects),logical_candidates_per_seed=110,antipodal_fallback_realizations=flags_total),
          numerical_arithmetic='Frozen FP32 text/scores; FP64 angle geometry; clipped vector restored to FP32; no text renormalization',
          independent_confirmation=False))
