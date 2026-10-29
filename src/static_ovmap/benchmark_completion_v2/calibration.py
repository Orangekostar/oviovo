"""Equal-budget calibration on the immutable exposed original-object domain."""
import numpy as np
from static_ovmap.semantic_update_evidence_v1.metrics import predictions, recognition
from static_ovmap.semantic_update_evidence_v1.calibration import angle_cap, best_candidate
from .common import SEEDS, METHODS, Path, verified, read, write, save_arrays, source_identity


def nmix_scores(alpha, b_scores, z_scores, b, z, text):
    flags = np.zeros(len(b_scores), bool)
    if alpha == 0:
        return b_scores.copy(), flags
    if alpha == 1:
        return z_scores.copy(), flags
    if not 0 < alpha < 1:
        raise ValueError('NMIX alpha outside [0,1]')
    combined = (1-alpha)*b.astype(np.float64) + alpha*z.astype(np.float64)
    norms = np.linalg.norm(combined, axis=-1)
    flags = norms <= 1e-12
    safe = np.where(flags, 1., norms)
    vector = (combined / safe[..., None]).astype(z.dtype)
    vector[flags] = b[flags]
    return vector @ text.T, flags


def recipes(spec):
    rows = []
    for kind in spec['kinds']:
        strengths = spec['tau_grid'] if kind == 'CAP' else spec['alpha_grid']
        for value in strengths:
            for gamma in spec['gamma_grid']:
                rows.append(dict(candidate_id=f'{kind}|{value:g}|{gamma:+g}', kind=kind,
                    alpha=value if kind != 'CAP' else None,
                    angle_degrees=value if kind == 'CAP' else None, gamma=gamma,
                    operations=int(value < 180 if kind == 'CAP' else value != 1) + int(gamma != 0)))
    if len(rows) != 165 or len({r['candidate_id'] for r in rows}) != 165:
        raise ValueError('Expected the exact 165-candidate union')
    return sorted(rows, key=lambda r: r['candidate_id'])


def source_defined(recipe, b_defined, z_defined):
    if recipe['kind'] in ('MIX', 'NMIX') and recipe['alpha'] in (0, 1):
        return b_defined if recipe['alpha'] == 0 else z_defined
    if recipe['kind'] == 'RAW':
        return z_defined
    return b_defined & z_defined


def zero_scores(recipe, b_scores, z_scores, b, z, text):
    kind = recipe['kind']
    flags = np.zeros(len(b_scores), bool)
    if kind == 'RAW':
        return z_scores.copy(), flags
    if kind == 'CAP':
        vector, flags = angle_cap(b, z, recipe['angle_degrees'])
        return vector @ text.T, flags
    alpha = recipe['alpha']
    if kind == 'NMIX':
        return nmix_scores(alpha, b_scores, z_scores, b, z, text)
    if kind == 'MIX':
        return (b_scores.copy() if alpha == 0 else z_scores.copy() if alpha == 1
                else (1-alpha)*b_scores + alpha*z_scores), flags
    raise ValueError('Unknown calibrator kind')


def compact(metric):
    return {k: metric[k] for k in ('B_macro','N_macro','H_macro','B_micro','N_micro','H_micro',
                                  'ordinary_top1','originals','abstentions')}


def run(binding):
    """Execute all1331 logical candidates, reusing the original MV220 exactly."""
    from static_ovmap.semantic_update_evidence_v1.readers import column_index
    from .binding import restore_A
    from .common import canonical_digest, digest_file
    root = Path(binding['output_root'])
    folder = Path(binding['semantic_root'])
    config = binding['specification']['calibration']
    producer = canonical_digest(dict(local=source_identity(('calibration',)), parent={
        name:digest_file(Path(__file__).parent.parent/'semantic_update_evidence_v1'/(name+'.py'))
        for name in ('calibration','metrics','replay','readers')}))
    identity = canonical_digest(dict(producer=producer,
        data={name:row['sha256'] for name,row in binding['public_A'].items()},config=config))
    receipt = root / 'calibration/summary.json'
    if receipt.exists():
        old = verified(receipt)
        if old['input_identity'] != identity:
            raise ValueError('Calibration producer changed; preserve prior completed run before affected repair')
        for record in old['outputs']:
            if digest_file(root / record['path']) != record['sha256']:
                raise ValueError('Completed calibration output changed')
        return old
    if not (root / 'A/restoration.json').exists():
        restore_A(binding)
    meta = verified(root / 'A/import.json')
    with np.load(root / 'A/import.npz', allow_pickle=False) as arrays:
        scores=arrays['scores']; vectors=arrays['embeddings']; defined=arrays['defined']
        ids=arrays['valid_ids']; text=arrays['text']
    target=np.array([o['class_id'] for o in meta['objects']],np.int64)
    keys=np.array([o['key'] for o in meta['objects']])
    families=np.array([o['family'] for o in meta['objects']])
    base=meta['class_split']['base_ids']; novel=meta['class_split']['heldout_ids']
    base_mask=np.isin(ids,base)
    with np.load(folder/'A/predictions.npz',allow_pickle=False) as arrays:
        for name, expected in [('targets',target),('original_keys',keys),('families',families),('valid_ids',ids)]:
            if not np.array_equal(arrays[name],expected):
                raise ValueError('Original candidate join differs: '+name)
        old_predictions=arrays['candidate_predictions'].copy()
    old_recipes=read(folder/'A/crossfit.json.gz')['candidate_recipes']
    if old_predictions.shape!=(2,len(old_recipes),len(target)):
        raise ValueError('Original candidate axes differ')
    old_axis={r['candidate_id']:i for i,r in enumerate(old_recipes)}
    old_aliases={(r['seed'],r['candidate_id']):r
                 for r in read(folder/'A/candidates.json.gz')['aliases']}
    candidates=recipes(config)
    static_index=column_index(meta['columns'],'B_FC',None)
    bs=scores[:,static_index,0]; b=vectors[:,static_index,0]; bd=defined[:,static_index,0]
    metrics=lambda p,mask=None: recognition(target if mask is None else target[mask],
        p if mask is None else p[mask],base,novel)
    records=[]; folds=[]; global_locks={}; fixed=[]; output_arrays={}
    counters=dict(logical_candidate_rows=0,reused_original_MV_predictions=0,
        physical_CPU_candidate_predictions=0,physical_base_score_passes=0,
        physical_CAP_text_matmuls=0,physical_NMIX_text_matmuls=0,
        antipodal_fallback_realizations=0,degenerate_nmix_fallback_realizations=0,
        new_GPU_workers=0,new_head_calls=0,new_training_updates=0,new_image_encodings=0)
    for method in METHODS:
        method_predictions=[]; raw_predictions=[]; recipe_scores={}
        for si,seed in enumerate(SEEDS):
            ci=column_index(meta['columns'],method,seed)
            zs=scores[:,ci,0]; z=vectors[:,ci,0]; zd=defined[:,ci,0]
            raw=predictions(zs,ids,zd); raw_predictions.append(raw)
            records.append(dict(method=method,seed=seed,panel='RAW',metrics=metrics(raw)))
            cache={}; seed_predictions=[]; aliases={}
            for recipe in candidates:
                rid=recipe['candidate_id']; counters['logical_candidate_rows']+=1
                # Original MV predictions are reused, not regenerated by the old110 grid.
                if method=='MV_ATTN' and rid in old_axis:
                    p=old_predictions[si,old_axis[rid]].copy()
                    counters['reused_original_MV_predictions']+=1
                    source_alias=old_aliases[seed,rid]
                    score_hash=source_alias['score_SHA256']; fallback_count=None
                else:
                    cachekey=(recipe['kind'],recipe['alpha'],recipe['angle_degrees'])
                    if cachekey not in cache:
                        cache[cachekey]=zero_scores(recipe,bs,zs,b,z,text)
                        counters['physical_base_score_passes']+=1
                        counters['physical_CAP_text_matmuls']+=int(recipe['kind']=='CAP')
                        counters['physical_NMIX_text_matmuls']+=int(recipe['kind']=='NMIX' and recipe['alpha'] not in (0,1))
                    unshifted,flags=cache[cachekey]
                    shifted=unshifted.copy(); shifted[:,base_mask]-=recipe['gamma']
                    use=source_defined(recipe,bd,zd)
                    p=predictions(shifted,ids,use)
                    counters['physical_CPU_candidate_predictions']+=1
                    fallback_count=int((flags&use).sum())
                    counters['antipodal_fallback_realizations' if recipe['kind']=='CAP' else 'degenerate_nmix_fallback_realizations']+=fallback_count
                    score_hash=__import__('hashlib').sha256(shifted.tobytes()).hexdigest()
                ph=__import__('hashlib').sha256(p.tobytes()).hexdigest()
                if method=='MV_ATTN' and rid in old_axis and ph!=source_alias['prediction_SHA256']:
                    raise ValueError('Original MV candidate prediction identity differs')
                previous=aliases.get(ph)
                if previous is not None and not np.array_equal(p,previous[1]):
                    raise ValueError('Prediction alias digest collision')
                records.append(dict(method=method,seed=seed,panel='CANDIDATE',recipe=recipe,
                    metrics=compact(metrics(p)),prediction_sha256=ph,score_sha256=score_hash,
                    exact_prediction_alias=previous[0] if previous else None,
                    reused_original=(method=='MV_ATTN' and rid in old_axis),fallback_count=fallback_count))
                aliases.setdefault(ph,(rid,p)); seed_predictions.append(p)
            method_predictions.append(np.stack(seed_predictions))
            if method=='MV_ATTN':
                fixed_recipes=[dict(candidate_id='RAW',kind='RAW',gamma=0.),
                    dict(candidate_id='OFFSET',kind='RAW',gamma=.025),
                    dict(candidate_id='CAP10',kind='CAP',angle_degrees=10.,gamma=0.),
                    dict(candidate_id='CAP10_OFFSET',kind='CAP',angle_degrees=10.,gamma=.025)]
                for r in fixed_recipes:
                    if r['kind']=='CAP':
                        rid=f"CAP|10|{r['gamma']:+g}"
                        p=old_predictions[si,old_axis[rid]]
                    elif r['gamma']==0:
                        p=raw
                    else:
                        shifted=zs.copy();shifted[:,base_mask]-=r['gamma']
                        p=predictions(shifted,ids,zd)
                    output_arrays['fixed_'+str(seed)+'_'+r['candidate_id']]=p
                    fixed.append(dict(seed=seed,recipe=r,metrics=metrics(p)))
        all_predictions=np.stack(method_predictions)
        output_arrays[method+'_candidates']=all_predictions
        output_arrays[method+'_raw']=np.stack(raw_predictions)
        oof=np.zeros((2,len(target)),np.int64)
        for family in sorted(set(families)):
            hold=families==family; train=~hold
            choices=[dict(recipe=r,metrics=compact(metrics(all_predictions[0,ri],train)))
                     for ri,r in enumerate(candidates)]
            winner=best_candidate(choices,tolerance=1e-10)
            ri=next(i for i,r in enumerate(candidates) if r['candidate_id']==winner['recipe']['candidate_id'])
            oof[:,hold]=all_predictions[:,ri,hold]
            folds.append(dict(method=method,held_family=str(family),selection_seed=17,
                applied_seeds=list(SEEDS),recipe=winner['recipe'],training_originals=int(train.sum()),
                held_keys=keys[hold].tolist(),training_candidate_metrics=choices))
        output_arrays[method+'_OOF']=oof
        for si,seed in enumerate(SEEDS):
            records.append(dict(method=method,seed=seed,panel='LOFO_POST',metrics=metrics(oof[si])))
        global_locks[method]={}
        for kind in config['kinds']:
            choices=[dict(recipe=r,metrics=compact(metrics(all_predictions[0,ri])))
                     for ri,r in enumerate(candidates) if r['kind']==kind]
            winner=best_candidate(choices,tolerance=1e-10)
            global_locks[method][kind]=winner
            ri=next(i for i,r in enumerate(candidates) if r['candidate_id']==winner['recipe']['candidate_id'])
            output_arrays[method+'_global_'+kind]=all_predictions[:,ri]
    static=[]; static_predictions=[]
    for gamma in config['gamma_grid']:
        shifted=bs.copy();shifted[:,base_mask]-=gamma
        p=predictions(shifted,ids,bd)
        recipe=dict(candidate_id=f'RAW|{gamma:+g}',kind='RAW',gamma=gamma,operations=int(gamma!=0))
        static.append(dict(recipe=recipe,metrics=metrics(p)))
        static_predictions.append(p)
        output_arrays['B_FC_'+f'{gamma:+g}']=p
        counters['logical_candidate_rows']+=1
        counters['physical_CPU_candidate_predictions']+=1
    static_predictions=np.stack(static_predictions)
    static_oof=np.zeros(len(target),np.int64)
    for family in sorted(set(families)):
        hold=families==family;train=~hold
        choices=[dict(recipe=row['recipe'],metrics=compact(metrics(static_predictions[i],train)))
                 for i,row in enumerate(static)]
        winner=best_candidate(choices,tolerance=1e-10)
        ri=next(i for i,row in enumerate(static) if row['recipe']['candidate_id']==winner['recipe']['candidate_id'])
        static_oof[hold]=static_predictions[ri,hold]
        folds.append(dict(method='B_FC',held_family=str(family),selection_seed=None,
            applied_seeds=[None],recipe=winner['recipe'],training_originals=int(train.sum()),
            held_keys=keys[hold].tolist(),training_candidate_metrics=choices))
    output_arrays['B_FC_OOF']=static_oof
    records.extend([dict(method='B_FC',seed=None,panel='RAW',metrics=metrics(predictions(bs,ids,bd))),
                    dict(method='B_FC',seed=None,panel='LOFO_POST',metrics=metrics(static_oof))])
    if counters['logical_candidate_rows']!=1331 or counters['reused_original_MV_predictions']!=220:
        raise ValueError('Logical candidate coverage differs')
    save_arrays(root/'calibration/predictions.npz',**output_arrays,targets=target,original_keys=keys,
                families=families,valid_ids=ids,candidate_ids=np.array([r['candidate_id'] for r in candidates]))
    write(root/'calibration/candidates.json.gz',dict(status='DONE',rows=records,recipes=candidates,
          static_B_FC=static,counters=counters,originals=len(target),independent_confirmation=False))
    write(root/'calibration/crossfit.json.gz',dict(status='DONE',folds=folds,selection_seed=17,
          apply_seeds=list(SEEDS),original_order=keys,OOF_macro_recomputed=True,independent_confirmation=False))
    write(root/'simple_recipe_lock.json',dict(status='DONE',selection_role='EXPOSED_CONF12_ALL_SEED17',
          novel_label_free=False,per_method=global_locks,
          static_B_FC=best_candidate(static,tolerance=1e-10),
          benchmark_simple_controls=global_locks['MV_ATTN'],
          immutable_final_method=dict(kind='CAP',angle_degrees=10.,gamma=.025)))
    outputs=[]
    for name in ['calibration/predictions.npz','calibration/candidates.json.gz',
                 'calibration/crossfit.json.gz','simple_recipe_lock.json']:
        outputs.append(dict(path=name,sha256=digest_file(root/name)))
    return write(receipt,dict(status='DONE',input_identity=identity,producer=producer,outputs=outputs,
        raw_and_post=[r for r in records if r['panel']!='CANDIDATE'],fixed_MV=fixed,
        counters=counters,originals=279,unsupported=9,families=12,
        exposure_role='OLD160_40_CONF12_EXPLORATORY',benchmark_gain_gate=False))
