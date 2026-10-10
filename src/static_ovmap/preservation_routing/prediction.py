"""Same physical parent banks; actual fixed-G1 F replacements only."""
from pathlib import Path
import time
import numpy as np
from .binding import load_scene
from .common import PathResolver,arrays_record,canonical_digest,verified,write,_array_digest
def run(binding,fc):
    import torch
    from .data import ObjectLoader
    from .recognition import load_head
    from .models import teacher
    root=Path(binding['output_root']);parent=Path(binding['lr_parent_root']);locked=verified(root/'predictor_lock.json')
    if not locked['activated_full_path']:raise ValueError('Whole-map path not activated')
    path=root/'recognition/regression.json'
    identity=canonical_digest(dict(predictor_lock=locked['identity'],parent_regression=verified(parent/'regression/manifest.json')['identity'],
                                    features=verified(parent/'features/regression.json')['identity'],producer=fc.index.identity(__file__)))
    if path.exists():
        previous=verified(path)
        if previous['input_identity']!=identity:raise ValueError('Actual regression inputs/code changed')
    else:
        started=time.perf_counter();loader=ObjectLoader(fc,verified(parent/'features/regression.json'))
        heads={m:load_head(locked['heads'][name]['checkpoint']['path'],fc.device)[0] for m,name in locked['map_heads'].items()}
        schema=list(heads);records={};loading_seconds=forward_seconds=0.;forwards=0
        resolver=PathResolver(binding['path_map']);parent_scores=resolver.rewrite(verified(parent/'recognition/regression.json'))
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for scene in [s for names in binding['cohorts'].values() for s in names]:
                manifest=verified(parent/'regression'/scene/'manifest.json')
                for obj in manifest['objects']:
                    begin=time.perf_counter();x,_,pooling=loader.load(obj,0,8);torch.cuda.synchronize();loading_seconds+=time.perf_counter()-begin
                    old=parent_scores['records'][obj['key']]
                    with np.load(old['arrays']['path'],allow_pickle=False) as a:
                        baseline=a['embeddings'][list(a['methods']).index('LR04_FC_8')]
                    np.testing.assert_allclose(teacher(x).cpu().numpy(),baseline,rtol=1e-5,atol=1e-6)
                    begin=time.perf_counter();values=[heads[m](x,fc.phi)['embedding'].cpu().numpy() for m in schema]
                    torch.cuda.synchronize();forward_seconds+=time.perf_counter()-begin;forwards+=len(schema)
                    context=resolver.rewrite(verified(binding['scenes'][scene]['context']['path']));fc.index.identity(context['FC_text']['path'],context['FC_text'])
                    if context['FC_physical_model_identity']!=fc.model_key:raise ValueError('Map/readout frozen FC identity differs')
                    with np.load(context['FC_text']['path'],allow_pickle=False) as a:text=a['text_embeddings'].copy();ids=a['valid_ids'].copy()
                    scores=[text@z for z in values]
                    arr=arrays_record(root/'recognition/regression'/scene/(str(obj['owner'])+'.npz'),
                              dict(methods=np.asarray(schema),valid_ids=ids,embeddings=np.asarray(values,np.float32),cosines=np.asarray(scores)),fc.index)
                    records[obj['key']]=dict(owner=obj['owner'],scene=scene,old_class=obj['old_class'],arrays=arr,views=obj['views'],ordinary_pooling=pooling)
                print('MAP_READOUT',scene,len(manifest['objects']),flush=True)
        previous=write(path,dict(status='ALL_NEW_F_LOCKED',input_identity=identity,records=records,methods=schema,
                    original_objects=len(records),head_forwards=forwards,physical_image_encodings=0,
                    timing_boundary='MODEL_RESIDENT_CACHED_PARENT_FP32_GRIDS',elapsed_seconds=time.perf_counter()-started,
                    loader_and_frozen_pooling_seconds=loading_seconds,selected_head_forward_seconds=forward_seconds,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),GT_used=False))
    link=root/'regression'
    if not link.exists():link.symlink_to(parent/'regression',target_is_directory=True)
    summary=root/'predictions/summary.json'
    if summary.exists():
        result=verified(summary)
        if result.get('readout_identity')!=previous['identity']:raise ValueError('Saved prediction source scores changed')
        return result
    started=time.perf_counter();result=build_predictions(binding)
    return write(summary,dict(result,readout_identity=previous['identity'],payload_construction_seconds=time.perf_counter()-started))

def build_predictions(binding):
    from static_ovmap.disagreement_query.outputs import build_payload
    from static_ovmap.module_validation.scannet_study import save_prediction,load_prediction
    from static_ovmap.paired_evidence_study.residuals import pool,simple_dependence
    root = Path(binding['output_root']); scores = verified(root/'recognition/regression.json')
    if scores['status'] != 'ALL_NEW_F_LOCKED': raise ValueError('All new object readouts must precede payload construction')
    methods = ['PR00_D2','PR01_G1','PR02_FC8']+scores['methods']; locks = {}
    for names in binding['cohorts'].values():
        for scene in names:
            p,original = load_scene(binding,scene); manifest = verified(root/'regression'/scene/'manifest.json')
            eligible = manifest['eligible_owners']; saved = {}; decisions = {}
            for method in methods:
                if method in ('PR00_D2','PR01_G1','PR02_FC8'):
                    old_method={'PR00_D2':'LR00_D2','PR01_G1':'LR01_G1','PR02_FC8':'LR04_FC_8'}[method]
                    parent=PathResolver(binding['path_map']).rewrite(verified(Path(binding['lr_parent_root'])/'predictions'/scene/'receipt.json')['methods'][old_method])
                    payload=load_prediction(parent['manifest'])
                    saved[method]=dict(status='PREDICTION_LOCKED',manifest=parent['manifest'],prediction_key=payload.prediction_key,
                                       record_key=payload.record_key,source_kind='EXACT_LR_PARENT_BASELINE')
                    continue
                relabels = {}; records = {}
                for owner in eligible:
                    acquired = scores['records'][scene+':'+str(owner)]
                    with np.load(acquired['arrays']['path'],allow_pickle=False) as arr:
                        position = list(arr['methods']).index(method); new = arr['cosines'][position]
                        if not np.array_equal(arr['valid_ids'],p.valid_ids): raise ValueError('Frozen text ordering changed')
                    source = {name:(np.asarray(p.sources[name]['objects'][str(owner)]['scores'],np.float64)
                              if p.sources[name]['objects'][str(owner)]['available'] else None) for name in ('N','Q','F')}
                    old_F = source['F']; source['F'] = new
                    base = pool(source,p.temperatures,('N','Q','F'))
                    final = simple_dependence(source,p.temperatures,base,'PE_D2_GROUPED')
                    if final is None or not np.isfinite(final).all(): raise ValueError('Failed required historical-F replacement')
                    label = p.valid_ids[int(final.argmax())]; relabels[owner] = label
                    records[str(owner)] = dict(old_class=acquired['old_class'],new_F_top1=p.valid_ids[int(new.argmax())],
                            old_F_top1=p.valid_ids[int(old_F.argmax())],applied_class=label,old_NQF_cosines={name:p.sources[name]['objects'][str(owner)]['scores'] for name in ('N','Q','F')},
                            old_D2_probability=p.probabilities[str(owner)]['probabilities'],new_F_cosines=new.tolist(),
                            final_NQ_newF_probability=final.tolist(),representation=acquired['arrays'],views=acquired['views'])
                payload = build_payload(original.g1,original.d2,original.raw,relabels,eligible,original.nearest,original.matched,
                                        method,{},dict(fixed_geometry='G1',historical_F_replaced_only=True))
                # Exact aliases require identical source owner/semantic/current ranks.
                alias = next((r for r in saved.values() if r['prediction_key']==payload.prediction_key),None)
                if alias:
                    item = {**alias,'source_kind':'EXACT_OWNER_SEMANTIC_CURRENT_RANK_ALIAS',
                            'alias_proof':dict(owners=_array_digest(payload.owner_ids),semantics=_array_digest(payload.semantic_labels),ranks=canonical_digest(payload.instance_ranks))}
                else:
                    path = save_prediction(payload,root/'predictions'/scene/method)
                    item = dict(status='PREDICTION_LOCKED',manifest=str(path),prediction_key=payload.prediction_key,
                                record_key=payload.record_key,source_kind='REAL_FIXED_G1_LEARNED_READOUT')
                saved[method] = item
                decisions[method] = write(root/'decisions'/scene/(method+'.json'),dict(status='DECISIONS_LOCKED',
                        scene=scene,method=method,owners=records,GT_used=False,source_grouping='EXACT_PE_D2_GROUPED'))['identity']
            locks[scene] = write(root/'predictions'/scene/'receipt.json',dict(status='PREDICTIONS_LOCKED',scene=scene,
                        methods=saved,decisions=decisions,eligible_owners=eligible,full_G1_registry=manifest['full_G1_positive_registry'],
                        owner_partition_unchanged=True,recovered_classes_unchanged=True,GT_used=False))['identity']
            p.index.write_memo(root/'inputs'/scene/'verifications.json')
    return write(root/'predictions/summary.json',dict(status='ALL_PREDICTIONS_LOCKED',scenes=locks,methods=methods,
                 cohorts=binding['cohorts'],logical_records=len(methods)*26,GT_used=False))
