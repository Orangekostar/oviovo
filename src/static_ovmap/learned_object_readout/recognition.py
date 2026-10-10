"""Nominated checkpoints and full-vocabulary heldout proposal recognition."""
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from .common import arrays_record,canonical_digest,verified,write


def load_heads(binding, fc):
    import torch
    from .model import ReadoutHead
    from .training import BRANCHES
    root = Path(binding['output_root']); heads = {}
    main,repeat = verified(root/'dev_nomination.json'),verified(root/'repeat_nomination.json')
    if main['status'] != 'NOMINATED' or repeat['status'] != 'NOMINATED' or repeat['proposed_architecture'] != main['proposed_architecture']:
        raise ValueError('Both seed checkpoint nominations must freeze the same architecture')
    for seed,nomination in ((17,main),(29,repeat)):
        if nomination['data'] != verified(root/'data_profile.json')['identity']:
            raise ValueError('Selected checkpoints refer to different supervised data')
        for branch,row in nomination['checkpoints'].items():
            fc.index.identity(row['path'],row)
            state = torch.load(row['path'],map_location='cpu',weights_only=False)
            if state['seed'] != seed or state['branch'] != branch:
                raise ValueError('Selected model seed/branch identity changed')
            head = ReadoutHead(BRANCHES[branch][0]).to(fc.device)
            head.load_state_dict(state['model'],strict=True); head.eval().requires_grad_(False)
            heads[branch if seed==17 else 'R29_'+branch] = head
    return heads


def compact_summary(result, inputs):
    import torch
    weights = result['ma_maps']
    summary = dict(fallback=result['fallback'],eta=result['eta'],grouping=result['grouping'],
              available_views=int(inputs['view_valid'].sum()),valid_local_tokens=int(inputs['local_valid'].sum()),
              possible_local_tokens=int(inputs['local_valid'].numel()),
              MA_effective_pixels=(1/weights.square().sum(-1)).cpu().tolist(),
              MA_spatial_entropy=(-(weights*weights.clamp_min(1e-30).log()).sum(-1)).cpu().tolist())
    if 'reliability' in result and result['local_valid'].any():
        r = result['reliability'][result['local_valid']]
        summary['reliability_logits'] = dict(min=float(r.min()),max=float(r.max()),mean=float(r.mean()),std=float(r.std(unbiased=False)))
    if 'local_attention' in result:
        summary.update(local_group_count=result['group_count'],
                       local_query_entropy=(-(result['local_attention']*result['local_attention'].clamp_min(1e-30).log()).sum(-1)).cpu().tolist(),
                       global_query_entropy=(-(result['global_attention']*result['global_attention'].clamp_min(1e-30).log()).sum(-1)).cpu().tolist())
    return summary


def recognition_metrics(records):
    if not records:
        return dict(original_objects=0,variant_records=0,categories=0,top1=None,macro_recall=None,NLL=None)
    classes = sorted({r['class_id'] for r in records})
    recall = [float(np.mean([r['prediction']==c for r in records if r['class_id']==c])) for c in classes]
    return dict(original_objects=len({r['key'] for r in records}),variant_records=len(records),categories=len(classes),
                top1=float(np.mean([r['prediction']==r['class_id'] for r in records])),
                macro_recall=float(np.mean(recall)),NLL=float(np.mean([r['NLL'] for r in records])),
                families=len({r['family'] for r in records}),correlated_variants=True,classification_denominator=200)


def dev_recognition(binding):
    """Measure selected-head DEV inference separately, without another selection."""
    import torch
    from static_ovmap.backbone_wave1.runtime import exclusive_lock
    from .features import FrozenFC,ObjectLoader,execution_config
    from .training import evaluate_dev
    root = Path(binding['output_root']); main = verified(root/'dev_nomination.json')
    repeat = verified(root/'repeat_nomination.json'); features = verified(root/'features/train-dev.json')
    identity = canonical_digest(dict(main=main['identity'],repeat=repeat['identity'],features=features['identity']))
    path = root/'recognition/dev_selected.json'
    if path.exists():
        old = verified(path)
        if old['input_identity']!=identity: raise ValueError('Selected DEV inference inputs changed')
        return old
    split = verified(root/'split_manifest.json')
    objects = [o for s in split['roles']['dev'] for o in verified(root/'data/generated'/s/'manifest.json')['objects']]
    with exclusive_lock(execution_config(binding)['gpu_lock']):
        fc = FrozenFC(binding); loader = ObjectLoader(fc,features); heads = load_heads(binding,fc)
        torch.cuda.reset_peak_memory_stats(); started = time.perf_counter(); rows = {}
        for method,head in {'LR04_FC_8':None,**heads}.items():
            begin = time.perf_counter(); row = evaluate_dev(head,loader,objects,fc); torch.cuda.synchronize()
            rows[method] = dict(**row,elapsed_seconds=time.perf_counter()-begin)
        result = write(path,dict(status='COMPLETE',input_identity=identity,methods=rows,selected_after_both_nominations=True,
                architecture_or_checkpoint_selection_repeated=False,scope='DEV_BASE_PROPOSAL_RECOGNITION',
                elapsed_seconds=time.perf_counter()-started,physical_image_encodings=0,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                timing_boundary='MODEL_RESIDENT; CACHED_FROZEN_GRIDS; SELECTED_HEAD_FORWARD_AND_METRICS',
                checkpoint_validation_costs_already_in_training=True))
        fc.index.write_memo(root/'features/verifications.json')
    return result


def holdout_recognition(binding):
    import torch
    from static_ovmap.backbone_wave1.runtime import exclusive_lock
    from .features import FrozenFC,ObjectLoader,execution_config
    root = Path(binding['output_root']); path = root/'recognition/holdout.json'
    main,repeat = verified(root/'dev_nomination.json'),verified(root/'repeat_nomination.json')
    profile = verified(root/'holdout_data_profile.json'); features = verified(root/'features/holdout.json')
    identity = canonical_digest(dict(main=main['identity'],repeat=repeat['identity'],data=profile['identity'],features=features['identity']))
    if path.exists():
        previous = verified(path)
        if previous['input_identity'] != identity: raise ValueError('H evaluation inputs changed')
        return previous
    split = verified(root/'split_manifest.json')
    objects = [o for scene in split['roles']['holdout'] for o in verified(root/'data/generated'/scene/'manifest.json')['objects']]
    started = time.perf_counter()
    with exclusive_lock(execution_config(binding)['gpu_lock']):
        fc = FrozenFC(binding); loader = ObjectLoader(fc,features); heads = load_heads(binding,fc)
        methods = ['LR02_FC_2','LR03_FC_4','LR04_FC_8',*heads]
        rows = {m:[] for m in methods}; summaries = {}
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for obj in objects:
                for condition in range(4):
                    inputs,vectors,pooling = loader.load(obj,condition,8)
                    representations = {method:torch.nn.functional.normalize(vectors[:prefix].mean(0),dim=0)
                              for method,prefix in (('LR02_FC_2',2),('LR03_FC_4',4),('LR04_FC_8',8))}
                    item_summaries = {}
                    for method,head in heads.items():
                        result = head(inputs,fc.phi); representations[method] = result['embedding']
                        item_summaries[method] = compact_summary(result,inputs)
                    values = {m:z.cpu().numpy().copy() for m,z in representations.items()}
                    cosines = {m:fc.text@z for m,z in values.items()}
                    if any(not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v),1,atol=1e-5) for v in values.values()):
                        raise ValueError('Failed mandatory heldout representation')
                    target = fc.ids.index(obj['class_id'])
                    for method in methods:
                        score = cosines[method]; logits = score/.07
                        rows[method].append(dict(key=obj['key'],family=obj['family'],class_id=obj['class_id'],base=obj['base'],
                            condition=('clean','truncate','append','truncate_append')[condition],
                            prediction=fc.ids[int(score.argmax())],NLL=float(logsumexp(logits)-logits[target]),
                            available_views=min(len(obj['views']),dict(LR02_FC_2=2,LR03_FC_4=4).get(method,8)),
                            no_op=condition>0 and ('clean','truncate','append','truncate_append')[condition] in obj['no_op_conditions']))
                    key = obj['key']+':'+str(condition)
                    array = arrays_record(root/'recognition/holdout'/obj['scene']/(str(obj['original_local_id'])+'_'+str(condition)+'.npz'),
                                  dict(methods=np.asarray(methods),embeddings=np.stack([values[m] for m in methods]),
                                       full200_cosines=np.stack([cosines[m] for m in methods]),valid_ids=np.asarray(fc.ids)),fc.index)
                    summaries[key] = dict(arrays=array,models=item_summaries,ordinary_pooling=pooling)
        metrics = {}
        for method,records in rows.items():
            metrics[method] = {subset:{condition:recognition_metrics([r for r in records
                        if (subset=='all' or r['base']==(subset=='base')) and (condition=='all_conditions' or r['condition']==condition)])
                        for condition in ('clean','truncate','append','truncate_append','all_conditions')}
                        for subset in ('base','heldout','all')}
        novel = [o for o in objects if not o['base']]
        novel_support = dict(original_objects=len(novel),families=len({o['family'] for o in novel}),
                             status='SUFFICIENT' if len(novel)>=20 and len({o['family'] for o in novel})>=2 else 'INSUFFICIENT_NOVEL_SUPPORT')
        contrasts = {}
        for name,a,b in [('MA_minus_FC8','LR05_MA_8','LR04_FC_8'),('VIEW_minus_MA','LR06_MV_VIEW','LR05_MA_8'),
                         ('SURFACE_minus_VIEW','LR07_MV_SURFACE','LR06_MV_VIEW'),('AUX_minus_SURFACE','LR08_MV_AUX','LR07_MV_SURFACE'),
                         ('REPEAT_PROPOSED_minus_MA','R29_'+main['proposed_architecture'],'R29_LR05_MA_8')]:
            first = {(r['key'],r['condition']):r for r in rows[a]}; second = {(r['key'],r['condition']):r for r in rows[b]}
            if first.keys()!=second.keys(): raise ValueError('Paired recognition original-object coverage changed')
            contrasts[name] = dict(first=a,second=b,original_objects=len(objects),variant_records=len(first),
                          top1_delta=float(np.mean([int(r['prediction']==r['class_id'])-int(second[k]['prediction']==r['class_id']) for k,r in first.items()])),
                          NLL_delta=float(np.mean([r['NLL']-second[k]['NLL'] for k,r in first.items()])))
        result = write(path,dict(status='COMPLETE',input_identity=identity,scope='NEW_FAMILY_GT_DERIVED_PROPOSAL_RECOGNITION',
                methods=methods,metrics=metrics,records=rows,compact_summaries=summaries,paired_contrasts=contrasts,
                original_objects=len(objects),novel_support=novel_support,full_text_classes=200,
                independent_whole_map_AP=False,elapsed_seconds=time.perf_counter()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        fc.index.write_memo(root/'features/verifications.json')
    return result
