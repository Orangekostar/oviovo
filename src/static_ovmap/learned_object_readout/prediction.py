"""GT-free real-map observation banks and exact historical-F replacements."""
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from .binding import load_scene
from .common import BASELINES, ConsumptionIndex, arrays_record, canonical_digest, verified, write, _array_digest
from .data import proposal_sites, site_observation
from .observations import choose_views, full_qualified


def prepare_regression_scene(binding, scene):
    from static_ovmap.samv_local_probe.query_plan import observations
    from static_ovmap.minimal_instance_repair.observations import SourceRowProjector
    from static_ovmap.disagreement_query.physical_support import canonical_sites, quadrature, direct_visibility
    root = Path(binding['output_root']); dest = root/'regression'/scene
    p, original = load_scene(binding, scene); index = p.index
    identity = canonical_digest(dict(binding=binding['identity'], scene=scene, geometry=p.geometry.to_dict(),
                                    owners=_array_digest(p.owner_ids), producer=index.identity(__file__)))
    path = dest/'manifest.json'
    if path.exists():
        previous = verified(path)
        if previous['input_identity'] != identity: raise ValueError('Fixed real-map proposal manifest changed')
        return previous
    started = time.perf_counter()
    old = verified(Path(binding['minimal_root'])/'source_binding.json')['specification']['observer']
    if any(old[k] != v for k,v in dict(max_frames=32,pose_translation_m=.2,pose_rotation_degrees=15.,
                                      depth_absolute_tolerance_m=.02,depth_relative_tolerance=.02,minimum_depth_m=1e-6).items()):
        raise ValueError('Inherited original camera observer differs from fixed protocol')
    shadow = {**binding,'specification':{**binding['specification'],'observer':old}}
    adapter = SimpleNamespace(scene=scene,index=index,g1=SimpleNamespace(geometry=p.geometry),capture=p.capture,
                              capture_path=p.capture_path,xyz=p.xyz,faces=p.faces)
    frames, observed = observations(shadow,adapter)
    rasters, depths = [], []
    for row in observed['frames']:
        with np.load(row['arrays']['path'],allow_pickle=False) as arr:
            source, valid = arr['source_rows'],arr['valid']
        raster = np.zeros(source.shape,np.int64); raster[valid] = p.owner_ids[source[valid]]
        rasters.append(raster)
        with np.load(row['depth']['path'],allow_pickle=False) as arr: depths.append(arr['depth_m'])
    physical = canonical_sites(p.xyz,p.faces,p.owner_ids)
    projector = SourceRowProjector(p.xyz,p.faces)
    d2 = set(map(int,np.unique(p.d2_owner_ids[p.d2_owner_ids>0])))
    inventory, objects = [], []
    for owner in sorted(map(int,np.unique(p.owner_ids[p.owner_ids>0]))):
        support = p.owner_ids == owner
        label = int(p.semantic_labels[np.flatnonzero(support)[0]])
        masks = [r == owner for r in rasters]
        qualified = [i for i,m in enumerate(masks) if full_qualified(m)]
        prior = p.probabilities.get(str(owner))
        historical = p.sources['F']['objects'].get(str(owner))
        reason = ('RECOVERED_G1_OWNER' if owner not in d2 else 'RAW_ZERO_PROTECTED' if np.any(p.raw[support]==0)
                  else 'HISTORICAL_F_UNAVAILABLE' if not historical or not historical['available']
                  else 'D2_PRIOR_UNAVAILABLE' if not prior or prior['probabilities'] is None
                  else 'FEWER_THAN_TWO_QUALIFIED_VIEWS' if len(qualified)<2 else 'ELIGIBLE')
        row = dict(owner=owner,old_class=label,reason=reason,eligible=reason=='ELIGIBLE',
                   source_vertices=int(support.sum()),qualified_views=len(qualified))
        inventory.append(row)
        if reason != 'ELIGIBLE': continue
        selected = physical['owners']==owner
        planned, area = quadrature(physical['xyz'][selected],physical['area'][selected],4096)
        footprints = [direct_visibility(planned,frames[i],depths[i],masks[i],projector.mesh.scene)[1] for i in qualified]
        bank = choose_views([frames[i]['frame_id'] for i in qualified],[masks[i] for i in qualified],footprints,area)
        chosen = [qualified[i] for i in bank]
        sites = proposal_sites(physical,p.owner_ids,support)
        points = physical['xyz'][sites]; nviews = len(chosen)
        uv = np.zeros((1,nviews,64,2),np.float64); metadata = np.zeros((1,nviews,64,8),np.float32)
        valid = np.zeros((1,nviews,64),bool)
        xyz = np.zeros((1,64,3),np.float64); available = np.zeros((1,64),bool)
        xyz[0,:len(points)] = points; available[0,:len(points)] = True
        support_points = p.xyz[support]; bbox = (support_points.min(0),support_points.max(0))
        for vi,fi in enumerate(chosen):
            u,q,v = site_observation(points,frames[fi],depths[fi],masks[fi],projector.mesh.scene,bbox)
            uv[0,vi,:len(points)] = u; metadata[0,vi,:len(points)] = q; valid[0,vi,:len(points)] = v
        inputs = arrays_record(dest/'objects'/f'{owner:06d}.inputs.npz',
                 dict(masks=np.stack([masks[i] for i in chosen])[None],uv=uv,metadata=metadata,
                      local_valid=valid,site_xyz=xyz,site_available=available),index)
        views = [dict(frames[i],rgb=observed['frames'][i]['rgb'],depth=observed['frames'][i]['depth'],
                      observation=observed['frames'][i]['arrays']) for i in chosen]
        objects.append(dict(key=scene+':'+str(owner),scene=scene,owner=owner,views=views,inputs=inputs,
                            old_class=label,role='EXPOSED_REAL_PREDICTED_MAP',GT_inputs=False))
    result = write(path,dict(status='PROPOSALS_LOCKED',input_identity=identity,scene=scene,objects=objects,
             inventory=inventory,full_G1_positive_registry=[r['owner'] for r in inventory],
             eligible_owners=[o['owner'] for o in objects],pose_representatives=len(frames),
             observations=observed['identity'],GT_used=False,no_GT_semantic_targets=True,
             physical_support=physical['audit'],elapsed_seconds=time.perf_counter()-started))
    index.write_memo(root/'inputs'/scene/'verifications.json')
    print('REGRESSION_PROPOSALS',scene,len(objects),'/',len(inventory),flush=True)
    return result


def prepare_regression(binding):
    root = Path(binding['output_root'])
    # These manifests are geometry-only and may be prepared while training runs.
    rows = [prepare_regression_scene(binding,s) for names in binding['cohorts'].values() for s in names]
    return write(root/'regression/manifest.json',dict(status='ALL_PROPOSALS_LOCKED',
                 scenes={r['scene']:r['identity'] for r in rows},cohorts=binding['cohorts'],
                 original_objects=sum(len(r['objects']) for r in rows),GT_used=False))


def regression_readout(binding):
    import torch
    from static_ovmap.backbone_wave1.runtime import exclusive_lock
    from .features import FrozenFC,ObjectLoader,execution_config
    from .recognition import load_heads,compact_summary
    root = Path(binding['output_root'])
    main, repeat = verified(root/'dev_nomination.json'),verified(root/'repeat_nomination.json')
    prepared = verified(root/'regression/manifest.json')
    manifests = [verified(root/'regression'/s/'manifest.json') for names in binding['cohorts'].values() for s in names]
    objects = [o for m in manifests for o in m['objects']]
    path = root/'recognition/regression.json'
    identity = canonical_digest(dict(proposals=prepared['identity'],main=main['identity'],repeat=repeat['identity']))
    if path.exists():
        previous = verified(path)
        if previous['input_identity'] != identity: raise ValueError('Nominated regression readout changed')
        return previous
    started = time.perf_counter()
    with exclusive_lock(execution_config(binding)['gpu_lock']):
        fc = FrozenFC(binding); frames = {}
        for obj in objects:
            for frame in obj['views']:
                key = obj['scene']+':'+str(frame['frame_id'])
                if key not in frames: frames[key] = fc.capture(frame)
        features = write(root/'features/regression.json',dict(status='COMPLETE',frames=frames,
              physical_images=len(frames),physical_encodings_this_invocation=fc.images,
              physical_encodings_persisted=sum(r['physical_image_encodings'] for r in {v['key']:v for v in frames.values()}.values()),
              FC_model=fc.model_key,raw_C=next(iter(frames.values()))['raw_C'],projected_D=768))
        loader = ObjectLoader(fc,features); heads = load_heads(binding,fc)
        rows = {}; schema = [r['id'] for r in binding['specification']['methods'] if r['id'] not in BASELINES]
        schema += [m for m in heads if m.startswith('R29_')]
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for obj in objects:
                inputs,vectors,pooling = loader.load(obj,0,8)
                representations = {}
                for method,prefix in (('LR02_FC_2',2),('LR03_FC_4',4),('LR04_FC_8',8)):
                    representations[method] = torch.nn.functional.normalize(vectors[:prefix].mean(0),dim=0)
                summaries = {}
                for method,head in heads.items():
                    result = head(inputs,fc.phi); representations[method] = result['embedding']
                    summaries[method] = compact_summary(result,inputs)
                values = {method:z.cpu().numpy().copy() for method,z in representations.items()}
                if any(not np.isfinite(z).all() or not np.isclose(np.linalg.norm(z),1,atol=1e-5) for z in values.values()):
                    raise ValueError('Nonfinite/zero real-map object representation')
                # Score each map with its original exact frozen dataset text ordering.
                context = fc.raw_cache.resolver.rewrite(verified(binding['scenes'][obj['scene']]['context']['path']))
                if context['FC_physical_model_identity']!=fc.model_key:
                    raise ValueError('Regression map and trained head have different frozen FC models')
                text_path = Path(context['FC_text']['path']); fc.index.identity(text_path,context['FC_text'])
                with np.load(text_path,allow_pickle=False) as arr:
                    text,ids = arr['text_embeddings'],arr['valid_ids']
                scores = {method:text@z for method,z in values.items()}
                array = arrays_record(root/'recognition/regression'/obj['scene']/(str(obj['owner'])+'.npz'),
                         dict(methods=np.asarray(schema),valid_ids=ids,embeddings=np.stack([values[m] for m in schema]),
                              cosines=np.stack([scores[m] for m in schema])),fc.index)
                rows[obj['key']] = dict(owner=obj['owner'],scene=obj['scene'],old_class=obj['old_class'],arrays=array,
                                  views=obj['views'],summaries=summaries,ordinary_pooling=pooling)
        result = write(path,dict(status='ALL_NEW_F_LOCKED',input_identity=identity,records=rows,methods=schema,
                  method_seeds={m:(None if m.startswith('LR0') and m in ('LR02_FC_2','LR03_FC_4','LR04_FC_8') else
                                  29 if m.startswith('R29_') else 17) for m in schema},features=features['identity'],
                  elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),GT_used=False))
        fc.index.write_memo(root/'features/verifications.json')
    return result


def build_predictions(binding):
    from static_ovmap.disagreement_query.outputs import build_payload
    from static_ovmap.module_validation.scannet_study import save_prediction
    from static_ovmap.paired_evidence_study.residuals import pool,simple_dependence
    root = Path(binding['output_root']); scores = verified(root/'recognition/regression.json')
    if scores['status'] != 'ALL_NEW_F_LOCKED': raise ValueError('All new object readouts must precede payload construction')
    methods = list(BASELINES)+scores['methods']; locks = {}
    for names in binding['cohorts'].values():
        for scene in names:
            p,original = load_scene(binding,scene); manifest = verified(root/'regression'/scene/'manifest.json')
            eligible = manifest['eligible_owners']; saved = {}; decisions = {}
            for method in methods:
                if method in BASELINES:
                    payload = original.d2 if method=='LR00_D2' else original.g1
                    parent = binding['scenes'][scene]['parent_predictions']['IR00_D2' if method=='LR00_D2' else 'IR01_G1']
                    saved[method] = dict(status='PREDICTION_LOCKED',manifest=parent['manifest'],prediction_key=payload.prediction_key,
                                         record_key=payload.record_key,source_kind='EXACT_PARENT_BASELINE')
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
