"""GT-free new-support masks and fixed-area views, before model acquisition."""

from pathlib import Path
import time

import numpy as np

from static_ovmap.a7_evidence_upgrade.region_adapter import resized_shape
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.recovery_wave2.binding import read

from .binding import load_scene, seal
from .observations import support_mask
from .reread import qualified_mask, recognition_masks, select_incumbents


def choose_view(choices, *, bank=None, minimum_pixels=100, minimum_extent=2):
    available = []
    for row in choices:
        if bank is not None and row['bank']!=bank:
            continue
        qualified,bbox = qualified_mask(row['mask'],minimum_pixels=minimum_pixels,
                                        minimum_extent=minimum_extent)
        if qualified:
            available.append({**row,'pixels':int(row['mask'].sum()),'bbox':bbox,
                              'mask_digest':_array_digest(row['mask'])})
    return min(available,key=lambda r:(-r['pixels'],r['frame_id'],r['mask_digest'])) if available else None


def region_identity(support, view, kind, model, text, dense_hw, output_hw, aggregation):
    mask_digest = view['mask_digest'] if 'mask_digest' in view else _array_digest(view['mask'])
    return canonical_digest({'geometry_support':support,'frame_id':view['frame_id'],
        'mask':mask_digest,
        'type':kind,'representation':aggregation,'model':model,'text':text,
        'dense_hw':dense_hw,'output_hw':output_hw,'pooling':'FINE_AREA_RAW_V_THEN_FROZEN_HEAD'
            if aggregation.startswith('ANYUP') else 'ORIGINAL_FC_AREA_FALLBACK_V2'})


def unpack_region(region, index=None):
    if index is not None:
        index.identity(region['mask']['path'],region['mask'])
    with np.load(region['mask']['path'],allow_pickle=False) as arrays:
        shape = tuple(map(int,arrays['shape']))
        mask = np.unpackbits(arrays['packed'],bitorder='little',count=int(np.prod(shape))).reshape(shape).astype(bool)
    if _array_digest(mask)!=region['mask_digest'] or int(mask.sum())!=region['pixels']:
        raise ValueError('new-support recognition mask content changed')
    return mask


def prepare_scene(binding, scene):
    inputs = load_scene(binding,scene)
    root = Path(binding['output_root'])
    observation = read(root/'observations'/scene/'receipt.json')
    proposal = read(root/'proposals'/scene/'receipt.json')
    if proposal['status']!='PROVISIONAL_STRUCTURE_LOCKED' or proposal['class_scores_read']:
        raise ValueError('new recognition may only read the already locked structural decisions')
    settings,repair = binding['specification']['reread'],binding['specification']['repair']
    key = canonical_digest({'observation':observation['identity'],'proposal':proposal['identity'],
        'supports':inputs.units.identity,'model':inputs.data['FC_physical_model_identity'],
        'text':inputs.data['FC_text'],'settings':settings,'repair':repair,
        'producer':inputs.index.identity(__file__)})
    dest = root/'recognition'/scene
    path = dest/'plan.json'
    if path.exists():
        previous = read(path)
        if previous['input_identity']!=key:
            from .resume import invalidate_descendants
            invalidate_descendants(binding,scene,'recognition_plan','new-support plan input identity changed')
        else:
            for item in previous['outputs']:
                inputs.index.identity(item['path'],item)
            return previous
    begin = time.perf_counter()
    incumbents = select_incumbents(inputs.units,inputs.probabilities,limit=settings['max_incumbents'])
    unions = {}
    for method in ('IR04_DIRECT_GROUP','IR05_VERIFIED_REPAIR'):
        for h in proposal['decisions'][method]['selected']:
            if h['host'] is None:
                row = unions.setdefault(h['digest'],{'hypothesis':h,'requested_by':[]})
                row['requested_by'].append(method)
    if len(unions)>2*binding['specification']['support']['max_applied_operations']:
        raise ValueError('selected union work exceeded the two-arm fixed budget')
    supports = {r['unit']:inputs.units.units[r['unit']].rows for r in incumbents}
    supports.update({digest:np.unique(np.concatenate([inputs.units.units[n].rows for n in row['hypothesis']['units']]))
                     for digest,row in unions.items()})
    best = {}
    observed = {int(f['frame_id']):f for f in observation['frames']}
    for frame in observation['frames']:
        inputs.index.identity(frame['arrays']['path'],frame['arrays'])
        with np.load(frame['arrays']['path'],allow_pickle=False) as arrays:
            rows,valid = arrays['source_rows'],arrays['valid']
        for name,source in supports.items():
            if name.startswith('I:') and frame['units'][name]['visible_pixels']<settings['minimum_full_pixels']:
                continue
            mask = support_mask(rows,valid,source)
            view = choose_view([{'frame_id':frame['frame_id'],'bank':frame['bank'],'mask':mask}],
                minimum_pixels=settings['minimum_full_pixels'] if name.startswith('I:') else repair['classification_min_pixels'],
                minimum_extent=repair['classification_min_bbox_extent'])
            if view is None:
                continue
            bank = frame['bank'] if name.startswith('I:') else 'one_union_view'
            identity = (name,bank)
            previous = best.get(identity)
            if previous is None or (-view['pixels'],view['frame_id'],view['mask_digest']) < (-previous['pixels'],previous['frame_id'],previous['mask_digest']):
                best[identity] = view
    captures = {int(f['frame_id']):f for f in inputs.capture['frames']}
    frames,regions,outputs = {},{},[]
    def add(view, support_hash, kind, role, owner, hypothesis=None):
        mask = view['mask']
        fid = int(view['frame_id'])
        frame = captures[fid]
        capture_root = inputs.capture_path.parent
        rgb = inputs.index.identity(capture_root/frame['rgb_path'],{'sha256':frame['rgb_sha256']})
        depth = inputs.index.identity(capture_root/frame['depth_path'],{'sha256':frame['depth_sha256']})
        resized = resized_shape(mask.shape)
        padded = tuple(n+(-n%32) for n in resized)
        output_hw = tuple(n//4 for n in padded)
        aggregation = 'ANYUP_FULL_EQUAL_COSINE' if role=='INCUMBENT_REREAD' else 'ONE_VIEW_FC_COSINE'
        rkey = region_identity(support_hash,view,kind,inputs.data['FC_physical_model_identity'],
                              inputs.data['FC_text']['sha256'],'ACTUAL_DENSE_GRID_BOUND_AT_EXECUTION',output_hw,aggregation)
        mask_path = dest/'masks'/(view['mask_digest']+'.npz')
        if not mask_path.exists():
            _write_npz(mask_path,{'shape':np.asarray(mask.shape,np.int64),
                                  'packed':np.packbits(mask,bitorder='little')})
        mask_file = inputs.index.identity(mask_path)
        outputs.append(mask_file)
        regions[rkey] = {'region_id':rkey,'support_hash':support_hash,'frame_id':fid,
            'bank':view['bank'],'type':kind,'role':role,'owner':owner,'hypothesis_digest':hypothesis,
            'mask':mask_file,'mask_digest':view['mask_digest'],'pixels':view['pixels'],'bbox':view['bbox'],
            'representation':aggregation,'original_size':list(mask.shape),'resized_size':list(resized),
            'padded_size':list(padded),'output_hw':list(output_hw)}
        group = frames.setdefault(str(fid),{'frame_id':fid,'rgb':rgb,'depth':depth,
                     'observation':observed[fid]['arrays'],'regions':[]})
        group['regions'].append(rkey)
        return rkey
    for row in incumbents:
        row.update(regions=[],missing=[])
        for bank in ('proposal','verification'):
            view = best.get((row['unit'],bank))
            if view is None:
                row['missing'].append({'bank':bank,'type':'FULL','reason':'NO_QUALIFIED_FIXED_BANK_VIEW'})
                continue
            observation_frame = observed[view['frame_id']]
            with np.load(observation_frame['arrays']['path'],allow_pickle=False) as arrays:
                panoptic = arrays['panoptic'] if observation_frame['panoptic_available'] else None
            masks,missing = recognition_masks(view['mask'],panoptic,
                core_min_fraction=settings['core_min_fraction'],
                intersection_min_fraction=settings['frontend_intersection_min_fraction'],
                minimum_pixels=settings['minimum_subregion_pixels'])
            row['missing'].extend({'bank':bank,'frame_id':view['frame_id'],**m} for m in missing)
            for item in masks:
                _,bbox = qualified_mask(item['mask'])
                region_view = {**view,'mask':item['mask'],'mask_digest':item['mask_digest'],
                               'pixels':item['pixels'],'bbox':bbox}
                row['regions'].append(add(region_view,row['support_hash'],item['type'],'INCUMBENT_REREAD',row['owner']))
    for digest,row in unions.items():
        view = best.get((digest,'one_union_view'))
        row['region'] = None if view is None else add(view,row['hypothesis']['support_hash'],
            'UNION_FULL','UNION_FC',row['hypothesis']['output_owner'],hypothesis=digest)
        row['status'] = 'KEEP_NO_QUALIFIED_UNION_VIEW' if view is None else 'UNION_RECOGNITION_REQUIRED'
    result = seal({'status':'REGIONS_LOCKED','scene':scene,'input_identity':key,
        'proposal_identity':proposal['identity'],'observation_identity':observation['identity'],
        'support_identity':inputs.units.identity,'incumbents':incumbents,'unions':unions,
        'regions':regions,'frames':frames,'outputs':list({r['path']:r for r in outputs}.values()),
        'selected_original_incumbents':len(incumbents),'selected_union_groups':len(unions),
        'required_FC_frames':len(frames),'required_AnyUp_frames':sum(any(regions[r]['role']=='INCUMBENT_REREAD'
            for r in f['regions']) for f in frames.values()),'blocked':observation['blocked'],
        'GT_used':False,'new_neural_inference':0,'elapsed_seconds':time.perf_counter()-begin})
    atomic_write_json(path,result)
    inputs.index.write_memo(root/'inputs'/scene/'verifications.json')
    print('PLANNED_REGIONS',scene,'incumbents',len(incumbents),'unions',len(unions),'frames',len(frames),flush=True)
    return result
