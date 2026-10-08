"""New whole-unit partition builder; inherited append-only safeguards stay intact."""

from pathlib import Path

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.module_validation.evaluation import PredictionPayload
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import native_ranks,save_prediction
from static_ovmap.recovery_wave2.binding import read

from .binding import load_scene,seal


def construct_partition(inputs, method, operations, *, union_classes=None, relabels=None, lock=True,
                        parent_prediction_keys=None):
    union_classes,relabels = union_classes or {},relabels or {}
    base,anchor = inputs.g1,inputs.d2
    if base.geometry != anchor.geometry or len(inputs.raw) != base.geometry.source_row_count:
        raise ValueError('new partition must preserve the original source geometry')
    owners,semantics = base.owner_ids.copy(),base.semantic_labels.copy()
    units = inputs.units.units
    used,applied,canceled = set(),[],[]
    declared = np.zeros(len(owners),bool)
    old_labels = owner_labels(anchor)
    maximum = int(max(inputs.raw.max(initial=0),anchor.owner_ids.max(initial=0),base.owner_ids.max(initial=0)))
    for operation in operations:
        members = operation['units']
        if not used.isdisjoint(members):
            raise ValueError('provisional operations must use disjoint original units')
        used.update(members)
        if any(name not in units for name in members):
            raise ValueError('operation leaves the fixed whole-unit registry')
        residuals = [units[name] for name in members if units[name].kind=='C']
        incumbents = [units[name] for name in members if units[name].kind=='I']
        if not residuals or len(incumbents)>1:
            raise ValueError('repair cannot split a residual or merge two original incumbents')
        host = operation['host']
        if host is not None:
            if len(incumbents)!=1 or incumbents[0].owner!=host or operation['output_owner']!=host:
                raise ValueError('attachment must retain its single original host identity')
            label = old_labels[host]
        else:
            if incumbents or operation['fresh_owner'] != operation['output_owner'] or operation['fresh_owner'] <= maximum:
                raise ValueError('fresh union ID collides with a historical owner or incumbent')
            label = union_classes.get(operation['digest'])
            if label is None:
                canceled.append({'digest':operation['digest'],'units':members,'status':'KEEP_UNION_CLASS_UNAVAILABLE'})
                continue
            if label not in inputs.valid_ids:
                raise ValueError('union class leaves the unchanged text vocabulary')
        rows = np.concatenate([u.rows for u in residuals])
        if (np.any(anchor.owner_ids[rows]!=0) or np.any(inputs.raw[rows]==0)
                or np.any(declared[rows]) or len(np.unique(rows)) != len(rows)):
            raise ValueError('repair cannot transfer original incumbent or raw-zero rows')
        owners[rows],semantics[rows] = operation['output_owner'],label
        declared[rows] = True
        applied.append({'digest':operation['digest'],'units':members,'output_owner':operation['output_owner'],
                        'host':host,'class':label,'source_rows':len(rows),
                        'physical_area':float(inputs.units.vertex_weights[rows].sum())})
    class_transitions = []
    for owner,label in sorted(relabels.items()):
        owner,label = int(owner),int(label)
        if owner not in old_labels or label not in inputs.valid_ids:
            raise ValueError('rereading may relabel only original incumbent IDs in the frozen vocabulary')
        support = owners == owner
        if not support.any():
            raise ValueError('original incumbent identity disappeared')
        before = int(semantics[support][0])
        semantics[support] = label
        if before != label:
            declared[support] = True
            class_transitions.append({'owner':owner,'old_class':before,'class':label,
                                      'evidence_anchor':units[f'I:{owner}'].support_hash})
    original = anchor.owner_ids>0
    if (not np.array_equal(owners[original],anchor.owner_ids[original])
            or not np.array_equal(owners[~declared],base.owner_ids[~declared])
            or not np.array_equal(semantics[~declared],base.semantic_labels[~declared])
            or not np.array_equal(owners[inputs.raw==0],base.owner_ids[inputs.raw==0])
            or not np.array_equal(semantics[inputs.raw==0],base.semantic_labels[inputs.raw==0])):
        raise ValueError('partition changed rows outside explicitly declared operations')
    labels = {int(owner):int(semantics[np.flatnonzero(owners==owner)[0]]) for owner in np.unique(owners) if owner>0}
    ranks = native_ranks(owners,labels,inputs.nearest,inputs.matched)
    costs = {**base.logical_cost,'repair_applied_operations':len(applied),'incumbent_class_changes':len(class_transitions)}
    if not lock and parent_prediction_keys is None:
        raise ValueError('cold output construction requires prebound parent keys outside the timer')
    keys = parent_prediction_keys or {'G1':base.prediction_key,'D2':anchor.prediction_key}
    metadata = {'parent_prediction_key':keys['G1'],'original_D2_prediction_key':keys['D2'],
        'method_recipe':method,'fixed_surface':True,'single_exclusive_partition':True,
        'original_incumbent_identity_preserved':True,'moved_incumbent_rows':0,
        'applied_operations':applied,'canceled_operations':canceled,'class_transition':class_transitions,
        'owner_semantic_decisions':{str(k):v for k,v in sorted(labels.items())},
        'semantic_evidence_on_original_incumbent_support':True,'official_current_class_ranks_recomputed':True}
    # Hash fields are populated only outside a caller's cold timer.
    payload = PredictionPayload(method,'COMBO',base.scene_id,base.geometry,owners,semantics,ranks,costs,metadata)
    if lock:
        payload.lock()
    moved = owners != base.owner_ids
    audit = {'applied':applied,'canceled':canceled,'class_transitions':class_transitions,
        'moved_residual_rows':int(moved.sum()),'moved_residual_area':float(inputs.units.vertex_weights[moved].sum()),
        'moved_incumbent_rows':0,'disappeared_owner_ids':sorted(set(map(int,np.unique(base.owner_ids)))-set(map(int,np.unique(owners)))),
        'host_growth':[{'host':row['host'],'added_rows':row['source_rows'],'added_area':row['physical_area']}
                       for row in applied if row['host'] is not None]}
    return payload,audit


def predict_study(binding, *, scenes=None):
    root = Path(binding['output_root'])
    methods = [row['id'] for row in binding['specification']['methods']]
    receipts = []
    for names in ([scenes] if scenes is not None else binding['cohorts'].values()):
        for scene in names:
            inputs = load_scene(binding,scene)
            structural = read(root/'proposals'/scene/'receipt.json')
            semantic = read(root/'recognition'/scene/'decisions.json')
            unions = {key:item['class'] for key,item in semantic['unions'].items() if item['available']}
            rereads = semantic['incumbents']
            results = {}
            for name in methods:
                blocks = [item for item in structural['blocked'] if name in item['methods']]
                if blocks:
                    results[name] = {'status':'BLOCKED_MISSING_FRONTEND_EVIDENCE','blocks':blocks}
                    continue
                if name in ('IR00_D2','IR01_G1'):
                    imported = binding['scenes'][scene]['predictions'][name]
                    results[name] = {'status':'PREDICTION_LOCKED','manifest':imported['prediction_manifest'],
                                     'prediction_key':imported['prediction_key'],'source_kind':'EXACT_PARENT_ALIAS'}
                    continue
                structural_arm = 'IR05_VERIFIED_REPAIR' if name=='IR08_COMBINATION' else name
                operations = structural['decisions'].get(structural_arm,{}).get('selected',[])
                semantic_arm = ('IR06_class' if name=='IR06_ANYUP_REREAD' else 'IR07_class')
                relabels = ({int(owner):row[semantic_arm] for owner,row in rereads.items()
                            if row[semantic_arm]!=row['old_class']}
                           if name in ('IR06_ANYUP_REREAD','IR07_BOUNDARY_STABLE','IR08_COMBINATION') else {})
                payload,audit = construct_partition(inputs,name,operations,union_classes=unions,relabels=relabels)
                manifest = save_prediction(payload,root/'predictions'/scene/name)
                results[name] = {'status':'PREDICTION_LOCKED','manifest':str(manifest),
                    'prediction_key':payload.prediction_key,'record_key':payload.record_key,
                    'owner_partition_digest':_array_digest(payload.owner_ids),'audit':audit,
                    'source_kind':'MEASURED_NEW_PARTITION' if payload.prediction_key!=inputs.g1.prediction_key else 'MEASURED_KEEP'}
            receipt = seal({'status':'PREDICTIONS_LOCKED','scene':scene,'structural_identity':structural['identity'],
                            'semantic_identity':semantic['identity'],'methods':results})
            atomic_write_json(root/'predictions'/scene/'receipt.json',receipt)
            receipts.append(receipt)
    return receipts
