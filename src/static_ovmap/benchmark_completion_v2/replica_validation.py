"""Replica first-scene agreement with the pinned Open3DIS author evaluator."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .common import (ConsumedInputs, StageFailure, canonical_digest, verified,
                     write, source_identity)


def author_evaluator(binding):
    from .external import PINS
    from static_ovmap.preservation_routing.scorer_compat import adapt
    root=Path(binding['output_root'])
    protocol=verified(root/'protocol_lock.json')
    scope=protocol['scopes']['replica48']
    source=Path(protocol['official_sources']['evaluator_replica']['path'])
    repo=root/'assets/external/Open3DIS'
    utility=repo/'open3dis/evaluation/instance_eval_util.py'
    index=ConsumedInputs(root)
    sources=[index.identity(source,protocol['official_sources']['evaluator_replica']),index.identity(utility)]
    pin=PINS['Open3DIS'][1]
    # Validate only the two consumed author files against the source pin.
    for path,name in ((source,'scannetv2_inst_eval.py'),(utility,'instance_eval_util.py')):
        original=subprocess.check_output(['git','show',pin+':open3dis/evaluation/'+name],cwd=repo)
        if hashlib.sha256(original).hexdigest()!=index.identity(path)['sha256']:
            raise ValueError('Replica author evaluator source differs from the pinned implementation')
    namespace=adapt(dict(np=np,deepcopy=deepcopy,json=json))
    for path,names in ((utility,{'Instance','get_instances'}),(source,{'ScanNetEval'})):
        nodes=[node for node in ast.parse(path.read_text()).body
               if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
        if {node.name for node in nodes}!=names:
            raise ValueError('Replica author evaluator AST boundary changed')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    author=namespace['ScanNetEval'](scope['instance_names'],dataset_name='replica')
    if (not np.array_equal(author.valid_class_ids,scope['instance_ids'])
            or not np.array_equal(author.ious,scope['IoU_thresholds'])
            or author.min_region_sizes[0]!=scope['minimum_region_vertices']):
        raise ValueError('Replica author class, IoU or full-GT region axes differ from the protocol')
    index.save()
    return author,sources


def verify_scene(binding, scene):
    root=Path(binding['output_root'])
    freeze_path=root/'benchmark/predictions/replica48'/scene/'receipt.json'
    if not freeze_path.is_file():
        raise StageFailure('Replica author agreement requires actual frozen predictions',missing=[str(freeze_path)])
    freeze=verified(freeze_path)
    if (freeze['scene']!=scene or freeze['benchmark']!='replica48'
            or not freeze['all_available_predictions_frozen_before_new_GT_scoring']):
        raise ValueError('Replica author agreement requires this scene complete available prediction freeze')
    score=verified(root/'benchmark/evaluation/replica48'/scene/'receipt.json')
    gt=verified(root/'benchmark/GT/replica48'/scene/'receipt.json')
    if (score['prediction_lock']!=freeze['identity'] or gt['prediction_freeze']!=freeze['identity']
            or score['GT']!=gt['identity']):
        raise ValueError('Replica common score and GT do not bind the actual prediction freeze')
    author,sources=author_evaluator(binding)
    identity=canonical_digest(dict(score=score['identity'],GT=gt['identity'],sources=sources,
                                   producer=source_identity(('replica_validation',))))
    path=root/'benchmark/evaluation/replica48'/scene/'G3_official_entry_agreement.json'
    if path.is_file():
        prior=verified(path)
        if prior['input_identity']!=identity:
            raise ValueError('Completed Replica author agreement changed; preserve before affected repair')
        return prior
    from .replica_io import _tuple_path
    from .standard_export import official_prediction
    index=ConsumedInputs(root)
    source=_tuple_path(binding,scene)
    index.identity(source,gt['inputs'][0])
    import torch
    value=torch.load(source,map_location='cpu',weights_only=False)
    if not isinstance(value,(list,tuple)) or len(value)!=4:
        raise ValueError('Replica author agreement requires the original four-field GT tuple')
    semantic,instances=(np.asarray(value[i]) for i in (2,3))
    checked={}
    maximum_error=0.
    for method,descriptor in freeze['methods'].items():
        manifest=verified(descriptor['path'])
        if manifest['identity']!=descriptor['identity']:
            raise ValueError('Replica prediction changed before author agreement')
        prediction=official_prediction(manifest)
        preds=[dict(scan_id=scene,label_id=int(label),conf=float(conf),pred_mask=prediction['pred_masks'][:,i])
               for i,(label,conf) in enumerate(zip(prediction['pred_classes'],prediction['pred_scores'],strict=True))]
        gt2pred,pred2gt=author.assign_instances_for_scan(preds,semantic.copy(),instances.copy())
        ap,rc=author.evaluate_matches({scene:dict(gt=gt2pred,pred=pred2gt)})
        averages=author.compute_averages(ap,rc)
        expected=np.array([averages[key] for key in ('all_ap','all_ap_50%','all_ap_25%')])
        actual=np.array([score['rows'][method]['metrics'][key] for key in ('apall','ap50','ap25')],dtype=float)
        if not np.allclose(expected,actual,atol=1e-12,rtol=0,equal_nan=True):
            raise ValueError('Replica common masks/GT evaluator differs from the unchanged author: '+method)
        finite=np.isfinite(expected)&np.isfinite(actual)
        error=float(np.max(np.abs(expected[finite]-actual[finite]))) if finite.any() else 0.
        maximum_error=max(maximum_error,error)
        checked[method]=dict(actual_common=[None if np.isnan(v) else float(v) for v in actual],
                             actual_author=[None if np.isnan(v) else float(v) for v in expected],
                             prediction=manifest['identity'],max_absolute_error=error)
    index.save()
    return write(path,dict(status='DONE',formal_scene=scene,benchmark='replica48',input_identity=identity,
        prediction_lock=freeze['identity'],GT=gt['identity'],checked_methods=checked,
        author_evaluator_sources=sources,author_entry='UNCHANGED_ScanNetEval_assign_instances_for_scan_evaluate_matches_compute_averages',
        max_absolute_error=maximum_error,agreement_tolerance=1e-12,synthetic=False,full_original_mesh_GT=True,
        new_GPU_calls=0))
