"""Finite actual two-scene FC/output/scorer integration, not scientific map expansion."""
from pathlib import Path
import contextlib
import numpy as np
import torch
from .common import PathResolver,canonical_digest,read,verified,write,_array_digest
from .binding import load_scene
from .data import ObjectLoader
from .models import teacher

def check(binding,fc):
    from static_ovmap.disagreement_query.outputs import build_payload
    from static_ovmap.module_validation.scannet_study import save_prediction,load_prediction
    from .evaluation import partition_evaluator
    from static_ovmap.composition_study.object_evidence import owner_labels
    from static_ovmap.cvpr_compact.evaluation import equivalent_scoring_input,fraction_metrics
    from static_ovmap.m2_reviewer_study.evaluation import official_view
    root=Path(binding['output_root']);parent=Path(binding['lr_parent_root']);path=root/'engineering/two_scene.json'
    if path.exists():return verified(path)
    loader=ObjectLoader(fc,verified(parent/'features/regression.json'));rows={}
    for scene in (binding['cohorts']['replica8'][0],binding['cohorts']['scannet_cf18'][0]):
        manifest=verified(parent/'regression'/scene/'manifest.json');obj=manifest['objects'][0]
        x,_,_=loader.load(obj,0,8)
        with torch.no_grad():z=teacher(x).cpu().numpy()
        record=PathResolver(binding['path_map']).rewrite(verified(parent/'recognition/regression.json')['records'][obj['key']])
        with np.load(record['arrays']['path'],allow_pickle=False) as a:
            old=a['embeddings'][list(a['methods']).index('LR04_FC_8')]
        np.testing.assert_allclose(z,old,rtol=1e-5,atol=1e-6)
        p,original=load_scene(binding,scene)
        payload=build_payload(original.g1,original.d2,original.raw,{},[],original.nearest,original.matched,
                              'PR_ENGINEERING_G1_IDENTITY',{},dict(diagnostic_only=True))
        if payload.prediction_key!=original.g1.prediction_key:raise ValueError('Literal no-relabel output differs from G1')
        folder=root/'engineering/scenes'/scene;manifest_path=save_prediction(payload,folder/'prediction')
        reloaded=load_prediction(manifest_path)
        if reloaded.prediction_key!=payload.prediction_key:raise ValueError('Actual payload save/load differs')
        evaluator=partition_evaluator(original,payload,binding,folder);labels=owner_labels(payload)
        if set(labels)!=set(evaluator.masks):raise ValueError('Recovered positive registry omitted')
        view={str(o):v for o,v in official_view(evaluator.owners,labels,100).items()}
        parent_row=binding['baseline_rows']['PR01_G1'][scene];prior=read(parent_row['evaluation_receipt'])
        if not equivalent_scoring_input(payload,original.g1,evaluator.context,prior['context'],view,prior['view']):
            raise ValueError('Actual output/scorer identity differs')
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'scorer.log').open('w') as stream,contextlib.redirect_stdout(stream):
            score=evaluator.evaluate(labels,'PR_ENGINEERING_G1_IDENTITY','OFFICIAL_CURRENT_CLASS',payload.prediction_key)
        actual=read(score['evaluation_receipt'])
        current=fraction_metrics(actual['metrics'])
        if not all(abs(current[k]-parent_row['metrics'][k])<=1e-10 for k in ('apall','ap50','ap25','miou','macc')):
            raise ValueError('Instance/semantic scorer parity failed')
        rows[scene]=dict(FC8_max_abs=float(np.max(np.abs(z-old))),actual_output=str(manifest_path),
                     unchanged_owner_digest=_array_digest(payload.owner_ids),unchanged_semantic_digest=_array_digest(payload.semantic_labels),
                     ranks=canonical_digest(payload.instance_ranks),actual_positive_registry=len(labels),
                     scoring_receipt=score,scorer_executed=True,scientific_scene_rows=0)
    return write(path,dict(status='COMPLETE',scenes=rows,literal_no_relabel_equals_G1=True,FC8_same_input_reproduced=True,
                          engineering_only=True,scientific_map_expansion=False))
