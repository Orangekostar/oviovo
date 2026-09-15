#!/usr/bin/env python3
"""Verify final cached payload isolation without any new model inference."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from src.static_ovmap.candidate_semantics import score_views
from src.static_ovmap.object_arbitration import query_quality
from scripts.evaluation.run_static_object_decoupling import read,write,identity


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=read(args.config);old=read(ROOT/cfg['existing_asset_configs'][0])
    root=Path(cfg['runtime_assets_to_resolve']['new_output_root']);pred=root/'predictions';out=root/'evaluation'
    rows=read(out/'performance.json');initial=read(root/'evaluation_initial/performance.json')
    assert len(rows)==8 and all(r['status']=='COMPLETE' for r in rows)
    for a,b in zip(rows,initial):
        for field in ['semantic','unique_high_iou_canonical_diagnostic']:
            assert a[field]==b[field]
        for family in ['unique','overlapping']:assert a[family]['released']==b[family]['released']
        assert a['cache_keys']['active']!=a['cache_keys']['unique']
    text=np.load(cfg['runtime_assets_to_resolve']['native_text_cache']);targets=0;features=0
    for run in cfg['runs']:
        source=Path(cfg['recorded_roots_to_verify']['frozen_prediction_root'])/run
        t0=np.load(next(r['t0'] for r in old['runs'] if r['id']==run));records=read(source/'candidates.json')
        query_quality(records,t0['query_ids'],t0['objectness_mask_scores'])
        semantics=read(pred/run/'semantics.json')
        assert len(semantics['targets'])<=128 and semantics['attempted_view_requests']<=384 and semantics['crop_inputs']<=2304
        for target in semantics['targets']:
            feats=[];weights=[]
            for view in target['views']:
                if 'abstention' in view:continue
                feature=np.load(pred/'semantic_cache'/(view['key']+'.npy'))
                assert feature.shape==(1024,) and np.isfinite(feature).all()
                feats.append(feature);weights.append(view['visible_pixels']);features+=1
            decision=score_views(feats,weights,text['text'],text['valid_ids'],records[target['candidate']]['original_class_id'],
                       str(text['feature_space_id']),str(text['feature_space_id']),canonical=text['canonical'])
            assert decision['class_id']==target['class_id']
            assert np.allclose(decision['aggregate_cosine'],target['aggregate_cosine'],rtol=0,atol=1e-14)
            assert np.allclose(decision['per_view_cosine'],target['per_view_cosine'],rtol=0,atol=1e-7)
            targets+=1
        for name in ['owners.npy','semantic.npy']:
            assert np.array_equal(np.load(out/run/'OD_E3_OBJECT_SEMANTIC'/name),np.load(out/run/'OD_E4_FINAL_RANK'/name))
    manifest=read(pred/'prediction_manifest.json');assert manifest['crop_inputs']==6*manifest['encoder_batches']<=4608
    result={'status':'VERIFIED','cells':8,'initial_final_metrics_exact':True,'semantic_targets_recomputed_without_encoder':targets,
         'score_recompute_tolerance':{'aggregate':1e-14,'float32_per_view':1e-7},
         'view_features_checked':features,'E3_E4_projected_owners_and_semantics_exact':True,'new_image_forwards':0,
         'final_source_identity':{str(p.relative_to(ROOT)):identity(p) for p in [
          ROOT/'src/static_ovmap/object_arbitration.py',ROOT/'src/static_ovmap/candidate_semantics.py',ROOT/'src/static_ovmap/final_instance_ranking.py',
          ROOT/'scripts/evaluation/run_static_object_decoupling.py',ROOT/'scripts/evaluation/evaluate_static_object_decoupling.py',Path(__file__)]},
         'command':[sys.executable,*sys.argv]}
    write(ROOT/cfg['delivery']['small_artifacts']/'final_verification.json',result)
    print(f'VERIFIED: {len(rows)} cells, exact metric parity, {targets} semantic targets, no new image forward')


if __name__=='__main__':main()
