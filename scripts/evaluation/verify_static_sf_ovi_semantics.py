#!/usr/bin/env python3
"""Verify all six saved label-only outputs, without model or evaluator reruns."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from src.static_ovmap.cached_semantic_transfer import METHODS,assert_fixed
from src.static_ovmap.cache_io import sha256_file
from scripts.evaluation.run_static_sf_ovi_semantics import read,write,identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg=read(args.config);root=Path(cfg['new_output_root']);pred=root/'predictions';ev=root/'evaluation';performance=read(ev/'performance.json')
    assert len(performance)==6 and all(r['status']=='COMPLETE' for r in performance)
    for run in cfg['runs']:
        source=Path(cfg['recorded_roots_to_verify']['frozen_prediction_root'])/run;base=read(source/'AT_O_AREA.json')
        owners=np.load(base['owner_path']);reference=np.load(Path(cfg['recorded_roots_to_verify']['reference_evaluation_root'])/run/'projection.npz')
        projected=np.where(reference['matched'],owners[reference['nearest']],0)
        assert np.array_equal(projected,np.load(ev/run/'owners.npy'))
        ledger=read(pred/run/'receiver_decisions.json');assert {r['receiver'] for r in ledger}==set(base['kept'])
        for method in METHODS:
            doc=read(pred/run/(method+'.json'));assert_fixed(base,doc);assert sha256_file(doc['owner_path'])==doc['owner_sha256']
            labels=np.asarray(doc['labels']);expected=np.zeros(len(projected),np.int64);positive=projected>0;expected[positive]=labels[projected[positive]-1]
            assert np.array_equal(expected,np.load(ev/run/method/'semantic.npy'))
            for receiver in ledger:assert labels[receiver['receiver']]==receiver['final_labels'][method]
            row=next(r for r in performance if r['run']==run and r['condition']==method)
            assert row['unique']['trace_parity']['ap_exact'] and row['unique']['trace_parity']['pr_and_fn_exact']
    evaluated=sum(r['execution_role']=='NEW_PAYLOAD_EVALUATION' for r in performance)
    assert evaluated==read(ev/'evaluation_manifest.json')['actual_new_payload_evaluations']
    for receipt in [read(pred/'prediction_manifest.json'),read(ev/'evaluation_manifest.json')]:
        assert receipt['new_inference']==0
    source={str(p.relative_to(ROOT)):identity(p) for p in [ROOT/'src/static_ovmap/cached_semantic_transfer.py',*sorted((ROOT/'scripts/evaluation').glob('*sf_ovi_semantics.py'))]}
    write(ROOT/cfg['delivery']['small_artifacts']/'execution_receipts.json',{'status':'VERIFIED','conditions':6,'actual_evaluated_payloads':evaluated,
          'reused_new_conditions':6-evaluated,'labels_geometry_ranks_and_semantic_arrays':'EXACT','new_inference':0,'source':source,'command':[sys.executable,*sys.argv]})
    print(f'VERIFIED: 6 conditions, {evaluated} actual payload evaluations, zero new inference')


if __name__=='__main__':main()
