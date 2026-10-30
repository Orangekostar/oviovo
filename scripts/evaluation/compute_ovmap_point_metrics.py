"""Read-only extension of frozen AP predictions with common full-point metrics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]
import numpy as np
from static_ovmap.benchmark_completion_v2.common import verified,write,digest_file,canonical_digest,read
from static_ovmap.benchmark_point_metrics import project_masks,confusion_counts,summarize

METHODS=('OVI-MAP','OpenMask3D','Open3DIS','MaskClustering')
PROJECTION='MAX_FROZEN_AP_CONFIDENCE_THEN_EARLIEST_ORIGINAL_COLUMN;VALID_AP_CLASSES_ONLY;UNCOVERED_UNKNOWN'


def descriptor(path):
    path=Path(path)
    return dict(path=str(path),sha256=digest_file(path),bytes=path.stat().st_size)


def check(row):
    assert Path(row['path']).stat().st_size==row['bytes'] and digest_file(row['path'])==row['sha256'],row['path']


def validate_standard_ap(score,pred,lock):
    assert score['prediction_identity']==pred['prediction_identity']
    if 'alias_of' not in score:
        assert score['prediction']==pred['identity']
        return
    source=lock['methods'][score['alias_of']];original=verified(source['path'])
    assert original['identity']==source['identity']==score['prediction']
    assert original['prediction_identity']==pred['prediction_identity']
    check(original['arrays']);check(pred['arrays'])
    if original['arrays']['sha256']!=pred['arrays']['sha256']:
        with np.load(original['arrays']['path'],allow_pickle=False) as a,np.load(pred['arrays']['path'],allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            assert all(np.array_equal(a[k],b[k]) for k in a.files)


def prepare(root,out):
    assert not (out/'inputs.json').exists()
    protocol=verified(root/'protocol_lock.json');store=verified(root/'result_store.json')
    result={};jobs={};missing=[]
    for benchmark,scope in protocol['scopes'].items():
        for scene in scope['scenes']:
            gt_path=root/'benchmark/GT'/benchmark/scene/'receipt.json';gt=verified(gt_path)
            assert gt['full_GT_denominator'] and gt['GT_filtered_by_prediction'] is False
            jobs[benchmark+':'+scene]=dict(benchmark=benchmark,scene=scene,class_ids=scope['instance_ids'],
                GT_receipt=descriptor(gt_path),GT_identity=gt['identity'],GT=gt['GT'],target_points=gt['target_points'],predictions={})
        names=[(m,'external:'+m+':'+benchmark) for m in METHODS]
        names += [(f'{v}_seed{s}','standard:'+benchmark+f':{v}_seed{s}')
                  for v in ('U00','U10','U01','U11') for s in (17,29)]
        for method,key in names:
            group=benchmark+':'+method
            if key not in store['results']:
                missing.append(dict(benchmark=benchmark,method=method,reason='COMPLETE_FROZEN_PREDICTIONS_NOT_PRESENT'))
                continue
            ap=store['results'][key];path=root/ap['source']['path'];pool=verified(path)
            assert pool['identity']==ap['source']['identity'] and pool['status']=='DONE'
            assert pool['ordered_scenes']==scope['scenes'] and pool.get('full_scene_coverage',True)
            result[group]=dict(benchmark=benchmark,method=method,AP=ap['metrics'],AP_receipt=descriptor(path),
                AP_identity=pool['identity'],ordered_scenes=scope['scenes'])
            for scene in scope['scenes']:
                job=jobs[benchmark+':'+scene]
                if method.startswith('U'):
                    lock=verified(root/'benchmark/predictions'/benchmark/scene/'receipt.json')
                    pred_path=Path(lock['methods'][method]['path']);pred=verified(pred_path)
                    assert pred['identity']==lock['methods'][method]['identity']
                    score=verified(root/'benchmark/evaluation'/benchmark/scene/'receipt.json')['rows'][method]
                    validate_standard_ap(score,pred,lock)
                    assert score['GT']==job['GT_identity']
                    item=dict(kind='CSR',manifest=descriptor(pred_path),identity=pred['identity'],arrays=pred['arrays'])
                    assert pred['target_points']==job['target_points']
                else:
                    score=verified(root/'external'/method/'evaluation'/benchmark/scene/'receipt.json')
                    assert score['identity']==pool['source_scenes'][scene]['score']
                    assert score['full_GT_denominator']
                    if method=='Open3DIS' and benchmark=='scannet200_val':
                        assert score['GT_receipt']==job['GT_identity']
                        assert all(score['GT'][k]==job['GT'][k] for k in ('path','sha256','bytes'))
                        item=dict(kind='OPEN3DIS_NATIVE',arrays=score['prediction'],identity=canonical_digest(score['prediction']),
                            AP_scene_score=score['identity'],confidence='PINNED_AUTHOR_EVALUATION_CONSTANT_1')
                    else:
                        pred_path=root/'external'/method/'predictions'/benchmark/scene/'manifest.json';pred=verified(pred_path)
                        assert score['prediction']==pred['identity'] and pred['target_points']==job['target_points']
                        gt_path=root/('benchmark' if method=='OVI-MAP' else 'external/'+method)/'GT'/benchmark/scene/'receipt.json'
                        other=verified(gt_path)
                        assert score['GT']==other['identity'] and other['GT']['sha256']==job['GT']['sha256']
                        item=dict(kind='CSR',manifest=descriptor(pred_path),identity=pred['identity'],arrays=pred['arrays'])
                # Freeze all supported predictions before this extension opens GT labels.
                check(item['arrays']);job['predictions'][method]=item
    sources={str(p):digest_file(p) for p in [Path(__file__),REPO/'src/static_ovmap/benchmark_point_metrics.py',
        REPO/'src/static_ovmap/benchmark_completion_v2/external.py']}
    protected={str(root/p):digest_file(root/p) for p in ['protocol_lock.json','model_lock.json','result_store.json',
        'tables.json','terminal.json','publication/final.json']}
    config=write(out/'inputs.json',dict(status='DONE',protocol=protocol['identity'],result_store=store['identity'],
        jobs=jobs,groups=result,missing_groups=missing,sources=sources,protected=protected,
        projection=PROJECTION,semantic_scope='COMMON_AP_OBJECT_CLASS_VOCABULARY;SCANNET198_REPLICA48',
        not_native_200_class_semantic_benchmark=True,macro_classes='GT_PRESENT_AFTER_POOLED_CONFUSION',
        new_model_calls=0,GT_used_to_choose_overlap_policy=False,all_available_predictions_frozen_before_extension_GT_read=True))
    print('POINT_INPUTS_FROZEN',config['identity'],len(jobs),sum(len(x['predictions']) for x in jobs.values()),flush=True)


def score_scene(args):
    out,job,source_hash=args;out=Path(out)
    path=out/'scenes'/job['benchmark']/(job['scene']+'.json')
    identity=canonical_digest(dict(job=job,source=source_hash,projection=PROJECTION))
    for p in job['predictions'].values():
        check(p['arrays'])
        if p['kind']=='CSR':check(p['manifest'])
    check(job['GT']);check(job['GT_receipt'])
    if path.exists():
        old=verified(path);assert old['input_identity']==identity;return old
    gt=np.loadtxt(job['GT']['path'],dtype=np.int64)//1000
    assert gt.shape==(job['target_points'],)
    rows={}
    for method,pred in job['predictions'].items():
        if pred['kind']=='CSR':
            with np.load(pred['arrays']['path'],allow_pickle=False) as arrays:
                params={k:arrays[k] for k in ['indptr','indices','labels','confidence']}
                semantic=project_masks(**params,target_points=job['target_points'],class_ids=job['class_ids'])
        else:
            from static_ovmap.benchmark_completion_v2.external import decode_open3dis
            p=decode_open3dis(pred['arrays']['path'],target_points=job['target_points'],class_ids=job['class_ids'])
            # The pinned author evaluates every proposal at confidence1. Stable
            # first column wins; this uses no archive confidence or GT labels.
            semantic=np.zeros(job['target_points'],np.int64)
            for i,label in enumerate(p['pred_classes']):
                semantic[(semantic==0)&(p['pred_masks'][:,i]!=0)]=label
        matrix=confusion_counts(gt,semantic,job['class_ids']);summary=summarize(matrix,job['class_ids'])
        assert matrix.sum()==np.isin(gt,job['class_ids']).sum()
        rows[method]=dict(prediction=pred['identity'],confusion=matrix.tolist(),metrics=summary,
            projected_semantic_sha256=__import__('hashlib').sha256(semantic.tobytes()).hexdigest())
    record=write(path,dict(status='DONE',input_identity=identity,benchmark=job['benchmark'],scene=job['scene'],
        GT=job['GT'],GT_identity=job['GT_identity'],rows=rows,projection=PROJECTION,new_model_calls=0))
    print('POINT_SCENE_DONE',job['benchmark'],job['scene'],len(rows),flush=True)
    return record


def run(root,out,workers):
    config=verified(out/'inputs.json')
    for p,sha in config['sources'].items():assert digest_file(p)==sha,p
    source_hash=canonical_digest(config['sources'])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(score_scene,[(str(out),job,source_hash) for job in config['jobs'].values()]):pass
    for p,sha in config['protected'].items():assert digest_file(p)==sha,p
    print('POINT_SCORING_CLOSED',flush=True)


def collect(out):
    config=verified(out/'inputs.json');groups={}
    for p,sha in config['sources'].items():assert digest_file(p)==sha,p
    source_hash=canonical_digest(config['sources'])
    for key,group in config['groups'].items():
        rows=[];matrix=None;benchmark=group['benchmark'];method=group['method']
        for scene in group['ordered_scenes']:
            p=out/'scenes'/benchmark/(scene+'.json');row=verified(p);x=row['rows'][method]
            job=config['jobs'][benchmark+':'+scene]
            assert row['input_identity']==canonical_digest(dict(job=job,source=source_hash,projection=PROJECTION))
            assert x['prediction']==job['predictions'][method]['identity']
            current=np.asarray(x['confusion'],np.int64);matrix=current.copy() if matrix is None else matrix+current
            rows.append(dict(scene=scene,receipt=row['identity'],prediction=x['prediction']))
        ids=config['jobs'][benchmark+':'+group['ordered_scenes'][0]]['class_ids']
        check(group['AP_receipt'])
        groups[key]=dict(group,metrics=dict(group['AP'],**{k:v for k,v in summarize(matrix,ids).items() if k!='per_class'}),
            semantic_per_class=summarize(matrix,ids)['per_class'],confusion=matrix.tolist(),source_scenes=rows,
            completed_scenes=len(rows),expected_scenes=len(group['ordered_scenes']),status='DONE')
    result=write(out/'results.json',dict(status='DONE' if not config['missing_groups'] else 'PARTIAL',
        inputs=config['identity'],groups=groups,missing_groups=config['missing_groups'],projection=PROJECTION,
        semantic_scope=config['semantic_scope'],no_scene_metric_averaging=True,new_model_calls=0))
    main=[];ablation=[]
    for benchmark in ['scannet200_val','replica48']:
        for method in [*METHODS,'U00','U10','U01','U11']:
            keys=[benchmark+':'+method] if method in METHODS else [benchmark+':'+method+f'_seed{s}' for s in (17,29)]
            rows=[groups[k] for k in keys if k in groups]
            values={m:float(np.mean([row['metrics'][m] for row in rows])) if len(rows)==len(keys) else None
                    for m in ['miou','macc','oacc','apall','ap50','ap25']}
            item=dict(benchmark=benchmark,method='Ours' if method=='U11' else method,metrics=values,
                source_groups=keys,complete=len(rows)==len(keys),scenes=len(rows[0]['ordered_scenes']) if rows else 0)
            (main if method in METHODS or method=='U11' else ablation).append(item)
    report=write(out/'comparison.json',dict(status=result['status'],results=result['identity'],main=main,ablation=ablation,
        semantic_scope=config['semantic_scope'],projection=PROJECTION,units='FRACTION;DISPLAY_PERCENT',
        aggregation='SUM_SCENE_CONFUSIONS_PER_SEED_THEN_MACRO;OURS_MEAN_TWO_SEED_METRICS',missing=config['missing_groups']))
    lines=['# Common point metrics and official instance AP','',
        'Semantic metrics project the same frozen AP masks to the original full point order. ScanNet uses the 198 AP object classes, Replica uses48. This is an instance-output projection evaluation, not native ScanNet200 semantic segmentation. Overlaps use frozen AP confidence, then earliest original column. Uncovered valid GT points count as errors. mIoU/mAcc average GT-present classes after pooling scene confusions; oAcc is overall point accuracy. Ours averages independently pooled seed17/29 metrics. AP is retained from the exact existing official full-scene pools.','',
        '| Dataset | Method | Scenes | mIoU | mAcc | oAcc | AP | AP50 | AP25 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in main+ablation:
        values=['--' if v is None else f'{100*v:.2f}' for v in x['metrics'].values()]
        lines.append('| '+' | '.join([x['benchmark'],x['method'],str(x['scenes']),*values])+' |')
    lines += ['', 'Missing full predictions remain explicit; no IoU/accuracy is inferred from author AP.']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print('POINT_COMPARISON_READY',report['identity'],report['status'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--work-root',type=Path,required=True);p.add_argument('--phase',choices=['prepare','score','pool'],required=True)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=3);args=p.parse_args()
    if args.phase=='prepare':prepare(args.output_root,args.work_root)
    elif args.phase=='score':run(args.output_root,args.work_root,args.workers)
    else:collect(args.work_root)

if __name__=='__main__':main()
