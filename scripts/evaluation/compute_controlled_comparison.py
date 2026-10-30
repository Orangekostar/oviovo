"""Common object metrics and freshly pooled AP on the fixed Replica8/ScanNet15."""
import argparse
import importlib.util
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO/'src'),str(REPO)]
import numpy as np
from static_ovmap.benchmark_completion_v2.common import verified,write,digest_file,canonical_digest
from static_ovmap.benchmark_point_metrics import summarize
from static_ovmap.controlled_baselines import check,descriptor

METHODS=['OVI-MAP','OpenMask3D','Open3DIS','MaskClustering']


def point_runner():
    spec=importlib.util.spec_from_file_location('original_point_runner',REPO/'scripts/evaluation/compute_ovmap_point_metrics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def missing_source(lock,module):
    for path,sha in lock['sources'].items():assert digest_file(path)==sha,path
    return canonical_digest(dict(lock=lock['identity'],runner=digest_file(module.__file__),
        core=digest_file(REPO/'src/static_ovmap/benchmark_point_metrics.py')))


def validate_point(row,job,source,module):
    assert row['status']=='DONE' and row['GT_identity']==job['GT_identity']
    assert row['input_identity']==canonical_digest(dict(job=job,source=source,projection=module.PROJECTION))
    for method,pred in job['predictions'].items():assert row['rows'][method]['prediction']==pred['identity']


class ReadOnlyInputs:
    """Use the original scorer loader without changing the old shared input index."""
    def __init__(self,root):self.consumed=[]
    def identity(self,path,expected=None):
        item=descriptor(path)
        if expected:
            for k in ['sha256','bytes']:
                if k in expected:assert item[k]==expected[k],str(path)
        self.consumed.append(item);return item
    def save(self):pass


def evaluator(root,benchmark):
    from static_ovmap.benchmark_completion_v2 import standard_score
    from static_ovmap.benchmark_completion_v2.score_compat import adapt
    previous=standard_score.ConsumedInputs;standard_score.ConsumedInputs=ReadOnlyInputs
    try:namespace=standard_score.evaluator(dict(output_root=str(root)),benchmark)
    finally:standard_score.ConsumedInputs=previous
    protocol=verified(root/'protocol_lock.json')
    proof=adapt(namespace,protocol['official_sources']['evaluator_scan']['path'])
    assert proof['AP_AST_unchanged']
    return namespace,proof


def raw_prediction(out,method,scene):
    if method=='OpenMask3D':return out/method/scene/'prediction.npz'
    return out/method/'data/prediction/scannet_controlled'/(scene+'.npz')


def freeze_missing(root,out):
    """All30 model outputs must exist before any new score opens GT labels."""
    scope=verified(out/'scope.json');rows={};points=verified(out.parent/'inputs.json')['jobs']
    for method in ['OpenMask3D','MaskClustering']:
        rows[method]={}
        for scene in scope['scopes']['scannet200_val']:
            path=raw_prediction(out,method,scene);original=descriptor(path)
            with np.load(path,allow_pickle=False) as arrays:
                masks=arrays['pred_masks'];labels=arrays['pred_classes'];conf=arrays['pred_scores'] if 'pred_scores' in arrays else arrays['pred_score']
                assert masks.ndim==2 and masks.shape[0]==points['scannet200_val:'+scene]['target_points']
                assert labels.shape==conf.shape==(masks.shape[1],) and np.isfinite(conf).all()
                assert np.isin(masks,[0,1]).all() and np.issubdtype(labels.dtype,np.integer)
                columns=[np.flatnonzero(masks[:,i]).astype(np.int64) for i in range(masks.shape[1])]
                indptr=np.cumsum([0]+[len(c) for c in columns],dtype=np.int64)
                indices=np.concatenate(columns) if columns else np.zeros(0,np.int64)
            folder=out/'predictions'/method/scene;folder.mkdir(parents=True,exist_ok=True)
            target=folder/'masks.npz'
            if not target.exists():np.savez_compressed(target,indptr=indptr,indices=indices,labels=labels,confidence=conf)
            with np.load(target,allow_pickle=False) as arrays:
                for key,val in dict(indptr=indptr,indices=indices,labels=labels,confidence=conf).items():assert np.array_equal(arrays[key],val)
            manifest=write(folder/'manifest.json',dict(status='DONE',method=method,scene=scene,original=original,
                arrays=descriptor(target),points=masks.shape[0],columns=masks.shape[1],class_axis=scope['taxonomy']['scannet200_val']['instance_ids'],
                confidence='UNCHANGED_ORIGINAL_AUTHOR_OUTPUT',GT_labels_read=False,column_order_unchanged=True,scope=scope['identity']))
            rows[method][scene]=dict(kind='CSR',manifest=descriptor(folder/'manifest.json'),arrays=manifest['arrays'],identity=manifest['identity'])
    write(out/'missing_prediction_lock.json',dict(status='DONE',scope=scope['identity'],predictions=rows,
        completed_scene_methods=30,expected_scene_methods=30,all_missing_predictions_frozen_before_new_GT_scoring=True,
        sources={str(path):digest_file(path) for path in [Path(__file__),REPO/'scripts/evaluation/compute_ovmap_point_metrics.py',
            REPO/'src/static_ovmap/benchmark_point_metrics.py']}))
    print('CONTROLLED_ALL_MISSING_PREDICTIONS_FROZEN',flush=True)


def score_missing(root,out):
    lock=verified(out/'missing_prediction_lock.json');scope=verified(out/'scope.json');assert lock['scope']==scope['identity']
    module=point_runner();source=missing_source(lock,module)
    from static_ovmap.benchmark_completion_v2.standard_score import ap_metrics,_json
    original=verified(out.parent/'inputs.json');namespace,proof=evaluator(root,'scannet200_val')
    for method,scenes in lock['predictions'].items():
        for scene,item in scenes.items():
            check(item['arrays']);check(item['manifest'])
    for scene in scope['scopes']['scannet200_val']:
        job=dict(original['jobs']['scannet200_val:'+scene]);job['predictions']={m:rows[scene] for m,rows in lock['predictions'].items()}
        module.score_scene((str(out/'point_metrics_missing'),job,source))
        for method,rows in lock['predictions'].items():
            item=rows[scene];path=out/'AP_missing'/method/(scene+'.json.gz')
            if path.exists():
                old=verified(path);assert old['prediction']==item['identity'] and old['GT']==job['GT_identity'];continue
            with np.load(item['arrays']['path'],allow_pickle=False) as arrays:
                active=np.flatnonzero(np.isin(arrays['labels'],job['class_ids']))
                masks=np.zeros((job['target_points'],len(active)),np.uint8)
                for column,i in enumerate(active):masks[arrays['indices'][arrays['indptr'][i]:arrays['indptr'][i+1]],column]=1
                pred=dict(pred_masks=masks,pred_classes=arrays['labels'][active],pred_scores=arrays['confidence'][active])
            a,b=namespace['assign_instances_for_scan'](pred,job['GT']['path']);matches={scene:dict(gt=a,pred=b)}
            metrics,classes,_=ap_metrics(namespace,matches)
            write(path,dict(status='DONE',prediction=item['identity'],GT=job['GT_identity'],points=job['target_points'],
                matches=_json(matches),metrics=metrics,per_class=classes,evaluator_compatibility=proof,full_GT_denominator=True))
            print('CONTROLLED_AP_SCENE_DONE',method,scene,flush=True)


def pool(root,out):
    from static_ovmap.benchmark_completion_v2.standard_score import ap_metrics
    scope=verified(out/'scope.json');config=verified(out.parent/'inputs.json');groups={}
    missing_lock=verified(out/'missing_prediction_lock.json') if (out/'missing_prediction_lock.json').exists() else None
    before={p:digest_file(p) for p in config['protected']};evaluator_proofs={}
    assert before==config['protected'],'The original frozen release changed'
    for path,sha in config['sources'].items():assert digest_file(path)==sha,path
    module=point_runner();original_source=canonical_digest(config['sources'])
    if missing_lock:
        assert missing_lock['scope']==scope['identity']
        new_source=missing_source(missing_lock,module)
    for benchmark,scenes in scope['scopes'].items():
        namespace,proof=evaluator(root,benchmark);evaluator_proofs[benchmark]=proof
        point_rows={s:verified(out.parent/'scenes'/benchmark/(s+'.json')) for s in scenes}
        new_points={s:verified(out/'point_metrics_missing/scenes'/benchmark/(s+'.json')) for s in scenes} if benchmark=='scannet200_val' and missing_lock else {}
        for scene in scenes:
            job=config['jobs'][benchmark+':'+scene];validate_point(point_rows[scene],job,original_source,module)
            if new_points:
                new_job=dict(job);new_job['predictions']={m:rows[scene] for m,rows in missing_lock['predictions'].items()}
                validate_point(new_points[scene],new_job,new_source,module)
        for method in METHODS+[v+f'_seed{s}' for v in ['U00','U10','U01','U11'] for s in [17,29]]:
            new=benchmark=='scannet200_val' and method in ['OpenMask3D','MaskClustering']
            if new and not missing_lock:continue
            if not new:check(config['groups'][benchmark+':'+method]['AP_receipt'])
            matches={};matrix=None;inputs=[]
            for scene in scenes:
                point=(new_points if new else point_rows)[scene];entry=point['rows'][method]
                current=np.asarray(entry['confusion'],np.int64);matrix=current.copy() if matrix is None else matrix+current
                if new:
                    score_path=out/'AP_missing'/method/(scene+'.json.gz');score=verified(score_path)
                    assert score['prediction']==entry['prediction'] and score['GT']==point['GT_identity'];match=score
                else:
                    score_path=root/('benchmark' if method.startswith('U') else 'external/'+method)/'evaluation'/benchmark/scene/'receipt.json'
                    record=verified(score_path);score=record['rows'][method] if method.startswith('U') else record
                    if 'matches_file' in score:
                        check(score['matches_file']);match=verified(score['matches_file']['path']);assert match['identity']==score['matches_identity']
                    else:
                        match=verified(score['matches']['path']);assert match['identity']==score['matches']['identity']
                matches[scene]=match['matches'][scene]
                inputs.append(dict(scene=scene,point_metrics=point['identity'],prediction=entry['prediction'],AP_score=descriptor(score_path),matches=match['identity']))
            metrics,classes,_=ap_metrics(namespace,matches)
            # Replica8 is unchanged: exact equality with all12 frozen official AP pools.
            if benchmark=='replica48':assert metrics==config['groups'][benchmark+':'+method]['AP']
            sem=summarize(matrix,scope['taxonomy'][benchmark]['instance_ids'])
            groups[benchmark+':'+method]=dict(benchmark=benchmark,method=method,scenes=scenes,metrics=dict(metrics,**{k:v for k,v in sem.items() if k!='per_class'}),
                semantic_per_class=sem['per_class'],AP_per_class=classes,confusion=matrix.tolist(),source_scenes=inputs)
            print('CONTROLLED_POOL_DONE',benchmark,method,metrics,flush=True)
    main=[];ablation=[]
    for benchmark,scenes in scope['scopes'].items():
        for method in METHODS+['U00','U10','U01','U11']:
            keys=[benchmark+':'+method] if method in METHODS else [benchmark+':'+method+f'_seed{s}' for s in [17,29]]
            values={m:float(np.mean([groups[k]['metrics'][m] for k in keys])) if all(k in groups for k in keys) else None for m in ['miou','macc','ap25','ap50','apall','oacc']}
            item=dict(benchmark=benchmark,method='Ours' if method=='U11' else method,scenes=len(scenes),metrics=values,source_groups=keys)
            (main if method in METHODS+['U11'] else ablation).append(item)
    complete=all(x['metrics']['miou'] is not None for x in main)
    name='comparison_complete_v1' if complete else 'comparison_available_v1'
    result=write(out/(name+'.json'),dict(status='DONE' if complete else 'PARTIAL',scope=scope['identity'],groups=groups,main=main,ablation=ablation,
        metric_scope='COMMON_AP_OBJECT_CLASS_PROJECTION;REPLICA48_SCANNET198;NOT_NATIVE_51_200_SEMANTICS',
        aggregation='SUM_SCENE_CONFUSIONS_BEFORE_MACRO;OFFICIAL_MATCHES_POOLED_OVER_SAME_SCENE_LIST;OURS_MEAN_TWO_SEEDS',
        evaluator_compatibility=evaluator_proofs,old_full_validation_AP_not_reused_for_subset=True,Replica8_AP_exact_parity=True,
        old_GT_exposure_erased=False,imported_Open3DIS_ScanNet_views='AUTHOR_ARCHIVE;200_FRAME_PARITY_UNVERIFIED'))
    lines=['# 统一对比：Replica8 + ScanNet15','',
        '范围参考 OVI-MAP 论文：Replica8 全部；ScanNet/ConceptFusion18 中15个验证场景，排除训练集 scene0168_00/01/02。',
        '指标为同一实例预测投影到完整原始点序后的对象类指标（Replica48/ScanNet198），不是论文 Replica51/ScanNet200 的完整语义评测。重叠按冻结 AP 置信度及原列序决定；未覆盖的有效GT点计错。ScanNet15 的 AP 从该15场景重新整体池化。Ours 对两个种子各自整体评分后取均值。Open3DIS ScanNet 使用作者预测归档，200帧输入一致性尚未证明；现有GT接触与训练角色限制继续适用。','',
        '| 数据集 | 方法 | 场景数 | mIoU | mAcc | AP25 | AP50 | APall | oAcc |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in main+ablation:
        lines.append('| '+' | '.join([x['benchmark'],x['method'],str(x['scenes']),*['待补' if v is None else f'{v*100:.2f}' for v in x['metrics'].values()]])+' |')
    (out/(name+'.md')).write_text('\n'.join(lines)+'\n')
    for p,sha in before.items():assert digest_file(p)==sha,p
    print('CONTROLLED_COMPARISON_READY',result['status'],result['identity'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['freeze','score','pool'],required=True);args=p.parse_args()
    {'freeze':freeze_missing,'score':score_missing,'pool':pool}[args.phase](args.root,args.out)

if __name__=='__main__':main()
