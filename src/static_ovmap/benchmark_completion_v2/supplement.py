"""Read-only historical evidence and actual GT-free source availability counts."""
from pathlib import Path
import itertools
import subprocess
import numpy as np

from .common import REPO, BASE, read, verified, write, digest_file


def historical(binding):
    root=Path(binding['output_root'])
    path=Path(binding['structure_public'])/'records/review/primary_point_results.json.gz'
    if not path.is_file():
        return write(root/'supplement/historical.json',dict(status='BLOCKED_DEPENDENCY',missing=[str(path)],rows=[]))
    relative=str(path.relative_to(REPO))
    blob=subprocess.check_output(['git','show',BASE+':'+relative],cwd=REPO)
    if blob != path.read_bytes():
        raise ValueError('Historical structural result differs from its frozen published blob')
    report=read(path)
    rows=[]
    for source in report['rows']:
        matched=source['GT']['matched1000']
        gt={group:matched['mean'][group]['balanced_top1'] for group in ('base','novel','all')}
        per_seed=matched['per_seed']
        for group in gt:
            if not np.isclose(gt[group], np.mean([s[group]['balanced_top1'] for s in per_seed]),rtol=0,atol=1e-12):
                raise ValueError('Historical per-seed recognition aggregation differs')
        real=source.get('REAL_selected')
        metrics=dict(GT_Base=gt['base'],GT_Novel=gt['novel'],GT_All=gt['all'],
                     Real_AP=None if real is None else real['mean']['apall'],
                     Real_AP50=None if real is None else real['mean']['ap50'])
        if real:
            for m,k in (('Real_AP','apall'),('Real_AP50','ap50')):
                if not np.isclose(metrics[m], np.mean([s['metrics'][k] for s in real['per_seed']]),rtol=0,atol=1e-12):
                    raise ValueError('Historical per-seed pooled AP aggregation differs')
        rows.append(dict(method=source['method'],metrics=metrics,
            historical_GT_profile='matched1000',historical_real_profile='selected',
            per_seed_GT=per_seed,per_seed_REAL=None if real is None else real['per_seed'],
            source_key='rows:'+source['method']))
    return write(root/'supplement/historical.json',dict(status='DONE',rows=rows,
        source=dict(commit=BASE,path=relative,sha256=digest_file(path),identity=report['identity']),
        historical_only=True,new_inference=0,original_objects=report['originals'],
        historical_scope=report['scope'],GT_source=report['GT'],REAL_source=report['REAL'],
        not_current_benchmark=True,not_recomputed_as_new_evidence=True))


def availability(binding):
    root=Path(binding['output_root'])
    protocol=verified(root/'protocol_lock.json')
    reports={}
    for benchmark,scope in protocol['scopes'].items():
        counts={''.join(name for name,yes in zip(('N','Q','F'),bits) if yes) or 'NONE':0
                for bits in itertools.product((False,True),repeat=3)}
        receipts={}
        for scene in scope['scenes']:
            prediction=root/'benchmark/predictions'/benchmark/scene/'receipt.json'
            if not prediction.is_file():
                continue
            lock=verified(prediction)
            f=verified(root/'benchmark/F'/benchmark/scene/'receipt.json')
            nq=verified(root/'benchmark/NQ'/benchmark/scene/'sources.json')
            with np.load(f['arrays']['path'],allow_pickle=False) as a:
                owners=a['owners'].astype(int)
                flags=a['available'].astype(bool)
            for owner,flag in zip(owners,flags,strict=True):
                key=('N' if nq['sources']['N'][str(owner)]['available'] else '')
                key+=('Q' if nq['sources']['Q'][str(owner)]['available'] else '')
                key+=('F' if flag else '')
                counts[key or 'NONE']+=1
            receipts[scene]=dict(F=f['identity'],NQ=nq['identity'],prediction=lock['identity'],owners=len(owners))
        reports[benchmark]=dict(counts=counts,completed_scenes=len(receipts),expected_scenes=len(scope['scenes']),
            scene_receipts=receipts,full_scene_coverage=len(receipts)==len(scope['scenes']),
            owners=sum(counts.values()),GT_labels_read=False)
    return write(root/'supplement/availability.json',dict(status='DONE',benchmarks=reports,
        scope='ACTUAL_COMPLETED_SCENES_ONLY',counts_do_not_duplicate_seeds_or_variants=True,
        definition='Available means the original frozen source returned finite supported class scores; no GT condition.'))


def job_manifest(binding):
    root=Path(binding['output_root'])
    jobs={}
    for path in sorted((root/'jobs').glob('**/receipt.json')):
        row=verified(path)
        jobs[str(path.parent.relative_to(root/'jobs'))]=dict(
            status=row['status'],receipt=row['identity'],argv=row['argv'],cwd=row['cwd'],
            env=row['env'],actual_child_exit=row['actual_child_exit'],attempts=row['attempts'],
            inputs=row['inputs'],outputs=row['outputs'])
    bootstrap={}
    for name in ('calibration_exit.json','model_lock_exit.json','legacy_predict_exit.json',
                 'legacy_score_exit.json','first_standard_exit.json','streaming_standard_exit.json'):
        path=root/'bootstrap'/name
        if path.is_file():
            bootstrap[name]=read(path)
    return write(root/'job_manifest.json',dict(status='DONE',jobs=jobs,early_actual_executions=bootstrap,
        in_progress_process_markers_are_not_completed_jobs=True,new_training_updates=0))
