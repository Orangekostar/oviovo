#!/usr/bin/env python3
"""Package-only checks. No repository, neural models, scans, or network access.

These four groups verify the *specification*, not the implemented research.
The production Codex run must independently perform its four scoped test groups.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional, Sequence


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def mix(probabilities: Sequence[Optional[Sequence[float]]], weights: Sequence[float]) -> Optional[list[float]]:
    if len(probabilities) != len(weights):
        raise ValueError('Mismatched input counts')
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('Weights must be finite and nonnegative')
    pairs = [(list(p), w) for p, w in zip(probabilities, weights) if p is not None and w > 0]
    if not pairs:
        return None  # caller implements UNKNOWN or the explicitly named native fallback
    width = len(pairs[0][0])
    for values, _ in pairs:
        if len(values) != width or not width or any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError('Malformed probabilities')
        if not math.isclose(sum(values), 1.0, abs_tol=1e-12):
            raise ValueError('Probability sum differs from one')
    total = sum(w for _, w in pairs)
    return [sum(values[i] * w for values, w in pairs) / total for i in range(width)]


def grouped(p: Sequence[Optional[Sequence[float]]]) -> Optional[list[float]]:
    if len(p) != 3:
        raise ValueError('N/Q/F required')
    return mix([mix(p[:2], [1., 1.]), p[2]], [1., 1.])


def same(a: Sequence[float], b: Sequence[float], tolerance: float = 1e-12) -> bool:
    return len(a) == len(b) and all(abs(x-y) <= tolerance for x, y in zip(a, b))


def factorial(p00: float, p10: float, p01: float, p11: float) -> dict[str, float]:
    e = p10-p00; r = p01-p00; interaction = p11-p10-p01+p00
    return dict(E_add=e, R_add=r, interaction=interaction, total=p11-p00,
                E_average=.5*((p10-p00)+(p11-p01)),
                R_average=.5*((p01-p00)+(p11-p10)))


def run(root: Path) -> dict:
    spec = json.loads((root/'02_冻结执行配置.json').read_text(encoding='utf-8'))
    source = json.loads((root/'source_evidence_manifest.json').read_text(encoding='utf-8'))
    groups = []
    p = [[.9,.1],[.3,.7],[.2,.8]]
    require(same(grouped(p), mix(p,[.25,.25,.5])), 'Full-source D2 equivalence')
    absent = [p[0], None, p[2]]
    require(not same(grouped(absent), mix(absent,[.25,.25,.5])), 'Missing-source behavior must not be conflated')
    require(grouped([None,None,None]) is None, 'No fabricated probabilities when all sources missing')
    require(mix([None,p[1],None],[1.,0.,0.]) is None, 'Zero weights cannot activate unavailable source')
    for a in (None,p[0]):
        for b in (None,p[1]):
            for c in (None,p[2]):
                m=grouped([a,b,c])
                require(m is None or math.isclose(sum(m),1.,abs_tol=1e-12), 'Availability normalization')
    groups.append(dict(group='T1_SPEC_PROBABILITY', status='PASSED',checks=['full-source equivalence','missing-source difference','empty-source semantics','eight availability patterns']))

    for cells in [(0.1,0.14,0.12,0.15),(.18,.16,.19,.18),(0.,.1,.2,.3)]:
        f=factorial(*cells)
        require(math.isclose(f['E_add']+f['R_add']+f['interaction'],f['total'],abs_tol=1e-12),'Factorial identity')
        require(math.isclose(f['E_average']+f['R_average'],f['total'],abs_tol=1e-12),'Average marginal identity')
    require(len(spec['new_main_methods'])==12,'Twelve main variants including three standalone F variants')
    require(len({x['id'] for x in spec['new_main_methods']})==12,'Unique main variant IDs')
    require(len(spec['cohorts']['replica8'])==8 and len(spec['cohorts']['scannet_cf18'])==18,'Cohort counts')
    require(len({x.rsplit('_',1)[0] for x in spec['cohorts']['scannet_cf18']})==7,'CF physical-family count')
    require(12*26==spec['limits']['max_main_scene_rows'],'Main count cap')
    require(2*2*26==spec['limits']['max_rank_new_offdiagonal_rows'],'Rank count cap')
    require(69*4*2*(1+5) < spec['limits']['max_frozen_head_forwards'],'Frozen head bound')
    groups.append(dict(group='T2_SPEC_ATTRIBUTION_AND_BUDGET',status='PASSED',checks=['factorial closure','12x26=312','rank 104','head bound','CF family count']))

    grid=spec['weights']['grid']
    require(len(grid)==15 and len({tuple(w) for w in grid})==15,'Quarter-simplex has 15 candidates')
    require(all(math.isclose(sum(w),1.,abs_tol=1e-12) and all(0<=x<=1 and x*4==int(x*4) for x in w) for w in grid),'Grid contract')
    require([.25,.25,.5] in grid,'D2 full-source weights retained')
    reg=set(spec['cohorts']['replica8']+spec['cohorts']['scannet_cf18'])
    require(not reg.intersection(spec['fusion_dev']),'Fusion developer role disjointness')
    require(len(spec['fusion_dev'])==4,'Only exact four BB development scenes')
    require(spec['confirmation']['execute'] is False,'No new confirmation execution')
    for k in ('new_optimizer_updates','new_rgb_encodings','new_sam_proposals','new_map_construction'):
        require(spec['limits'][k]==0,'Unauthorized new resource: '+k)
    require(spec['phases'][-1]=='all' and len(spec['phases'])==len(set(spec['phases'])),'Phase contract')
    groups.append(dict(group='T3_SPEC_SELECTION_AND_SCOPE',status='PASSED',checks=['15 fixed weights','development role isolation','no training/new encoding','confirmation plan only']))

    main=(root/'00_给Codex的最终执行指令.md').read_text(encoding='utf-8')
    expected={f'S{i:02d}' for i in range(1,36)}
    require({s['id'] for s in source['sources']}==expected,'Complete 35-item evidence bindings')
    referenced=set(re.findall(r'\bS\d\d\b',main))
    require(referenced <= expected,'Unknown source references')
    for name in ['README.md','01_源码证据与任务绑定.md','02_冻结执行配置.json','03_交付前审核报告.md','04_合同自检.py']:
        require((root/name).is_file(),'Missing package file: '+name)
    for s in source['sources']:
        require(re.fullmatch('[0-9a-f]{40}',s['ref']) is not None,'Unpinned source')
        require(s.get('git_blob_sha') is None or re.fullmatch('[0-9a-f]{40}',s['git_blob_sha']) is not None,'Malformed blob SHA')
    for token in ['score_explicit_view','NOT_SELECTED_NO_FUSION_DEV','DIAGNOSTIC_ONE_TO_ONE_GEOMETRY','shortlist_prelock.json','PUBLICATION_RECEIPT.json','submission_evidence_complete','CT_A5_FC_ONLY']:
        require(token in main,'Unspecified boundary: '+token)
    require('0.25,0.25,0.5' in main,'Missing D2 equivalence boundary')
    groups.append(dict(group='T4_PACKAGE_CROSS_REFERENCES',status='PASSED',checks=['files exist','sources pinned','required ambiguity resolutions','source reference integrity']))
    return dict(status='PASSED',test_groups=groups,number_of_groups=4,
                scope='PACKAGE_CONTRACT_ONLY_NOT_PROJECT_IMPLEMENTATION',
                actual_repository_integration=False,actual_FC_inference=False,
                actual_metric_reproduction=False,actual_GitHub_push=False,
                new_training=0,new_rgb_encodings=0,new_sam_proposals=0)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    try:
        result=run(args.package_root.resolve())
    except (AssertionError,ValueError,OSError,KeyError) as exc:
        result=dict(status='FAILED',error=f'{type(exc).__name__}: {exc}',scope='PACKAGE_CONTRACT_ONLY')
    body=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(body,encoding='utf-8')
    print(body,end='')
    return 0 if result['status']=='PASSED' else 1

if __name__=='__main__':
    raise SystemExit(main())
