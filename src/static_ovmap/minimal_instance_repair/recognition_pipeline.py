"""FC-environment acquisition only; released evaluation stays in its own env."""

import argparse
from pathlib import Path
import sys

import torch

from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.cvpr_compact.region_worker import FCSession
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read
from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache

from .binding import load_binding, seal
from .recognition_worker import acquire_frame, assemble_decisions, load_anyup


def run(root, phase):
    binding = load_binding(root)
    assets = read(Path(root)/'assets.json')
    scenes = binding['specification']['pilots'] if phase=='pilot' else [s for names in binding['cohorts'].values() for s in names]
    torch.set_num_threads(4)
    cfg = assets['execution_config']
    index = ConsumptionIndex(Path(root)/'recognition/model_verifications.json')
    session,model,decisions = None,None,[]
    with exclusive_lock(Path(cfg['gpu_lock'])),torch.inference_mode():
        for scene in scenes:
            data = read(binding['scenes'][scene]['context']['path'])
            text_session = FCSession(cfg,data,index,cache=None)
            if session is None:
                session = text_session
                session.load_model()
                model = load_anyup(assets)
                load_path = Path(root)/'recognition/model_load.json'
                if not load_path.exists():
                    atomic_write_json(load_path,seal({'FC_model':session.model_key,
                        'FC_model_load_seconds':session.model_load_seconds,'actual_worker_python':sys.executable,
                        'actual_torch':torch.__version__,'FC_weight_audit':session.weight_audit,
                        'models_required':['FC_FROZEN','ORIGINAL_ANYUP']}))
            elif session.model_key!=text_session.model_key:
                raise ValueError('same worker cannot silently change the physical FC model')
            session.text,session.ids,session.text_identity = text_session.text,text_session.ids,text_session.text_identity
            cache = ContentCache(assets['read_only_dense_cache_roots'],Path(root)/'content_cache/fc',session.model_key,
                                 index=index,resolver=PathResolver(binding['path_map']))
            plan = read(Path(root)/'recognition'/scene/'plan.json')
            from .reconcile import recognition_frame
            for fid in sorted(map(int,plan['frames'])):
                recognition_frame(binding,plan,fid,session,assets)
            records = [acquire_frame(binding,plan,fid,session,model,cache,assets,
                validate=phase=='pilot' and not (Path(root)/'pilots/ordinary_anyup_parity.json').exists())
                for fid in sorted(map(int,plan['frames']))]
            decisions.append(assemble_decisions(binding,plan,records))
        index.write_memo(Path(root)/'recognition/model_verifications.json')
    result = seal({'status':'RECOGNITION_COMPLETE','scenes':{r['scene']:r['identity'] for r in decisions},
        'scene_count':len(decisions),'targets_opened_by_GPU_worker':False,'released_evaluation_invoked':False})
    atomic_write_json(Path(root)/'recognition'/(phase+'_acquisition.json'),result)
    if phase=='recognize':
        atomic_write_json(Path(root)/'recognition/summary.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--phase',choices=('pilot','recognize'),required=True)
    args = parser.parse_args()
    run(args.root,args.phase)
