"""Task-owned identities, finite budgets and immutable execution records."""
from pathlib import Path
import shutil
from static_ovmap.contribution_attribution_v1.common import (
    write,verified,read_document,digest_file,plain,immutable_record,
    ConsumptionIndex,PathResolver,canonical_digest,DependencyError,
)
from static_ovmap.frozen_fc_routing.common import FORWARD_KEYS,objects,arrays_record

REPO=Path(__file__).resolve().parents[3]
TASK='fg2-structure-confirmation-v1'
BASE='2f0d8a4f7043557462f05875eca8b7f71eb66761'
CONDITIONS=('clean','truncate','append','truncate_append')
METHODS=('S_LL','V_LL','G_LL','S_UL','S_LU','S_UU','S_PRE','MV_ATTN')
STATICS=('B_FC','B_SURFACE_MEAN','B_PFE_ADAPTED')
CHECKPOINTS=(250,500,1000,1500,2000)
PHASES=('preflight','bind','prepare_cache','train','dev_select','freeze','confirm_acquire','confirm_geometry','confirm_proposals','confirm_features','confirm_real_predict','confirm_gt_prepare','confirm_gt_predict','confirm_score','map_predict','map_score','statistics','cost','qualitative')

def family(scene):return scene.split('_')[0] if scene.startswith('scene') else scene

def producer(names):
    folder=Path(__file__).parent
    return canonical_digest({name:digest_file(folder/(name+'.py')) for name in names})

def unit_key(phase,inputs,*,producers,**parameters):
    return canonical_digest(dict(task=TASK,schema=1,phase=phase,producer=producer(producers),inputs=inputs,parameters=parameters))

def fixed_spec(path):
    source=REPO/'docs/paper/static_ovmap/structure_confirmation_v1/prompt/02_冻结执行配置.json'
    if Path(path).read_bytes()!=source.read_bytes():raise ValueError('Frozen task specification differs')
    return __import__('json').loads(source.read_text())

def existing(path,identity):
    path=Path(path)
    if not path.exists():return None
    value=verified(path)
    if value.get('input_identity')!=identity:raise ValueError('Unit identity changed; preserve old result and affected descendants: '+str(path))
    return value if value['status']=='COMPLETE' else None

def reserve(root,*,initial=False):
    free=shutil.disk_usage(root).free;required=(80 if initial else 20)*2**30
    if free<required:raise OSError('Runtime free-space budget: '+str(free)+' < '+str(required))
    return free

class Budget:
    """One GPU writer, with explicit invocation counters and optional batching."""
    def __init__(self,path,limits):
        self.path=Path(path);self.limits={k:int(v) for k,v in limits.items()};self.counts=dict.fromkeys(self.limits,0)
        if self.path.exists():
            old=verified(self.path)
            if old['limits']!=self.limits:raise ValueError('Runtime budget definition changed')
            self.counts.update(old['counts'])
    def consume(self,key,n=1,*,persist=True):
        if key not in self.limits or int(n)!=n or n<0:raise ValueError('Undefined or negative budget consumption')
        value=self.counts[key]+int(n)
        if value>self.limits[key]:raise ValueError(f'{key} budget exceeded: {value}>{self.limits[key]}')
        self.counts[key]=value
        if persist:self.flush()
    def flush(self):return write(self.path,dict(status='COMPLETE',limits=self.limits,counts=self.counts))

def workload_budget(root,spec):
    l=spec['limits']
    return Budget(Path(root)/'workload.json',dict(main_head=l['max_main_training_validation_head_calls'],confirmation_head=l['max_confirmation_head_calls'],map_head=l['max_old_map_head_calls'],engineering_head=l['max_engineering_head_calls'],cost_head=l['max_cost_head_calls'],optimizer_updates=spec['train']['max_new_updates']+l['max_replayed_optimizer_updates'],replayed_updates=l['max_replayed_optimizer_updates'],confirmation_FC=l['max_new_confirmation_FC_encodings'],SAM_images=l['max_new_SAM_image_encodings']))
