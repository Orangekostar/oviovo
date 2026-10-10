"""Target-free adapter and pre-outcome role locks; no split regeneration."""
import hashlib
import re
from pathlib import Path
from .common import PathResolver,ConsumptionIndex,canonical_digest,objects,read,verified,write
from static_ovmap.learned_object_readout.features import FrozenFC,ObjectLoader as ParentLoader,execution_config

class ObjectLoader(ParentLoader):
    def __init__(self,fc,feature_manifest,**kwargs):
        self.resolver=PathResolver(fc.binding['path_map'])
        super().__init__(fc,self.resolver.rewrite(feature_manifest),**kwargs)
    def load(self,obj,condition=0,requested_views=8):
        inputs,vectors,audit=super().load(self.resolver.rewrite(obj),condition,requested_views)
        inputs['fc_vectors']=vectors
        return inputs,vectors,audit
    def targets(self,obj,condition,requested_views):
        return super().targets(self.resolver.rewrite(obj),condition,requested_views)

def lock_H2_inventory(binding,plan):
    from static_ovmap.learned_object_readout.inventory import inventory_roots,RAW_SUFFIXES
    root=Path(binding['output_root']);path=root/'H2_pretraining_inventory.json'
    if path.exists():
        receipt=verified(path)
        if receipt['plan']!=plan['identity']:raise ValueError('H2 pretraining inventory plan differs')
        return receipt
    if list((root/'training').glob('seed*/*/updates.jsonl')):
        raise ValueError('H2 inventory must be recorded before scientific training')
    inventory=inventory_roots(binding['data_roots']+[str(Path(binding['lr_parent_root'])/'data/scannet')])
    index=ConsumptionIndex();selected={}
    for scene in plan['selected']:
        entry=inventory['scans'].get(scene)
        selected[scene]=dict(complete=bool(entry and entry['complete']),
             files={s:index.identity(p) if Path(p).is_file() and Path(p).stat().st_size else None
                    for s,p in entry['files'].items()} if entry else {},
             missing=entry['missing'] if entry else list(RAW_SUFFIXES),calibration=None,
             calibration_reason='NOT_PARSED_BEFORE_NOMINATION')
    return write(path,dict(status='LOCKED_BEFORE_SCIENTIFIC_TRAINING',plan=plan['identity'],selected=selected,
                 inspected_immediate_directories=inventory['inspected_immediate_directories'],annotations_parsed=0,
                 outcomes_used=False,scientific_updates=0))

def prepare(binding):
    root=Path(binding['output_root']);parent=Path(binding['lr_parent_root'])
    split=verified(root/'split_manifest.json');cls=verified(root/'class_split.json')
    counts={r:objects(binding,r) for r in ('train','dev')}
    if sum(o['base'] for o in counts['train'])!=622 or sum(o['base'] for o in counts['dev'])!=69:
        raise ValueError('Actual base-positive object count differs; cannot edit expected count')
    if len(cls['base_ids'])!=160 or len(cls['heldout_ids'])!=40:raise ValueError('Class split changed')
    observed=sorted({o['class_id'] for o in counts['train'] if o['base']})
    if len(observed)!=74:raise ValueError('Observed positive categories changed')
    queue=parent/'external/upstream_sources/ScanNet/Tasks/Benchmark/scannetv2_train.txt'
    denied=set(split['exclusions']);denied.update(s.split('_')[0] for role in split['roles'].values() for s in role)
    denied.update(s.split('_')[0] for s in binding['scenes'] if s.startswith('scene'))
    # Finite lineage inventories: names only, not annotation contents or outcomes.
    exposure_sources=[]
    for p in [parent/'data_inventory.json',Path(binding['prior_data_root'])/'acquisition_lock.json']:
        if p.exists():
            value=read(p);denied.update(re.findall(r'scene\d{4}(?=_\d{2})',str(value)))
            exposure_sources.append(str(p))
    denied.update(re.findall(r'scene\d{4}(?=_\d{2})',str(binding['reference'])))
    families={}
    for scan in queue.read_text().splitlines():
        scan=scan.strip();family=scan.split('_')[0]
        if scan and family not in denied:families.setdefault(family,[]).append(scan)
    ranked=sorted(families,key=lambda f:(hashlib.sha256(('PR1-H2-20261010|'+f).encode()).hexdigest(),f))
    selected=[min(families[f],key=lambda s:int(s.rsplit('_',1)[1])) for f in ranked[:4]]
    if len(selected)!=4:raise ValueError('Official H2 name queue incomplete')
    frames={}
    for scene in split['roles']['dev']:
        manifest=verified(parent/'data/generated'/scene/'manifest.json')
        bank=sorted(manifest['frames'],key=lambda f:f['frame_id'])
        frames[scene]=PathResolver(binding['path_map']).rewrite([bank[i] for i in sorted({0,(len(bank)-1)//2})])
    path=root/'H2_plan.json'
    plan=dict(status='NAMES_LOCKED_BEFORE_SCIENTIFIC_TRAINING',selected=selected,exclusions=sorted(denied),
              official_train=ConsumptionIndex().identity(queue),exposure_sources=exposure_sources,
              annotations_parsed=0,selection_depends_on_semantics=False,replacement_after_outcomes=False,
              real_proposal_frames=frames)
    if path.exists():
        old=verified(path)
        if {k:v for k,v in old.items() if k!='identity'}!=plan:raise ValueError('Prelocked H2/proposal names changed')
    else:write(path,plan)
    lock_H2_inventory(binding,verified(path))
    return write(root/'prepare.json',dict(status='COMPLETE',split=split['identity'],classes=cls['identity'],
                 original_objects={r:len(v) for r,v in counts.items()},base_original_objects={r:sum(o['base'] for o in v) for r,v in counts.items()},
                 observed_positive_base_ids=observed,zero_positive_base_ids=[i for i in cls['base_ids'] if i not in observed],
                 H2_plan=verified(path)['identity'],new_feature_encodings=0,parent_data_read_only=True))
