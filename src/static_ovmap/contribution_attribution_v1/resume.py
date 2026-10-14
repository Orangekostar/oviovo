"""Finite consumed-input checks; unchanged publication launches no workers."""
from pathlib import Path
from .common import ConsumptionIndex,verified,read,write,source_digest,canonical_digest,read_document,PathResolver

PRODUCERS={
    'preflight':('binding',), 'inventory':('common','binding','inventory'),
    'history':('common','history'),
    'fusion_dev_predict':('common','sources','payloads','prediction'),
    'main_predict':('common','sources','payloads','prediction'),
    'fusion_dev_score':('common','scoring'), 'main_score':('common','scoring'),
    'freeze':('common','selection'), 'surface':('common','binding','surface'),
    'attribute':('common','records','recovery','history','attribute','lineage'),
    'rank':('common','records','scoring','rank'),
    'stability':('common','records','stability'),
    'qualitative':('common','records','recovery','nomination','visualize'),
    'nominate':('common','records','nomination')}

def check_consumed(root):
    """Only task memo files, not a disk-wide scan or rehash of all parents."""
    root=Path(root);identities={};memos=[]
    if (root/'surface.json').exists():check_frozen_models(root)
    for path in sorted(root.rglob('*verifications.json')):
        if not path.is_file():continue
        memo=read(path)
        if 'entries' not in memo:continue
        index=ConsumptionIndex(path)
        for row in memo['entries']:
            expected=row['identity'];actual=index.identity(expected['path'],expected)
            previous=identities.get(actual['path'])
            if previous is not None and previous!=actual:raise ValueError('Conflicting consumed identities: '+actual['path'])
            identities[actual['path']]=actual
        memos.append(str(path))
    return dict(consumed_files=len(identities),memos=memos,input_digest=canonical_digest(identities))

def check_frozen_models(root):
    """Recheck the finite files consumed by FrozenFC, including its private index."""
    from .binding import load_binding
    root=Path(root);binding=load_binding(root);resolver=PathResolver(binding['path_map']);index=ConsumptionIndex(root/'resume/model_verifications.json')
    registry=resolver.rewrite(read_document(Path(binding['fg_parent_root'])/'export/weights_registry.json',index))
    selected={}
    for name in ('FG2_SURFACE','S29_FG2_SURFACE'):
        descriptor=registry['heads'][name]['weights'];selected[name]=index.identity(descriptor['path'],descriptor)
    text=index.identity(registry['text_identity']['path'],registry['text_identity'])
    scene=binding['cohorts']['scannet_cf18'][0];descriptor=binding['scenes'][scene]['context'];context=resolver.rewrite(read_document(descriptor['path'],index))
    reference=read_document(Path(context['FC_model_reference_root'])/'receipt.json',index,sealed=False)
    physical=[]
    for row in reference['inputs']:
        if '/assets/fc_frozen/' in row['path'] or Path(row['path']).name=='region_adapter.py':
            physical.append(index.identity(resolver.resolve(row['path']),row))
    if not physical or reference['physical_model_identity']!=registry['FC_identity']:raise ValueError('Frozen physical FC provenance differs')
    lock=write(root/'model_input_lock.json',dict(status='COMPLETE',heads=selected,text=text,FC_identity=registry['FC_identity'],FC_consumed_weights_and_operator=physical,original_surface_used_same_expected_descriptors=True))
    index.write_memo(root/'resume/model_verifications.json');return lock

def phase_record(root,phase,output):
    return write(Path(root)/'resume'/f'{phase}.json',dict(status='COMPLETE',phase=phase,
                 producer=source_digest(PRODUCERS[phase]),output_identity=verified(output)['identity']))

def phase_unchanged(root,phase,output):
    path=Path(root)/'resume'/f'{phase}.json'
    if not path.exists() or not Path(output).exists():return False
    value=verified(path)
    return value['producer']==source_digest(PRODUCERS[phase]) and value['output_identity']==verified(output)['identity']
