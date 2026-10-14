"""Verify every named G1 lineage node rather than infer intermediate aliases."""
from pathlib import Path
from .common import read_document,write,canonical_digest

def verify(root,records):
    binding=records.binding;index=records.index;resolver=records.resolver
    minimal=resolver.rewrite(read_document(Path(binding['minimal_root'])/'source_binding.json',index))
    evidence_root=Path(minimal['parent_root']);rows=[]
    for scenes in binding['cohorts'].values():
        for scene in scenes:
            evidence=resolver.rewrite(read_document(evidence_root/'predictions'/scene/'receipt.json',index))
            repair=resolver.rewrite(read_document(Path(binding['minimal_root'])/'predictions'/scene/'receipt.json',index))
            paths=dict(A3=records.manifest(scene,'A3'),EV01_G1_V2=evidence['outcomes']['EV01_G1_V2']['manifest'],
                       IR01_G1=repair['methods']['IR01_G1']['manifest'],FG_REF_G1=records.manifest(scene,'REF_G1'))
            domains={}
            for node,path in paths.items():
                manifest=read_document(path,index,sealed=False);index.identity(Path(path).parent/manifest['arrays']['path'],manifest['arrays'])
                domains[node]={k:manifest[k] for k in ('geometry','owner_ids','semantic_labels','instance_ranks','prediction_key')}
            a0=read_document(records.manifest(scene,'A0'),index,sealed=False);recipe=a0['metadata']['method_recipe']
            if recipe['existing']!='NATIVE' or recipe['recovery']!='NONE':raise ValueError('A0 original recipe differs')
            same=all(value==domains['A3'] for value in domains.values())
            rows.append(dict(scene=scene,paths=paths,node_domain_digests={k:canonical_digest(v) for k,v in domains.items()},
                             exact_geometry_owner_semantics_rank_identity=same,A0_recipe=recipe,
                             status='SAME_DOMAIN' if same else 'DIFFERENT_DOMAIN_NO_GAIN_WATERFALL'))
    return write(Path(root)/'lineage.json',dict(status='COMPLETE',rows=rows,checked_nodes=['A3','EV01_G1_V2','IR01_G1','FG_REF_G1'],
                 all_exact_domains=all(r['exact_geometry_owner_semantics_rank_identity'] for r in rows),new_model_forwards=0,new_scene_assign_calls=0))
