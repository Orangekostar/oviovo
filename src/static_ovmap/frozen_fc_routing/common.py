from pathlib import Path
from static_ovmap.learned_object_readout.common import (
    ConsumptionIndex,PathResolver,arrays_record,canonical_digest,read,verified,write,_array_digest,
)
REPO=Path(__file__).resolve().parents[3]
METRICS=('apall','ap50','ap25','miou','macc')

def fixed_spec(path):
    original=REPO/'docs/paper/static_ovmap/frozen_fc_routing_v1/PROTOCOL_SPEC.json'
    if Path(path).read_bytes()!=original.read_bytes():raise ValueError('Fixed specification changed; use CLI path overrides')
    return read(path)

def objects(binding,role):
    root=Path(binding['lr_parent_root']);split=verified(root/'split_manifest.json')
    role='holdout' if role=='old_H' else role
    values=[o for s in split['roles'][role] for o in verified(root/'data/generated'/s/'manifest.json')['objects']]
    return PathResolver(binding['path_map']).rewrite(values)

def immutable_write(path,value):
    path=Path(path)
    if path.exists():
        old=verified(path)
        if {k:v for k,v in old.items() if k!='identity'}!=value:raise ValueError('Locked artifact changed: '+str(path))
        return old
    return write(path,value)

FORWARD_KEYS=frozenset(('fc_vectors','view_valid','frame_ids','local_raw','local_unit','metadata','local_valid'))
