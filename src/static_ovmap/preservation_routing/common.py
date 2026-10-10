from pathlib import Path
from static_ovmap.learned_object_readout.common import (
    ConsumptionIndex,PathResolver,arrays_record,canonical_digest,read,verified,write,_array_digest,
)
REPO=Path(__file__).resolve().parents[3]

def fixed_spec(path):
    original=REPO/'docs/paper/static_ovmap/preservation_routing_v1/PROTOCOL_SPEC.json'
    if Path(path).read_bytes()!=original.read_bytes():
        raise ValueError('Fixed scientific specification changed; paths use CLI overrides')
    return read(path)

def objects(binding,role):
    root=Path(binding['lr_parent_root']);split=verified(root/'split_manifest.json')
    values=[o for s in split['roles'][role] for o in verified(root/'data/generated'/s/'manifest.json')['objects']]
    return PathResolver(binding['path_map']).rewrite(values) if binding['path_map'] else values

def method(binding,identifier):
    return next(m for m in binding['specification']['R_methods']+binding['specification']['G_methods'] if m['id']==identifier)
