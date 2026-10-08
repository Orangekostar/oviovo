"""Lossless compact decisions must preserve exact JSON types and sealed identities."""

import json

import numpy as np
import pytest

from static_ovmap.module_validation.contracts import canonical_digest
from static_ovmap.source_preserving_update.binding import seal


def test_compact_vectors_lossless_deduplication_and_corruption_detection():
    from static_ovmap.source_preserving_update.reporting import pack_vectors,unpack_vectors
    floats=[-0.0,1e-200,0.2,None,1.0,2.0,3.0,4.0,5.0]
    mixed=[1,2.0,None,4,5,6,7,8,9]
    large_integers=[2**60+i for i in range(9)]
    original=seal({'method_a':{'p':floats,'mixed':mixed},'method_b':{'p':floats},
                   'large_integers':large_integers,'bools':[True]*9})
    vectors={};packed=pack_vectors(original,vectors)
    assert len(vectors)==2
    assert packed['method_a']['p']==packed['method_b']['p']
    assert packed['large_integers']==large_integers
    restored=unpack_vectors(packed,vectors)
    assert json.dumps(restored,sort_keys=True)==json.dumps(original,sort_keys=True)
    assert canonical_digest({k:v for k,v in restored.items() if k!='identity'})==restored['identity']
    assert all(a.dtype==np.float64 for a in vectors.values())
    broken={k:v.copy() for k,v in vectors.items()}
    broken[packed['method_a']['p']['array_reference']][2]+=0.01
    with pytest.raises(ValueError,match='vector content'):
        unpack_vectors(packed,broken)
