from static_ovmap.semantic_update_evidence_v1.metrics import harmonic
import numpy as np
import pytest
from static_ovmap.semantic_update_evidence_v1.metrics import recognition,predictions
from static_ovmap.semantic_update_evidence_v1.readers import column_index,public_GT_scores
from static_ovmap.semantic_update_evidence_v1.diagnostics import rank_record
from static_ovmap.semantic_update_evidence_v1.common import gzip_json
from static_ovmap.structure_confirmation_v1.prediction_codec import encode,decode

def test_harmonic_empty_and_zero_are_distinct():
    assert harmonic(None, .5) is None
    assert harmonic(0., 0.) == 0.
    assert harmonic(.25, .75) == .375

def test_macro_micro_unknown_and_stable_row_order(tmp_path):
    y=np.array([7,7,7,8,9]);p=np.array([7,7,0,0,9])
    m=recognition(y,p,[7,8],[9]);assert m['B_macro']==1/3
    assert m['B_micro']==.5 and m['originals']==5 and m['abstentions']==2
    assert m['H_macro']!=m['H_micro']
    a=np.array([[.5,.5],[np.nan,np.nan],[.1,.9]],np.float32)
    assert predictions(a,[9,7],[True,False,False]).tolist()==[9,0,0]
    f=tmp_path/'a.gz';encode(f,a);assert decode(f).tobytes()==a.tobytes()

def test_column_join_is_metadata_based():
    columns=[dict(method='S_LL',seed=17,profile='matched1000',step=1000,prefix=8),
             dict(method='B_FC',seed=None,profile='selected',step=None,prefix=8)]
    assert column_index(columns,'B_FC',29)==1
    assert column_index(list(reversed(columns)),'S_LL',17)==1
    with pytest.raises(RuntimeError):column_index(columns,'S_LL',29)

def test_exact_public_fallback_joins_record_metadata_and_keeps_defined(tmp_path):
    a=np.array([[[.2,.9]]],np.float32);path=tmp_path/'scores/GT/one.scores.gz';encode(path,a)
    row=dict(kind='GT',dtype='float32',lossless=True,record_metadata={'key':'object_a'},
             metadata={'defined':[[False]]},valid_ids=[7,9],path='scores/GT/one.scores.gz')
    gzip_json(tmp_path/'scores/manifest.json.gz',dict(records=[row]))
    s,d,ids=public_GT_scores(tmp_path,'object_a')
    assert s.tobytes()==a.tobytes() and not d[0,0]
    assert predictions(s,ids,d).item()==0
    assert rank_record(s[0,0],9,ids,[9],[7],False)['full_rank'] is None
