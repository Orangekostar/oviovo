import numpy as np


def test_historical_registry_keeps_original_scores_and_reports_nonexported_semantics(tmp_path):
    from src.static_ovmap.attribution_bridge import load_historical_registry
    owners=np.array([2,2,2,5,5,9]);semantic=np.array([4,4,4,4,4,7])
    for owner in [2,5]:np.save(tmp_path/f'inst-{owner}_label-4.npy',(owners==owner)[:,None])
    manifest=tmp_path/'pred.txt';manifest.write_text('inst-2_label-4.npy 4 1.000000\ninst-5_label-4.npy 4 0.666667\n')
    result=load_historical_registry(manifest,owners,semantic)
    assert result['owner_ids'].tolist()==[2,5]
    assert result['scores_serialized']==['1.000000','0.666667']
    assert result['scores_full_precision'][1]==2/3
    assert result['positive_semantics_outside_emitted_registry']==1
    assert result['registry_semantic'].tolist()==[4,4,4,4,4,0]
