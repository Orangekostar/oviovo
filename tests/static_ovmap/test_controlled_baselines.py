"""The paper-sized subset and frame adapter must never pick favorable results."""
import numpy as np
import pytest


def test_scope_is_predeclared_paper_intersection_not_metric_selection():
    from static_ovmap.controlled_baselines import controlled_scenes
    paper={'replica8':['office0','room0'],'scannet_cf18':['scene0011_00','scene0168_00','scene0050_00']}
    validation=['scene0050_00','scene0011_00','other']
    scopes,excluded=controlled_scenes(paper,validation)
    assert scopes=={'replica48':['office0','room0'],'scannet200_val':['scene0011_00','scene0050_00']}
    assert excluded==['scene0168_00']
    with pytest.raises(ValueError):controlled_scenes(dict(paper,scannet_cf18=['scene0011_00']*2),validation)


def test_frame_reindex_keeps_exact_200_source_frames_and_original_time_order():
    from static_ovmap.controlled_baselines import frame_mapping
    frames=np.linspace(0,2373,200,dtype=int).tolist();mapping=frame_mapping(frames)
    assert [x['source'] for x in mapping]==frames
    assert [x['staged'] for x in mapping]==list(range(0,2000,10))
    for invalid in [frames[:-1],frames[::-1],[0]*200]:
        with pytest.raises(ValueError):frame_mapping(invalid)


def test_readonly_stage_link_refuses_changed_target(tmp_path):
    from static_ovmap.controlled_baselines import link
    a=tmp_path/'source';a.write_bytes(b'original');b=tmp_path/'different';b.write_bytes(b'other')
    target=tmp_path/'staged/rgb.jpg';link(target,a);link(target,a)
    assert target.read_bytes()==b'original'
    with pytest.raises(ValueError):link(target,b)


def test_object_vocabulary_retains_id_order_and_excludes_stuff():
    from static_ovmap.controlled_baselines import object_vocabulary
    ids,names=object_vocabulary(dict(text_ids=[1,2,3,5],text_names=['wall','chair','floor','table'],instance_ids=[2,5]))
    assert ids==[2,5] and names==['chair','table']
    with pytest.raises(ValueError):object_vocabulary(dict(text_ids=[1],text_names=['wall'],instance_ids=[2]))


def test_checkpoint_allows_only_the_unused_training_loss_buffer():
    import torch
    from static_ovmap.controlled_baselines import load_frozen_checkpoint
    model=torch.nn.Module();model.model=torch.nn.Linear(2,2);model.criterion=torch.nn.Module()
    model.criterion.register_buffer('empty_weight',torch.ones(3))
    state={k:torch.full_like(v,.25) for k,v in model.state_dict().items() if k!='criterion.empty_weight'}
    assert load_frozen_checkpoint(model,state)==['criterion.empty_weight']
    assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
    assert torch.equal(model.criterion.empty_weight,torch.ones(3))
    with pytest.raises(AssertionError):load_frozen_checkpoint(model,{k:v for k,v in state.items() if k!='model.bias'})
