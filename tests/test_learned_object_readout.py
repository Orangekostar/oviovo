"""Bounded production checks for the fixed supervised experiment."""
import ast
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)


def production(name):
    try:
        return importlib.import_module('static_ovmap.learned_object_readout.' + name)
    except ModuleNotFoundError as exc:
        pytest.fail(f'Required production module absent: {exc.name}')


def test_family_class_separation_and_lowest_complete_suffix():
    split = production('inventory')
    scans = [f'scene{i:04d}_{j:02d}' for i in range(60) for j in (2, 0, 1)]
    roles = split.family_split(scans, {'scene0011', 'scene0050'})
    assert [len(roles[k]) for k in ('train', 'dev', 'holdout')] == [24, 4, 4]
    families = [s.split('_')[0] for values in roles.values() for s in values]
    assert len(set(families)) == 32
    assert not {'scene0011', 'scene0050'} & set(families)
    assert all(s.endswith('_00') for values in roles.values() for s in values)
    base, novel = split.class_split(range(1, 201))
    assert len(base) == 160 and len(novel) == 40
    assert not set(base) & set(novel)
    assert set(base) | set(novel) == set(range(1, 201))
    with pytest.raises(ValueError, match='BLOCKED_TRAINING_DATA'):
        split.family_split(scans[:12], set())


def test_inventory_is_bounded_and_does_not_parse_annotations(tmp_path):
    inventory = production('inventory')
    root = tmp_path / 'declared'
    scene = root / 'scans' / 'scene0999_00'
    scene.mkdir(parents=True)
    suffixes = ('.sens', '.txt', '_vh_clean_2.ply', '_vh_clean_2.0.010000.segs.json', '.aggregation.json')
    for suffix in suffixes:
        (scene / ('scene0999_00' + suffix)).write_bytes(b'not parsed during inventory')
    hidden = root / 'unrelated' / 'deep' / 'scans' / 'scene0998_00'
    hidden.mkdir(parents=True)
    for suffix in suffixes:
        (hidden / ('scene0998_00' + suffix)).write_bytes(b'not authorized by immediate-root discovery')
    result = inventory.inventory_roots([root])
    assert list(result['scans']) == ['scene0999_00']
    assert result['scans']['scene0999_00']['complete']
    assert result['annotation_files_parsed'] == 0


def numerical_source(root):
    """Load only pinned numerical AST nodes, without registry/framework imports."""
    from einops import rearrange, repeat
    from timm.layers import DropPath
    namespace = dict(torch=torch, nn=nn, F=F, rearrange=rearrange, repeat=repeat,
                     cp=torch.utils.checkpoint, DropPath=DropPath)
    for path, names in [(root / 'fcclip/modeling/meta_arch/convnext.py', {'ConvNextBlock', 'LayerNorm'}),
                        (root / 'mask_adapter/modeling/meta_arch/mask_adapter_head.py', {'MASKAdapterHead', 'LayerNorm2d'})]:
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
        for node in nodes:
            node.decorator_list = []
            node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef) or n.name != 'from_config']
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    method.decorator_list = []
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['MASKAdapterHead']


def test_mask_adapter_output_and_gradient_port_parity():
    port = production('mask_adapter')
    root = Path('/mnt/shared/ww/ovimap-learned-object-readout-v1/attempt_001/external/upstream_sources/MaskAdapter')
    original = numerical_source(root)
    torch.manual_seed(31)
    reference = original('convnext_large_d_320', 16, 16, False, 4)
    model = port.MASKAdapterHead('convnext_large_d_320', 16, 16, False, 4)
    model.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(1, 768, 4, 5, dtype=torch.float32, requires_grad=True)
    xp = x.detach().clone().requires_grad_(True)
    masks = torch.rand(1, 2, 16, 20, dtype=torch.float32)
    expected, actual = reference(x, masks), model(xp, masks)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(xp.grad, x.grad, atol=0, rtol=0)
    for (name, p), (other, q) in zip(reference.named_parameters(), model.named_parameters()):
        assert name == other
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_depth_color_registration_scale_bounds_and_world_frame():
    sensor = production('sensor')
    rgb = np.zeros((1, 3, 3), np.uint8)
    rgb[0, :, 0] = [10, 30, 50]
    e = np.eye(4); e[0, 3] = 2
    pose = np.eye(4); pose[2, 3] = 3
    result = sensor.register_depth_rgb(rgb, np.array([[2000]], np.uint16), np.eye(3), np.eye(3), e, pose, 1000)
    assert result['depth_m'][0, 0] == 2
    assert result['color_valid'][0, 0]
    assert result['rgb'][0, 0, 0] == 30
    np.testing.assert_allclose(result['pose_wc'], pose @ np.linalg.inv(e))
    e[0, 3] = 10
    absent = sensor.register_depth_rgb(rgb, np.array([[2000]], np.uint16), np.eye(3), np.eye(3), e, pose, 1000)
    assert not absent['color_valid'].any()
    assert not absent['rgb'].any()
    xyz = np.array([[1., 0, 2]])
    alignment = np.eye(4); alignment[0, 3] = 7
    aligned, poses = sensor.change_world_frame(xyz, [np.eye(4)], alignment)
    local = (aligned - poses[0][:3, 3]) @ poses[0][:3, :3]
    np.testing.assert_allclose(local, xyz)


def test_sampled_features_use_pixel_centers_and_exclude_padding():
    observations = production('observations')
    raw = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    values, valid = observations.sample_raw(raw, np.array([[0, 0], [1, 1], [2, 0]]), (2, 2), (4, 4), (4, 4))
    torch.testing.assert_close(values[:, 0], torch.tensor([2.5, 12.5, 0.]))
    assert valid.tolist() == [True, True, False]


def test_corruption_area_cap_and_group_zero_is_a_real_instance():
    data = production('data')
    xyz = np.column_stack((np.r_[np.arange(10), 9.02, 9.03, 9.04], np.zeros(13), np.zeros(13)))
    owners = np.r_[np.ones(10, np.int64), np.full(3, 2, np.int64)]
    supports = data.perturb_support(xyz, np.ones(13), owners, 1, direction=0)
    assert supports['clean'].sum() == 10
    assert supports['truncate'].sum() == 7
    assert supports['append'].sum() == 12
    assert supports['truncate_append'].sum() == 9
    groups = [dict(id=0, segments=[7], label='chair'), dict(id=2, segments=[9], label='unmapped')]
    mapped, labels, registry = data.instance_arrays(np.array([7, 7, 9]), groups, {'chair': 5}, [5])
    assert mapped.tolist() == [1, 1, 0]
    assert labels.tolist() == [5, 5, 0]
    assert registry[1]['original_group_id'] == 0


def model_fixture():
    torch.manual_seed(73)
    phi = nn.Sequential(nn.Linear(5, 768), nn.GELU()).requires_grad_(False)
    local = torch.randn(2, 3, 5)
    inputs = dict(raw=torch.randn(2, 5, 4, 5), projected=torch.randn(2, 768, 4, 5),
                  masks=torch.rand(2, 1, 16, 20), view_valid=torch.tensor([True, True]),
                  local_raw=local, local_unit=F.normalize(phi(local), dim=-1),
                  metadata=torch.randn(2, 3, 8), local_valid=torch.ones(2, 3, dtype=torch.bool))
    return phi, inputs


def test_frozen_projection_gradient_grouping_and_permutation():
    model = production('model')
    phi, inputs = model_fixture()
    torch.manual_seed(19)
    surface = model.ReadoutHead(grouping='SURFACE')
    view = model.ReadoutHead(grouping='VIEW')
    view.load_state_dict(surface.state_dict(), strict=True)
    before = {k: v.clone() for k, v in phi.state_dict().items()}
    result = surface(inputs, phi)
    (-result['embedding'][0]).backward()
    assert all(p.grad is None for p in phi.parameters())
    assert all(torch.equal(before[k], v) for k, v in phi.state_dict().items())
    for name in ('ma.final.weight', 'embed.0.weight', 'quality.0.weight', 'queries'):
        gradient = dict(surface.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
    vperm, sperm = torch.tensor([1, 0]), torch.tensor([2, 0, 1])
    permuted = {k: v[vperm] for k, v in inputs.items()}
    for name in ('local_raw', 'local_unit', 'metadata', 'local_valid'):
        permuted[name] = permuted[name][:, sperm]
    torch.testing.assert_close(surface(permuted, phi)['embedding'], result['embedding'], atol=2e-6, rtol=2e-6)
    assert (view(inputs, phi)['embedding'] - result['embedding']).abs().max() > 1e-5


def test_missing_views_and_no_local_support_have_exact_ma_fallback():
    model = production('model')
    phi, inputs = model_fixture()
    head = model.ReadoutHead(grouping='SURFACE')
    inputs['local_valid'][:] = False
    result = head(inputs, phi)
    assert result['fallback'] == 'NO_LOCAL_SUPPORT_MA_FALLBACK'
    torch.testing.assert_close(result['embedding'], result['ma_embedding'], atol=0, rtol=0)
    padded = {k: torch.cat((v, torch.zeros_like(v[:1])), dim=0) for k, v in inputs.items()}
    padded['raw'][-1] = 10000
    padded['projected'][-1] = -10000
    torch.testing.assert_close(head(padded, phi)['embedding'], result['embedding'], atol=0, rtol=0)


def test_auxiliary_unknown_labels_are_masked_and_targets_stay_separate():
    losses = production('losses')
    logits = torch.tensor([[2., -2., 20.]], requires_grad=True)
    result = dict(reliability=logits, local_valid=torch.ones(1, 3, dtype=torch.bool),
                  local_hidden=torch.randn(1, 3, 128, requires_grad=True))
    loss, counts = losses.auxiliary_losses(result, torch.tensor([[1, 0, -1]]))
    loss.backward()
    assert logits.grad[0, 2] == 0
    assert counts['positive_tokens'] == 1 and counts['negative_tokens'] == 1 and counts['correspondence_pairs'] == 0


def test_training_draws_resume_roundtrip_and_optimizer_decay(tmp_path):
    training = production('training')
    objects = [dict(key='a', class_id=2, base=True), dict(key='b', class_id=2, base=True),
               dict(key='c', class_id=7, base=True), dict(key='novel', class_id=9, base=False)]
    draws = training.draw_schedule(objects, 17, updates=3)
    np.testing.assert_array_equal(draws, training.draw_schedule(objects, 17, updates=3))
    assert draws.shape == (48, 3) and 3 not in draws[:, 0]
    assert draws[:9, 1:].tolist() == [[2,1],[4,1],[8,1],[2,2],[4,2],[8,2],[2,3],[4,3],[8,3]]
    torch.manual_seed(9)
    head = production('model').ReadoutHead('SURFACE')
    optimizer = training.optimizer_for(head)
    groups = {id(p): g['weight_decay'] for g in optimizer.param_groups for p in g['params']}
    assert groups[id(head.queries)] == 0 and groups[id(head.embed[3].weight)] == 0
    assert groups[id(head.embed[0].weight)] == .01
    assert training.select_earliest([dict(step=1000,criterion=1-5e-13),dict(step=500,criterion=1)])['step']==500
    # A genuine AdamW continuation matches uninterrupted parameters and RNG draws.
    parameters = [head.queries, head.embed[0].weight]
    def update():
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().mean() for p in parameters).backward(); optimizer.step()
    update()
    training.save_checkpoint(tmp_path/'last.pt', head, optimizer, seed=17, branch='test', step=1, input_identity='fixed')
    expected_rng = torch.rand(4)
    update(); expected = {k:v.clone() for k,v in head.state_dict().items()}
    training.restore_checkpoint(tmp_path/'last.pt', head, optimizer, input_identity='fixed')
    torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)
    update()
    for k,v in head.state_dict().items(): torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
    with pytest.raises(ValueError, match='identity'):
        training.restore_checkpoint(tmp_path/'last.pt', head, optimizer, input_identity='different')


def test_resume_finishes_due_DEV_validation_without_an_optimizer_update(tmp_path):
    from types import SimpleNamespace
    training = production('training'); resume = production('resume')
    phi,inputs = model_fixture(); head = production('model').ReadoutHead('NONE')
    optimizer = training.optimizer_for(head)
    optimizer.zero_grad(); head.ma.final.weight.square().mean().backward(); optimizer.step()
    folder=tmp_path/'LR05_MA_8';folder.mkdir()
    training.save_checkpoint(folder/'last.pt',head,optimizer,seed=17,branch='LR05_MA_8',step=500,
                             input_identity='same-data',validations=[],elapsed_seconds=1.,auxiliary_totals={})
    before=torch.load(folder/'last.pt',weights_only=False)
    torch.manual_seed(8)
    text=F.normalize(torch.randn(200,768),dim=-1)
    fc=SimpleNamespace(phi=phi,base_text=text[:160],text_tensor=text,ids=list(range(200)),
                       base_targets={7:7},device='cpu')
    class Loader:
        def load(self,obj,condition,prefix): return inputs,None,None
    objects=[dict(key='new:1',family='new',class_id=7,base=True)]
    result=resume.repair_validation(folder,'same-data',fc,Loader(),objects)
    assert result['optimizer_updates']==0 and result['completed_validation_step']==500
    after=torch.load(folder/'last.pt',weights_only=False)
    selected=torch.load(folder/'selected.pt',weights_only=False)
    assert after['step']==selected['step']==500 and after['sampler_index']==8000
    assert len(after['validations'])==1 and np.isfinite(after['validations'][0]['criterion'])
    for k,v in before['model'].items():torch.testing.assert_close(v,after['model'][k],rtol=0,atol=0)
    for k,v in before['optimizer']['state'].items():
        for n,x in v.items():torch.testing.assert_close(x,after['optimizer']['state'][k][n],rtol=0,atol=0)
    torch.testing.assert_close(before['RNG']['torch'],after['RNG']['torch'],rtol=0,atol=0)
    assert resume.repair_validation(folder,'same-data',fc,Loader(),objects) is None
    # The weights at525 cannot reconstruct an omitted validation at500.
    training.save_checkpoint(folder/'last.pt',head,optimizer,seed=17,branch='LR05_MA_8',step=525,
                             input_identity='same-data',validations=[])
    with pytest.raises(ValueError,match='historical'):
        resume.repair_validation(folder,'same-data',fc,Loader(),objects)


def test_completed_branch_requires_all_four_actual_DEV_receipts(tmp_path):
    resume=production('resume');common=production('common')
    rows=[]
    for step in (500,1000,1500):
        path=tmp_path/f'dev_{step:04d}.json'
        common.write(path,dict(step=step,seed=17,branch='LR05_MA_8',criterion=1.,original_objects=1,
                               classification_denominator='160_BASE_CLASSES'))
        rows.append(dict(step=step,criterion=1.,receipt=str(path)))
    receipt=dict(status='COMPLETE',seed=17,branch='LR05_MA_8',completed_steps=2000,
                 validations=rows,selected_step=500,DEV_criterion=1.)
    with pytest.raises(ValueError,match='four'):
        resume.validate_branch_receipt(receipt)
    path=tmp_path/'dev_2000.json'
    common.write(path,dict(step=2000,seed=17,branch='LR05_MA_8',criterion=1.,original_objects=1,
                           classification_denominator='160_BASE_CLASSES'))
    rows.append(dict(step=2000,criterion=1.,receipt=str(path)))
    resume.validate_branch_receipt(receipt)
    common.write(path,dict(step=2000,seed=17,branch='LR05_MA_8',criterion=.1,original_objects=1,
                           classification_denominator='160_BASE_CLASSES'))
    with pytest.raises(ValueError,match='criterion'):
        resume.validate_branch_receipt(receipt)


def test_whole_payload_zero_update_parity_and_current_class_ranks():
    from static_ovmap.disagreement_query.outputs import build_payload
    from static_ovmap.module_validation.evaluation import GeometryIdentity,PredictionPayload
    from static_ovmap.module_validation.scannet_study import native_ranks
    geometry=GeometryIdentity('a'*64,'b'*64,'c'*64,'fixed_projection',500)
    owners=np.repeat([1,2,3],[100,200,200]);semantic=np.repeat([2,2,4],[100,200,200])
    nearest=np.arange(500);matched=np.ones(500,bool);raw=owners.copy();raw[100]=0
    g1=PredictionPayload('G1','COMBO','scene',geometry,owners,semantic,native_ranks(owners,{1:2,2:2,3:4},nearest,matched),{})
    d2owners=owners.copy();d2owners[300:]=0;d2sem=semantic.copy();d2sem[300:]=0
    d2=PredictionPayload('D2','COMBO','scene',geometry,d2owners,d2sem,native_ranks(d2owners,{1:2,2:2},nearest,matched),{})
    g1.lock();d2.lock()
    checks=production('output_checks')
    proof=checks.zero_update_parity(g1,d2,raw,[1],nearest,matched)
    assert proof['prediction_key']==g1.prediction_key and proof['owners']==3
    from dataclasses import replace
    bad=replace(g1,instance_ranks=((1,.75),*g1.instance_ranks[1:]));bad.lock()
    with pytest.raises(ValueError,match='zero-update'):
        checks.zero_update_parity(bad,d2,raw,[1],nearest,matched)
    unchanged=build_payload(g1,d2,raw,{},[1],nearest,matched,'no_op',{})
    assert unchanged.prediction_key==g1.prediction_key
    changed=build_payload(g1,d2,raw,{1:5},[1],nearest,matched,'new_F',{})
    np.testing.assert_array_equal(changed.owner_ids,g1.owner_ids)
    np.testing.assert_array_equal(changed.semantic_labels[100:],g1.semantic_labels[100:])
    assert dict(changed.instance_ranks)[1]!=dict(g1.instance_ranks)[1]
    with pytest.raises(ValueError,match='raw-zero'):
        build_payload(g1,d2,raw,{2:5},[2],nearest,matched,'protected',{})
    with pytest.raises(ValueError,match='incumbents'):
        build_payload(g1,d2,raw,{3:5},[3],nearest,matched,'recovered',{})


def test_complete_ordered_pool_coverage_rejects_partial_or_reordered_inputs():
    evaluate=production('evaluation')
    cohorts={'replica':['r1','r0']};methods=['m0','m1']
    rows=[dict(scene=s,method=m,status='COMPLETE',evaluation_identity=s+m) for s in cohorts['replica'] for m in methods]
    pools={'replica':{m:dict(ordered_scenes=['r1','r0'],ordered_inputs=['r1'+m,'r0'+m]) for m in methods}}
    assert evaluate.required_coverage(cohorts,methods,rows,pools)==(4,2)
    with pytest.raises(ValueError,match='coverage'):evaluate.required_coverage(cohorts,methods,rows[:-1],pools)
    pools['replica']['m0']['ordered_inputs'].reverse()
    with pytest.raises(ValueError,match='ordered'):evaluate.required_coverage(cohorts,methods,rows,pools)


def test_strict_gate_uses_both_cohorts_all_metrics_and_fixed_tolerances():
    selection = production('selection')
    metrics = dict(apall=.1, ap50=.2, ap25=.3, miou=.4, macc=.5)
    pools = {c:{m:dict(metrics=metrics.copy()) for m in ('LR00_D2','LR01_G1','LR04_FC_8','LR05_MA_8')}
             for c in ('replica8','scannet_cf18')}
    for c in pools:
        pools[c]['LR04_FC_8']['metrics']['apall'] += .001
        pools[c]['LR05_MA_8']['metrics']['apall'] += .001
    pools['replica8']['LR04_FC_8']['metrics']['macc'] -= 5e-11
    registry = [dict(id=m,trained=m=='LR05_MA_8') for m in pools['replica8']]
    result = selection.select(pools, registry, {'LR05_MA_8':900})
    assert result['best_regression_candidate'] == 'LR04_FC_8'
    assert result['flags']['LR04_FC_8']['material_target_met']
    pools['replica8']['LR04_FC_8']['metrics']['macc'] -= 2e-10
    assert selection.select(pools,registry,{'LR05_MA_8':900})['best_regression_candidate']=='LR05_MA_8'
    pools['scannet_cf18']['LR05_MA_8']['metrics']['apall'] = .1+5e-11
    result = selection.select(pools,registry,{'LR05_MA_8':900})
    assert result['research_retained'] == 'LR01_G1' and result['best_regression_candidate'] is None
    assert result['deployment']=='N0_UNCHANGED'
    pools['replica8']['LR01_G1']['metrics']['miou'] = float('nan')
    with pytest.raises(ValueError,match='finite'): selection.select(pools,registry,{'LR05_MA_8':900})


def test_gt_reference_preserves_undefined_and_separates_F_from_final_labels():
    diagnostics = production('diagnostics')
    refs = [dict(owner=1,best=dict(iou=.8,**{'class':7}),instance_size_eligible=True),
            dict(owner=2,best=None,instance_size_eligible=True),
            dict(owner=3,best=dict(iou=.5,**{'class':7}),instance_size_eligible=True)]
    decisions = {'1':dict(old_class=2,old_F_top1=7,new_F_top1=2,applied_class=7),
                 '2':dict(old_class=7,old_F_top1=7,new_F_top1=7,applied_class=7),
                 '3':dict(old_class=2,old_F_top1=2,new_F_top1=7,applied_class=7)}
    result = diagnostics.owner_outcomes(decisions,refs,.5)
    assert result['original_incumbent_denominator']==3
    assert result['counts']['final']['WRONG_TO_RIGHT']==1
    assert result['counts']['F']['RIGHT_TO_WRONG']==1
    assert result['counts']['final']['UNDEFINED_GT_REFERENCE']==2
    assert len(result['ledger'])==3
    assert result['ledger'][2]['defined_GT_reference'] is False


def test_diagnostic_support_identity_matches_parent_mesh_and_source_rows():
    from static_ovmap.minimal_instance_repair.support_units import build_units
    from static_ovmap.module_validation.evaluation import GeometryIdentity
    diagnostics = production('diagnostics')
    xyz=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    owners=np.ones(3,np.int64);faces=np.array([[0,1,2]])
    geometry=GeometryIdentity('a'*64,'b'*64,'c'*64,'fixed_projection',3)
    units=build_units(xyz,faces,owners,owners,geometry,
                      dict(minimum_residual_source_rows=1,candidate_cap_inherited=32,max_repair_seeds=1))
    rows=np.arange(3,dtype=np.int64)
    assert diagnostics.fixed_support_hash(geometry,rows)==units.units['I:1'].support_hash
    other=GeometryIdentity('f'*64,'b'*64,'c'*64,'fixed_projection',3)
    assert diagnostics.fixed_support_hash(other,rows)!=units.units['I:1'].support_hash


def test_regression_table_preserves_negative_results_and_rejects_missing_pools():
    import json
    reporting = production('reporting')
    registry = json.loads(Path('configs/static_ovmap/learned_object_readout_v1.json').read_text())['methods']
    methods = [r['id'] for r in registry]+['R29_LR05_MA_8','R29_LR06_MV_VIEW']
    pools = {c:{m:dict(metrics=dict(apall=.1,ap50=.2,ap25=.3,miou=.4,macc=.5)) for m in methods}
             for c in ('replica8','scannet_cf18')}
    pools['replica8']['LR05_MA_8']['metrics']['apall'] = .099
    counts = {r['id']:100 for r in registry if r['trained']}
    rows = reporting.table1_rows({'methods':methods,'pooled_metrics':pools},registry,counts)
    row = next(r for r in rows if r['cohort']=='replica8' and r['method']=='LR05_MA_8')
    assert row['apall']==.099 and row['seed']==17 and row['trained'] is True
    assert len(rows)==22
    del pools['scannet_cf18']['LR07_MV_SURFACE']
    with pytest.raises(KeyError):reporting.table1_rows({'methods':methods,'pooled_metrics':pools},registry,counts)


def test_missing_capture_clock_does_not_invent_readout_time():
    reporting = production('reporting')
    original = dict(elapsed_seconds=804.9, peak_allocated_bytes=2**30,
                    peak_reserved_bytes=2*2**30)
    row = reporting.map_readout_stage({}, original)
    assert row['seconds'] is None
    assert row['timing_missing_reason']=='CAPTURE_CLOCK_NOT_RECORDED'
    assert row['peak_allocated_GiB']==1
    assert row['add_to_total'] is False
    row = reporting.map_readout_stage({'elapsed_seconds':100.}, original)
    assert row['seconds']==pytest.approx(704.9)
    with pytest.raises(ValueError):
        reporting.map_readout_stage({'elapsed_seconds':900.}, original)
