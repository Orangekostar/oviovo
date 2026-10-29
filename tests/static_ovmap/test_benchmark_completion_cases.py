def test_case_selection_uses_owner_order_and_includes_harm():
    from static_ovmap.benchmark_completion_v2.cases import choose_cases
    pairs = [dict(owner=30, class_id=3), dict(owner=22, class_id=2),
             dict(owner=10, class_id=1), dict(owner=40, class_id=4)]
    before = {30: 3, 22: 3, 10: 2, 40: 5}
    after = {30: 4, 22: 2, 10: 1, 40: 5}
    result = choose_cases(pairs, before, after)
    assert result['cases']['corrected']['owner'] == 10
    assert result['cases']['harmed']['owner'] == 30
    assert result['counts'] == dict(matched=4, changed=3, corrected=2, harmed=1,
                                    other_changed=0, unchanged=1)
    assert pairs[0]['owner'] == 30


def test_absent_case_categories_remain_absent_and_unknowns_count():
    from static_ovmap.benchmark_completion_v2.cases import choose_cases
    result = choose_cases([dict(owner=1, class_id=7)], {1: 0}, {1: 0})
    assert result['cases'] == dict(corrected=None, harmed=None)
    assert result['counts'] == dict(matched=1, changed=0, corrected=0, harmed=0,
                                    other_changed=0, unchanged=1)


def test_publication_includes_case_artifacts_and_excludes_raw_geometry(tmp_path):
    from static_ovmap.benchmark_completion_v2.publication_export import candidates
    folder = tmp_path / 'cases/scannet200_val/scene0568_00'
    folder.mkdir(parents=True)
    allowed = {'data.json', 'receipt.json', 'caption.md', 'figure.pdf', 'figure.png'}
    for name in allowed | {'raw_geometry.npz', 'raw_mesh.ply', 'raw_GT.json'}:
        (folder / name).write_bytes(b'fixture')
    assert {path.name for path in candidates(tmp_path)} == allowed
