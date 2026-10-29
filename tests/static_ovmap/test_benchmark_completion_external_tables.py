def test_partial_local_results_cannot_replace_complete_author_reference():
    from static_ovmap.benchmark_completion_v2.tables import external_cell, _fmt
    key = 'external:MaskClustering:replica48'
    reference = 'author:MaskClustering:replica48'
    results = {key: dict(full_scene_coverage=False, metrics=dict(apall=.99)),
               reference: dict(compatible_frozen_variant=True, metrics=dict(apall=.12))}
    cell = external_cell(results, 'MaskClustering', 'replica48', 'apall')
    assert cell['value'] == .12 and cell['results'] == [reference]
    assert cell['annotation'] == '†' and _fmt(cell) == '12.00†'
    results[key]['full_scene_coverage'] = True
    cell = external_cell(results, 'MaskClustering', 'replica48', 'apall')
    assert cell['value'] == .99 and cell['results'] == [key]
    assert cell.get('annotation', '') == '' and _fmt(cell) == '99.00'
    results[key]['full_scene_coverage'] = False
    results[reference]['compatible_frozen_variant'] = False
    cell = external_cell(results, 'MaskClustering', 'replica48', 'apall')
    assert cell['value'] is None and _fmt(cell) == '--'
