from pathlib import Path

from src.static_ovmap.released_loader import load_released_module


def test_relative_imports_are_bound_to_each_evaluator_root(tmp_path):
    modules = []
    for name, value in [('original', 48), ('rebuilt', 51)]:
        root = tmp_path/name
        (root/'utils').mkdir(parents=True)
        (root/'utils/semantic_const.py').write_text(f'VALUE = {value}\n')
        (root/'eval_utils.py').write_text('def init():\n    from .utils.semantic_const import VALUE\n    return VALUE\n')
        modules.append(load_released_module(root/'eval_utils.py'))
    assert [m['init']() for m in modules] == [48, 51]
