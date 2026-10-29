"""Boundary-only frame-list adapter around the inherited native OVI implementation."""
import ast
import os
from pathlib import Path
import sys

from .common import verified, digest_file


def main():
    plan = verified(os.environ['OVMAP_BENCHMARK_FRAME_PLAN'])
    source = Path(os.environ['OVMAP_BENCHMARK_NATIVE_ENTRY'])
    if digest_file(source) != os.environ['OVMAP_BENCHMARK_NATIVE_ENTRY_SHA256']:
        raise ValueError('Bound original native entry changed')
    frames = plan['frame_ids']
    if not frames or len(frames) > 200 or frames != sorted(set(frames)):
        raise ValueError('Invalid frozen finite-pose frame list')
    tree = ast.parse(source.read_text())
    main_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    counts = dict(num_frames=0, frame_ids=0)
    for node in ast.walk(main_function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == 'num_frames':
            if ast.unparse(node.value) != 'int(np.ceil((end - start) / step))':
                raise ValueError('Native frame-count expression differs from reviewed source')
            node.value = ast.parse('len(_OVMAP_BENCHMARK_FRAMES)', mode='eval').body
            counts[name] += 1
        elif name == 'frame_ids':
            if ast.unparse(node.value) != 'range(start, end, step)':
                raise ValueError('Native frame iterator differs from reviewed source')
            node.value = ast.Name(id='_OVMAP_BENCHMARK_FRAMES', ctx=ast.Load())
            counts[name] += 1
    if counts != dict(num_frames=1, frame_ids=1):
        raise ValueError('Native boundary adaptation must change exactly two frame declarations')
    ast.fix_missing_locations(tree)
    sys.path.insert(0, str(source.parent))
    sys.argv[0] = str(source)
    namespace = dict(__name__='__main__', __file__=str(source), _OVMAP_BENCHMARK_FRAMES=frames)
    exec(compile(tree, str(source), 'exec'), namespace)


if __name__ == '__main__':
    main()
