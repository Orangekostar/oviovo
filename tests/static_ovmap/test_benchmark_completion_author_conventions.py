"""Pinned author confidence and overlap conventions, using temporary masks only."""
import importlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from static_ovmap.benchmark_completion_v2.common import digest_file

def load(name):
    return importlib.import_module('static_ovmap.benchmark_completion_v2.' + name)

class AuthorConventionTests(unittest.TestCase):
    def test_open3dis_uses_author_evaluation_confidence_and_preserves_overlap(self):
        external = load('external')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'scene.pth'
            torch.save(dict(ins=[dict(length=5, counts='1 3'), dict(length=5, counts='3 3')], final_class=torch.tensor([1, 0]), conf=torch.tensor([0.2, 0.9])), path)
            before = digest_file(path)
            result = external.decode_open3dis(path, target_points=5, class_ids=[7, 11])
            np.testing.assert_array_equal(result['pred_scores'], [1.0, 1.0])
            np.testing.assert_array_equal(result['pred_classes'], [11, 7])
            np.testing.assert_array_equal(result['pred_masks'], [[1, 0], [1, 0], [1, 1], [0, 1], [0, 1]])
            self.assertEqual(digest_file(path), before)

    def test_other_native_manifest_scores_remain_unchanged(self):
        external = load('external')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            np.save(folder / 'first.npy', [1, 0, 1])
            np.save(folder / 'second.npy', [0, 1, 1])
            (folder / 'masks.txt').write_text('first.npy 7 0.25\nsecond.npy 11 0.875\n')
            result = external.decode_text_manifest(folder / 'masks.txt', target_points=3)
            np.testing.assert_array_equal(result['pred_scores'], [0.25, 0.875])
