from pathlib import Path

import numpy as np
import pytest

from src.static_ovmap.module_validation.evaluation import (
    GeometryIdentity,
    PredictionPayload,
)
from src.static_ovmap.replica_transfer import evaluation, protocol


def test_replica_evaluation_uses_transfer_authorization():
    assert evaluation.require_access is protocol.require_access


def test_released_evaluator_is_replica(tmp_path):
    upstream = Path('/home/ww/crove/ovimap-module-validation-upstream')
    adapter = evaluation.released_evaluator(upstream, tmp_path)
    # Resolve the released vocabulary independently: objects only for AP.
    names, semantic_ids, ap_ids = protocol.vocabulary(upstream)
    assert len(names) == len(semantic_ids) == 51
    assert set(semantic_ids) - set(ap_ids) == {1, 2, 3}
    assert tuple(map(int, adapter._evaluator.evaluator_namespace['VALID_CLASS_IDS'])) == ap_ids


def test_real_replica_evaluator_includes_wall_in_semantics_only(tmp_path):
    upstream = Path('/home/ww/crove/ovimap-module-validation-upstream')
    owners = np.repeat([1, 2, 3], 150)
    labels = np.repeat([1, 8, 32], 150)
    gt_path = tmp_path / 'gt.npy'
    np.save(gt_path, labels * 1000 + owners)
    targets = {'nearest': np.arange(len(labels)), 'matched': np.ones(len(labels), bool),
               'gt_semantic': labels, 'valid_ids': list(range(1, 52)), 'gt_instance_path': gt_path}
    payloads = []
    for method, semantic in [('N0', labels), ('WALL_CHANGED', np.repeat([2, 8, 32], 150))]:
        payload = PredictionPayload(method_id=method, branch='N0' if method == 'N0' else 'S',
            scene_id='synthetic', geometry=GeometryIdentity('a' * 64, 'b' * 64, 'c' * 64, 'projection', len(labels)),
            owner_ids=owners, semantic_labels=semantic, instance_ranks=((1, .9), (2, .8), (3, .7)),
            logical_cost={}, metadata={})
        payload.lock()
        payloads.append(payload)
    rows = evaluation.evaluate_predictions(payloads, {'synthetic': targets}, upstream, tmp_path / 'eval')
    assert rows[0]['metrics']['uap'] == pytest.approx(1)
    assert rows[1]['metrics']['uap'] == pytest.approx(1)
    assert rows[0]['metrics']['miou'] == pytest.approx(1)
    assert rows[1]['metrics']['miou'] == pytest.approx(2 / 3)
    assert all(row['metrics']['trace_parity'] for row in rows)
