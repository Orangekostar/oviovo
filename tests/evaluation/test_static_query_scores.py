import numpy as np
import pytest

from src.static_ovmap.query_scores import score_queries


def test_query_scores_use_explicit_ids_and_reject_same_dimension_different_spaces():
    result = score_queries(np.array([0., 2.]), np.eye(2), [7, 49], 'clip', 'clip')
    assert result['class_id'] == 49
    with pytest.raises(ValueError):
        score_queries(np.array([0., 2.]), np.eye(2), [7, 49], 'siglip', 'clip')
