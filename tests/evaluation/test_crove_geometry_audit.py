import numpy as np


def test_full_geometry_counts_include_duplicates_but_voxels_do_not():
    from scripts.evaluation.audit_crove_apartment_readout_geometry import (
        full_geometry_counts,
    )

    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    result = full_geometry_counts(points, np.zeros((1, 3)), np.array([0, 2, 2]))
    assert result["source_rows"] == 3
    assert result["confirmed_free_conflict_rows"] == 2  # Strictly less than 5 cm.
    assert result["unique_5cm_voxels"] == 2
    assert result["conflicting_5cm_voxels"] == 1
    assert result["fixed_legacy_sources"]["2"]["source_rows"] == 2
    assert result["fixed_legacy_sources"]["2"]["confirmed_free_conflict_rows"] == 1
