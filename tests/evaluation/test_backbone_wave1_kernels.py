"""Scoped production-kernel checks; scene evidence is recorded separately."""

import numpy as np
import pytest

from static_ovmap.backbone_wave1.association import directional_assignments, plan_objects
from static_ovmap.backbone_wave1.fusion import canonical_fragments, simultaneous_fusion


@pytest.mark.parametrize("sizes", [(100, 80, 70), (100, 100, 100), (40,)])
def test_simultaneous_partition_and_native_initial_gate(sizes):
    depth = np.ones((15, 20), np.int32)
    instances = np.zeros_like(depth)
    offset = 0
    for obj, size in enumerate(sizes, 1):
        instances.ravel()[offset:offset + size] = obj
        offset += size
    rows = simultaneous_fusion(depth, instances)
    np.testing.assert_array_equal(np.stack([r.mask for r in rows]).sum(0), depth)
    assert all(np.isfinite(r.overlap_ratio) for r in rows)
    assert any(r.mask.sum() == sizes[0] for r in rows)
    assert simultaneous_fusion(np.ones((7, 7), int), np.ones((7, 7), int)) == []


def test_physical_fusion_is_invariant_to_group_numbers_and_depth_visit_order():
    depth = np.repeat([4, 2], 300).reshape(20, 30)
    instances = np.zeros_like(depth)
    for start, stop, obj in [(0, 100, 11), (100, 180, 22), (180, 250, 33),
                             (300, 390, 22), (390, 480, 44)]:
        instances.ravel()[start:stop] = obj
    renamed = np.zeros_like(instances)
    for original, replacement in [(11, 3), (22, 100), (33, 1), (44, 19)]:
        renamed[instances == original] = replacement
    assert canonical_fragments(simultaneous_fusion(depth, instances)) == canonical_fragments(
        simultaneous_fusion(np.where(depth == 4, 2, 4), renamed))


def test_directional_assignment_uses_distinct_denominators_and_dummy_unmatched():
    result = directional_assignments([[800, 500, 500], [700, 100, 700]],
                                     [2700, 1600], [2400, 1400, 1400])
    assert result.forward == {(0, 0), (1, 2)}
    assert result.backward == {(0, 1), (1, 2)}
    assert result.mutual == {(1, 2)}
    assert directional_assignments([[50, 20]], [200], [100, 100]).forward == set()
    assert directional_assignments(np.zeros((2, 0)), [100, 100], []).mutual == set()
    with pytest.raises(ValueError, match="support"):
        directional_assignments([[100, 100]], [150], [100, 100])


def test_planner_reverse_area_includes_probe_support_outside_local_foreground():
    depth = np.ones((20, 30), np.float32)
    groups = np.zeros_like(depth, np.int32)
    groups[:, :10] = 17
    prior = np.full_like(groups, 9)
    fragments = simultaneous_fusion(np.ones_like(groups), groups)
    plan = plan_objects(fragments, depth, prior, mode="object_bidirectional")
    assert plan.local_areas == [200]
    assert plan.global_areas == [600]
    assert plan.intersections == [[200]]
    assert plan.planned_owners == {17: 9}
    assert plan.forward_coverages == [[1.0]]
    assert plan.reverse_coverages == [[pytest.approx(1 / 3)]]
