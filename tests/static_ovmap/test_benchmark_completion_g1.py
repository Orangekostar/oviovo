"""G1: numerical contracts; real-source comparison is recorded by cache_post."""
import numpy as np
from static_ovmap.benchmark_completion_v2 import calibration as c


def test_nmix_preserves_endpoints_normalizes_interior_and_handles_zero():
    # Using unnormalized score MIX for NMIX fails the independent sqrt(1/2) value.
    operation = getattr(c, 'nmix_scores', None)
    assert callable(operation), 'NMIX numerical operator is not implemented'
    b = np.array([[1., 0.], [1., 0.]], np.float32)
    z = np.array([[0., 1.], [-1., 0.]], np.float32)
    text = np.eye(2, dtype=np.float32)
    sb, sz = b @ text.T, z @ text.T
    zero, flags = operation(0., sb, sz, b, z, text)
    one, _ = operation(1., sb, sz, b, z, text)
    middle, flags = operation(.5, sb, sz, b, z, text)
    assert np.array_equal(zero, sb) and np.array_equal(one, sz)
    np.testing.assert_allclose(middle[0], [0.7071067811865475] * 2, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(middle[1], [1., 0.])
    assert flags.tolist() == [False, True]
