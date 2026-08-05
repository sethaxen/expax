import jax.numpy as jnp
import numpy as np
import pytest

from expax._theta import _theta


def test_double_theta_matches_reference_values():
    theta = _theta(jnp.dtype(jnp.float64), None, 55)

    np.testing.assert_allclose(
        [theta[0], theta[29], theta[54]],
        [2.220446049250313e-16, 3.5396663487436895, 9.8674966757534],
        rtol=1e-15,
    )


def test_single_theta_matches_reference_values():
    theta = _theta(jnp.dtype(jnp.float32), None, 55)

    np.testing.assert_allclose(
        [theta[0], theta[29], theta[54]],
        [1.1920927533992653e-7, 6.32108211517334, 13.358800888061523],
        rtol=1e-15,
    )


def test_requested_tolerance_uses_supported_floor():
    received = _theta(jnp.dtype(jnp.float64), 1e-10, 55)
    expected = _theta(jnp.dtype(jnp.float64), 2.0**-53, 55)

    assert received == expected


def test_requested_tolerance_is_clamped_to_dtype_floor():
    received = _theta(jnp.dtype(jnp.float64), 2.0**-60, 55)
    expected = _theta(jnp.dtype(jnp.float64), 2.0**-53, 55)

    assert received == expected


def test_unsupported_tolerance_is_rejected():
    with pytest.raises(ValueError, match="theta values"):
        _theta(jnp.dtype(jnp.float64), 1e-2, 55)


def test_dtype_without_a_supported_tolerance_is_rejected():
    with pytest.raises(ValueError, match="theta values"):
        _theta(np.float128, None, 55)


def test_custom_theta_values_are_positive_and_increase_with_degree():
    theta = np.asarray(_theta(jnp.dtype(jnp.float64), 1e-10, 12))

    assert np.all(theta > 0)
    assert np.all(np.diff(theta) > 0)
