import jax.numpy as jnp
import numpy as np

from expax._theta import _compute_theta, _theta


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


def test_computed_theta_recovers_reference_bound():
    received = _compute_theta(30, 2.0**-53)

    np.testing.assert_allclose(received, 3.5396663487436895, rtol=1e-14)


def test_custom_theta_values_are_positive_and_increase_with_degree():
    theta = np.asarray(_theta(jnp.dtype(jnp.float64), 1e-10, 12))

    assert np.all(theta > 0)
    assert np.all(np.diff(theta) > 0)
