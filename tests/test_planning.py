import jax.numpy as jnp
import numpy as np

from expax._planning import (
    _compute_p_max,
    _condition_3_13,
    _select_from_matrix,
    _select_from_norm,
)


def test_compute_p_max_uses_degree_constraint():
    assert _compute_p_max(55) == 8
    assert _compute_p_max(1) == 2


def test_condition_3_13_uses_estimator_cost():
    theta = jnp.array([1.0, 4.0])

    assert bool(_condition_3_13(1.0, 80.0, 8.0, theta, 2))
    assert not bool(_condition_3_13(1.0, 81.0, 8.0, theta, 2))


def test_select_from_norm_chooses_smallest_degree_on_cost_tie():
    degree, scaling = _select_from_norm(
        jnp.array(2.0), jnp.array(1.0), jnp.array([1.0, 4.0])
    )

    assert int(degree) == 1
    assert float(scaling) == 2.0


def test_select_from_norm_handles_zero_and_vector_scales():
    degrees, scalings = _select_from_norm(
        jnp.array([0.0, 2.0, 9.0]),
        jnp.array(1.0),
        jnp.array([1.0, 4.0]),
    )

    np.testing.assert_array_equal(degrees, [0, 1, 2])
    np.testing.assert_array_equal(scalings, [1.0, 2.0, 3.0])


def test_select_from_matrix_implements_equation_3_14():
    matrix = jnp.array([[2.0, 1.0], [5.0, 0.5]])
    valid = jnp.array([[True, True], [False, True]])

    degrees, scalings = _select_from_matrix(jnp.array([0.0, 1.0, 5.0]), matrix, valid)

    np.testing.assert_array_equal(degrees, [0, 1, 2])
    np.testing.assert_array_equal(scalings, [1.0, 2.0, 3.0])


def test_select_from_matrix_uses_one_scaling_for_zero_estimates():
    matrix = jnp.zeros((2, 3))
    valid = jnp.ones_like(matrix, dtype=bool)

    degree, scaling = _select_from_matrix(jnp.array(2.0), matrix, valid)

    assert int(degree) == 1
    assert float(scaling) == 1.0
