import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from expax._operator import _linear_adjoint, _validate_vector_space


def _assert_tree_allclose(received, expected):
    assert jax.tree.structure(received) == jax.tree.structure(expected)
    for received_leaf, expected_leaf in zip(
        jax.tree.leaves(received), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(received_leaf, expected_leaf)


def test_linear_adjoint_transposes_real_array_operator():
    matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    def matvec(x):
        return matrix @ x

    vector = jnp.array([2.0, -1.0])
    adjoint = _linear_adjoint(matvec, jnp.zeros_like(vector))

    np.testing.assert_allclose(adjoint(vector)[0], matrix.T @ vector)


def test_linear_adjoint_is_hermitian_for_complex_pytree():
    x_like = {
        "a": jnp.zeros(2, dtype=jnp.complex128),
        "b": jnp.zeros(1, dtype=jnp.complex128),
    }
    matrix = jnp.array(
        [[1 + 2j, 2, 0], [3j, -1j, 4], [2, 1, 3 - 2j]],
        dtype=jnp.complex128,
    )

    def matvec(x):
        flat, unravel = ravel_pytree(x)
        return unravel(matrix @ flat)

    vector = {
        "a": jnp.array([1 + 1j, 2 - 1j]),
        "b": jnp.array([3j]),
    }
    adjoint = _linear_adjoint(matvec, x_like)
    flat, unravel = ravel_pytree(vector)

    _assert_tree_allclose(adjoint(vector)[0], unravel(matrix.T.conj() @ flat))


def test_validate_vector_space_returns_coordinate_map():
    x_like = {
        "a": jnp.zeros((2, 2), dtype=jnp.float32),
        "b": jnp.zeros(3, dtype=jnp.float32),
    }

    flat, unravel = _validate_vector_space(x_like)

    assert flat.shape == (7,)
    assert flat.dtype == jnp.float32
    _assert_tree_allclose(unravel(flat), x_like)


@pytest.mark.parametrize(
    ("x_like", "error"),
    [
        ({}, ValueError),
        (jnp.zeros(2, dtype=jnp.int32), TypeError),
        (
            {
                "a": jnp.zeros(1, dtype=jnp.float32),
                "b": jnp.zeros(1, dtype=jnp.float64),
            },
            TypeError,
        ),
        (jnp.zeros(0, dtype=jnp.float32), ValueError),
    ],
)
def test_validate_vector_space_rejects_unsupported_spaces(x_like, error):
    with pytest.raises(error):
        _validate_vector_space(x_like)
