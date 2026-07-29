import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import expax


def _dense_matvec(x, matrix):
    return matrix @ x


def test_onenormest_reports_code_fragment_3_1_cost():
    _, cost = expax.normest.onenormest(block_size=3)

    assert cost == 12


@pytest.mark.parametrize("seed", range(4))
def test_onenormest_is_exact_for_diagonal_operators(seed):
    matrix = jnp.diag(jnp.array([-2.0, 7.0, 1.0, -4.0]))
    estimate, _ = expax.normest.onenormest(block_size=2)

    received = estimate(_dense_matvec, jnp.zeros(4), jax.random.key(seed), matrix)

    np.testing.assert_allclose(received, 7.0)


@pytest.mark.parametrize("seed", range(4))
def test_onenormest_bounds_complex_nonnormal_operator(seed):
    matrix = jnp.array(
        [
            [1 + 2j, 8 - 3j, 0, 1j],
            [0, -2j, 5 + 4j, 0],
            [3, 0, -1 + 1j, 6],
            [0, 2 - 1j, 0, 4j],
        ],
        dtype=jnp.complex128,
    )
    exact = np.linalg.norm(np.asarray(matrix), ord=1)
    estimate, _ = expax.normest.onenormest(block_size=2)

    received = float(
        estimate(
            _dense_matvec,
            jnp.zeros(4, dtype=jnp.complex128),
            jax.random.key(seed),
            matrix,
        )
    )

    assert exact / 3 <= received <= exact


def test_onenormest_accepts_parameterized_pytree_operators():
    v_like = {
        "left": jnp.zeros(2, dtype=jnp.float64),
        "right": jnp.zeros(1, dtype=jnp.float64),
    }
    matrix = jnp.array([[2.0, -1.0, 0.0], [0.0, 3.0, 4.0], [1.0, 0.0, -2.0]])

    def matvec(x, scale, matrix):
        flat, unravel = ravel_pytree(x)
        return unravel(scale * (matrix @ flat))

    estimate, _ = expax.normest.onenormest(block_size=2)
    received = jax.jit(
        lambda scale, matrix, key: estimate(matvec, v_like, key, scale, matrix)
    )(
        jnp.array(2.0),
        matrix,
        jax.random.key(12),
    )

    exact = 2.0 * np.linalg.norm(np.asarray(matrix), ord=1)
    assert exact / 3 <= float(received) <= exact


def test_onenormest_handles_operator_powers_without_changing_its_cost():
    matrix = jnp.array([[2.0, 1.0, 0.0], [0.0, -1.0, 3.0], [1.0, 0.0, 2.0]])
    power = 3

    def powered_matvec(x, matrix):
        for _ in range(power):
            x = matrix @ x
        return x

    estimate, cost = expax.normest.onenormest(block_size=2)
    received = estimate(powered_matvec, jnp.zeros(3), jax.random.key(5), matrix)

    exact = np.linalg.norm(np.linalg.matrix_power(np.asarray(matrix), power), ord=1)
    assert exact / 3 <= float(received) <= exact
    assert power * cost == 24


def test_onenormest_does_not_materialize_small_operators():
    estimate, _ = expax.normest.onenormest(block_size=3)

    with pytest.raises(ValueError, match="block_size"):
        estimate(_dense_matvec, jnp.zeros(3), jax.random.key(0), jnp.eye(3))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block_size": 0}, "block_size"),
        ({"max_steps": 1}, "max_steps"),
    ],
)
def test_onenormest_rejects_invalid_static_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        expax.normest.onenormest(**kwargs)
