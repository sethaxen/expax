import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import expax


def _dense_matvec(x, matrix):
    return matrix @ x


def test_vectors_needing_resampling_prioritizes_earlier_current_vectors():
    block = jnp.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
        ]
    )
    previous = block[2:]

    received = jax.jit(expax.normest._vectors_needing_resampling)(block, previous)

    np.testing.assert_array_equal(received, [False, True, True])


def test_resample_parallel_vectors_uses_one_block_retry_loop():
    block = jnp.ones((2, 3))
    previous = jnp.array([[1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])

    jaxpr = jax.make_jaxpr(expax.normest._resample_parallel_vectors)(
        jax.random.key(0), block, previous
    ).jaxpr

    assert sum(eqn.primitive.name == "while" for eqn in jaxpr.eqns) == 1


def test_resample_parallel_vectors_removes_all_conflicts_at_minimum_dimension():
    previous = jnp.array([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])

    received = jax.jit(expax.normest._resample_parallel_vectors)(
        jax.random.key(0), previous, previous
    )

    needs_resampling = expax.normest._vectors_needing_resampling(received, previous)
    assert not bool(jnp.any(needs_resampling))


def test_initial_block_is_normalized_and_has_no_parallel_vectors():
    _, received = jax.jit(expax.normest._initial_block, static_argnums=(1, 2, 3))(
        jax.random.key(0), 4, 3, jnp.float32
    )

    np.testing.assert_array_equal(received[0], jnp.full(4, 0.25))
    unscaled = received * received.shape[-1]
    previous = jnp.empty((0, received.shape[-1]), dtype=received.dtype)
    needs_resampling = expax.normest._vectors_needing_resampling(unscaled, previous)
    assert not bool(jnp.any(needs_resampling))


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


def test_onenormest_skips_adjoint_on_final_step(monkeypatch):
    matrix = jnp.array(
        [
            [-0.98128909, 0.03826571, -0.9366923, -0.69846063, -0.29595587, 0.31419297],
            [0.63443117, -1.08491685, 0.25676977, 0.47592282, 0.85303157, -0.16485969],
            [1.04476288, 1.53548419, 0.44388999, -0.70602427, -1.39838878, 0.33018547],
            [-1.73440706, 0.72005797, 0.56592782, -0.79026839, -0.11883412, 0.7663085],
            [
                -0.48582007,
                0.56101067,
                -0.30619089,
                -0.27750305,
                -0.56399731,
                -1.92742072,
            ],
            [-1.9018335, 0.86788021, -0.34433705, 1.53220957, -0.63933366, 1.56270841],
        ]
    )
    adjoint_calls = []

    def record(_):
        adjoint_calls.append(None)

    original_linear_adjoint = expax.normest._linear_adjoint

    def instrumented_adjoint(func, *primals):
        adjoint = original_linear_adjoint(func, *primals)

        def instrumented(vector):
            jax.debug.callback(record, vector[0])
            return adjoint(vector)

        return instrumented

    monkeypatch.setattr(expax.normest, "_linear_adjoint", instrumented_adjoint)
    estimate, _ = expax.normest.onenormest(block_size=2, max_steps=2)
    received = jax.jit(lambda key: estimate(_dense_matvec, jnp.zeros(6), key, matrix))(
        jax.random.key(0)
    )
    jax.block_until_ready(received)

    assert len(adjoint_calls) == 2


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
