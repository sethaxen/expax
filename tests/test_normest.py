import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Inexact, PyTree, Real

import expax


def _dense_matvec(
    x: Inexact[Array, " dim"],
    matrix: Inexact[Array, "dim dim"],
) -> Inexact[Array, " dim"]:
    return matrix @ x


def test_vectors_needing_resampling_prioritizes_earlier_current_vectors() -> None:
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


def test_complex_sign_vectors_never_trigger_parallel_stop() -> None:
    sign_vectors = jnp.array([[1.0, 1j, -1.0, -1j]], dtype=jnp.complex64)

    received = expax.normest._all_sign_vectors_parallel_to_previous(
        1,
        sign_vectors,
        sign_vectors,
        check_parallelism=False,
    )

    assert not bool(received)


def test_complex_sign_vectors_are_not_resampled() -> None:
    key = jax.random.key(0)
    sign_vectors = jnp.array([[1.0, 1j, -1.0, -1j]], dtype=jnp.complex64)

    received_key, received_sign_vectors = expax.normest._resample_parallel_sign_vectors(
        key,
        sign_vectors,
        sign_vectors,
        check_parallelism=False,
    )

    np.testing.assert_array_equal(
        jax.random.key_data(received_key),
        jax.random.key_data(key),
    )
    np.testing.assert_array_equal(received_sign_vectors, sign_vectors)


def test_resample_parallel_vectors_uses_one_block_retry_loop() -> None:
    block = jnp.ones((2, 3))
    previous = jnp.array([[1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])

    jaxpr = jax.make_jaxpr(expax.normest._resample_parallel_vectors)(
        jax.random.key(0), block, previous
    ).jaxpr

    assert sum(eqn.primitive.name == "while" for eqn in jaxpr.eqns) == 1


def test_resample_parallel_vectors_removes_all_conflicts_at_minimum_dimension() -> None:
    previous = jnp.array([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])

    received = jax.jit(expax.normest._resample_parallel_vectors)(
        jax.random.key(0), previous, previous
    )

    needs_resampling = expax.normest._vectors_needing_resampling(received, previous)
    assert not bool(jnp.any(needs_resampling))


def test_initial_block_is_normalized_and_has_no_parallel_vectors() -> None:
    _, received = jax.jit(expax.normest._initial_block, static_argnums=(1, 2, 3))(
        jax.random.key(0), 4, 3, jnp.float32
    )

    np.testing.assert_array_equal(received[0], jnp.full(4, 0.25))
    unscaled = received * received.shape[-1]
    previous = jnp.empty((0, received.shape[-1]), dtype=received.dtype)
    needs_resampling = expax.normest._vectors_needing_resampling(unscaled, previous)
    assert not bool(jnp.any(needs_resampling))


def test_initial_block_resamples_parallel_real_vectors_stored_as_complex() -> None:
    _, received = expax.normest._initial_block(
        jax.random.key(4),
        3,
        2,
        jnp.complex64,
    )

    unscaled = received * received.shape[-1]
    assert not bool(expax.normest._parallel_vectors(unscaled[:1], unscaled[1:])[0, 0])


def test_onenormest_reports_code_fragment_3_1_cost() -> None:
    _, cost = expax.normest.onenormest(block_size=3)

    assert cost == 12


@pytest.mark.parametrize("seed", range(4))
def test_onenormest_is_exact_for_diagonal_operators_above_its_cost(seed: int) -> None:
    matrix = jnp.diag(jnp.array([-2.0, 7.0, 1.0, -4.0, 3.0, -6.0, 5.0, 2.0, -1.0]))
    estimate, _ = expax.normest.onenormest(block_size=2)

    received = estimate(_dense_matvec, jnp.zeros(9), jax.random.key(seed), matrix)

    np.testing.assert_allclose(received, 7.0)


def test_onenormest_exact_estimate_is_differentiable() -> None:
    matrix = jnp.array([[1.0, 2.0], [3.0, -5.0]])
    estimate, _ = expax.normest.onenormest(block_size=2)

    def objective(matrix):
        return estimate(_dense_matvec, jnp.zeros(2), jax.random.key(0), matrix)

    received = jax.jit(jax.grad(objective))(matrix)
    expected = jnp.array([[0.0, 1.0], [0.0, -1.0]])

    np.testing.assert_array_equal(received, expected)


def test_onenormest_stochastic_estimate_is_differentiable() -> None:
    diagonal = jnp.arange(1.0, 10.0)
    estimate, _ = expax.normest.onenormest(block_size=2)

    def objective(diagonal):
        matrix = jnp.diag(diagonal)
        return estimate(_dense_matvec, jnp.zeros(9), jax.random.key(0), matrix)

    received = jax.jit(jax.grad(objective))(diagonal)
    expected = jax.nn.one_hot(8, 9)

    np.testing.assert_array_equal(received, expected)


@pytest.mark.parametrize("seed", range(4))
def test_onenormest_bounds_complex_nonnormal_operator_above_its_cost(seed: int) -> None:
    matrix = jnp.pad(
        jnp.array(
            [
                [1 + 2j, 8 - 3j, 0, 1j],
                [0, -2j, 5 + 4j, 0],
                [3, 0, -1 + 1j, 6],
                [0, 2 - 1j, 0, 4j],
            ],
            dtype=jnp.complex128,
        ),
        ((0, 5), (0, 5)),
    )
    exact = np.linalg.norm(np.asarray(matrix), ord=1)
    estimate, _ = expax.normest.onenormest(block_size=2)

    received = float(
        estimate(
            _dense_matvec,
            jnp.zeros(9, dtype=jnp.complex128),
            jax.random.key(seed),
            matrix,
        )
    )

    assert exact / 3 <= received <= exact


def test_onenormest_accepts_parameterized_pytree_operators_above_its_cost() -> None:
    v_like = {
        "left": jnp.zeros(5, dtype=jnp.float64),
        "right": jnp.zeros(4, dtype=jnp.float64),
    }
    matrix = jnp.pad(
        jnp.array([[2.0, -1.0, 0.0], [0.0, 3.0, 4.0], [1.0, 0.0, -2.0]]),
        ((0, 6), (0, 6)),
    )

    def matvec(
        x: PyTree[Inexact[Array, "..."]],
        scale: Real[Array, ""],
        matrix: Inexact[Array, "dim dim"],
    ) -> PyTree[Inexact[Array, "..."]]:
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


def test_onenormest_handles_operator_powers_above_its_cost_without_changing_cost() -> (
    None
):
    matrix = jnp.pad(
        jnp.array([[2.0, 1.0, 0.0], [0.0, -1.0, 3.0], [1.0, 0.0, 2.0]]),
        ((0, 6), (0, 6)),
    )
    power = 3

    def powered_matvec(
        x: Inexact[Array, " dim"],
        matrix: Inexact[Array, "dim dim"],
    ) -> Inexact[Array, " dim"]:
        for _ in range(power):
            x = matrix @ x
        return x

    estimate, cost = expax.normest.onenormest(block_size=2)
    received = estimate(powered_matvec, jnp.zeros(9), jax.random.key(5), matrix)

    exact = np.linalg.norm(np.linalg.matrix_power(np.asarray(matrix), power), ord=1)
    assert exact / 3 <= float(received) <= exact
    assert power * cost == 24


def test_estimate_1norm_from_test_vectors_uses_adjoint_before_final_step() -> None:
    adjoint_calls = []

    def record(sign_vectors: Inexact[np.ndarray, "block dim"]) -> None:
        adjoint_calls.append(np.asarray(sign_vectors))

    def apply_adjoint_to_sign_vectors(
        sign_vectors: Inexact[Array, "block dim"],
    ) -> Inexact[Array, "block dim"]:
        jax.debug.callback(record, sign_vectors)
        return sign_vectors

    state = expax.normest._Block1NormEstimatorState(
        test_vectors=jnp.ones((2, 3)),
        estimate=jnp.array(0.0),
        previous_sign_vectors=jnp.zeros((2, 3)),
        test_vector_indices=jnp.zeros((2,), dtype=jnp.int32),
        visited_basis_indices=jnp.zeros((3,), dtype=bool),
        key=jax.random.key(0),
        done=jnp.array(False),
    )
    estimate_before_final_step = jax.jit(
        lambda state: expax.normest._estimate_1norm_from_test_vectors(
            state,
            step=jnp.array(0),
            apply_operator_to_test_vectors=lambda test_vectors: test_vectors,
            apply_adjoint_to_sign_vectors=apply_adjoint_to_sign_vectors,
            max_steps=2,
            check_sign_parallelism=True,
        )
    )

    received = estimate_before_final_step(state)
    jax.block_until_ready(received.estimate)

    assert [call.shape for call in adjoint_calls] == [(2, 3)]


def test_estimate_1norm_from_test_vectors_skips_adjoint_on_final_step() -> None:
    adjoint_calls = []

    def record(sign_vectors: Inexact[np.ndarray, "block dim"]) -> None:
        adjoint_calls.append(np.asarray(sign_vectors))

    def apply_adjoint_to_sign_vectors(
        sign_vectors: Inexact[Array, "block dim"],
    ) -> Inexact[Array, "block dim"]:
        jax.debug.callback(record, sign_vectors)
        return sign_vectors

    state = expax.normest._Block1NormEstimatorState(
        test_vectors=jnp.ones((2, 3)),
        estimate=jnp.array(0.0),
        previous_sign_vectors=jnp.zeros((2, 3)),
        test_vector_indices=jnp.zeros((2,), dtype=jnp.int32),
        visited_basis_indices=jnp.zeros((3,), dtype=bool),
        key=jax.random.key(0),
        done=jnp.array(False),
    )
    estimate_on_final_step = jax.jit(
        lambda state: expax.normest._estimate_1norm_from_test_vectors(
            state,
            step=jnp.array(1),
            apply_operator_to_test_vectors=lambda test_vectors: test_vectors,
            apply_adjoint_to_sign_vectors=apply_adjoint_to_sign_vectors,
            max_steps=2,
            check_sign_parallelism=True,
        )
    )

    received = estimate_on_final_step(state)
    jax.block_until_ready(received.estimate)

    assert not adjoint_calls
    assert bool(received.done)


def test_onenormest_evaluates_below_cost_operators_exactly_under_jit() -> None:
    matrix = jnp.pad(
        jnp.array([[1.0, 4.0, 0.0], [2.0, -5.0, 0.0], [3.0, 2.0, 1.0]]),
        ((0, 4), (0, 4)),
    )
    estimate, _ = expax.normest.onenormest(block_size=2)

    received = jax.jit(lambda key: estimate(_dense_matvec, jnp.zeros(7), key, matrix))(
        jax.random.key(0)
    )

    np.testing.assert_allclose(received, 11.0)


def test_onenormest_evaluates_exactly_when_block_size_equals_dimension() -> None:
    matrix = jnp.array([[1.0, 4.0], [2.0, -5.0]])
    estimate, _ = expax.normest.onenormest(block_size=2)

    received = jax.jit(lambda key: estimate(_dense_matvec, jnp.zeros(2), key, matrix))(
        jax.random.key(0)
    )

    np.testing.assert_allclose(received, 9.0)


def test_onenormest_evaluates_cost_sized_operators_exactly_under_jit() -> None:
    matrix = jnp.pad(
        jnp.array([[1.0, 4.0, 0.0], [2.0, -5.0, 0.0], [3.0, 2.0, 1.0]]),
        ((0, 5), (0, 5)),
    )
    estimate, _ = expax.normest.onenormest(block_size=2)

    received = jax.jit(lambda key: estimate(_dense_matvec, jnp.zeros(8), key, matrix))(
        jax.random.key(0)
    )

    np.testing.assert_allclose(received, 11.0)


def test_onenormest_exact_dispatch_avoids_constructing_an_adjoint() -> None:
    def forward_only_matvec(
        vector: Inexact[Array, " dim"],
    ) -> Inexact[Array, " dim"]:
        return jax.pure_callback(
            lambda value: value,
            jax.ShapeDtypeStruct(vector.shape, vector.dtype),
            vector,
            vmap_method="sequential",
        )

    estimate, _ = expax.normest.onenormest(block_size=2)

    received = jax.jit(lambda key: estimate(forward_only_matvec, jnp.zeros(8), key))(
        jax.random.key(0)
    )

    np.testing.assert_allclose(received, 1.0)
    with pytest.raises(ValueError, match="Pure callbacks do not support transpose"):
        jax.jit(lambda key: estimate(forward_only_matvec, jnp.zeros(9), key))(
            jax.random.key(0)
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block_size": 0}, "block_size"),
        ({"max_steps": 1}, "max_steps"),
    ],
)
def test_onenormest_rejects_invalid_static_configuration(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        expax.normest.onenormest(**kwargs)
