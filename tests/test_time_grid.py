import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
from jax.flatten_util import ravel_pytree

import expax


def _dense_matvec(x, matrix):
    return matrix @ x


def _exact_norm_estimator(matvec, v_like, key, *parameters):
    del key
    flat_like, unravel = ravel_pytree(v_like)
    basis = jnp.eye(flat_like.size, dtype=flat_like.dtype)

    def apply(vector):
        image, _ = ravel_pytree(matvec(unravel(vector), *parameters))
        return image

    images = jax.vmap(apply)(basis)
    return jnp.max(jnp.sum(jnp.abs(images), axis=-1))


def _known_trace(_matvec, _v_like, _key, matrix):
    return jnp.trace(matrix)


def test_time_grid_matches_dense_reference():
    matrix = jnp.array([[1.0, 3.0], [-2.0, 0.5]])
    vector = jnp.array([2.0, -1.0])
    times = jnp.linspace(-0.2, 0.8, 7)

    action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        times=times,
        v_like=jnp.zeros_like(vector),
        key=jax.random.key(0),
        trace_estimator=_known_trace,
        norm_estimator=(_exact_norm_estimator, 8),
        algorithm="time_grid",
    )
    received = action(vector)
    expected = np.stack(
        [
            scipy.linalg.expm(float(time) * np.asarray(matrix)) @ np.asarray(vector)
            for time in times
        ]
    )

    np.testing.assert_allclose(received, expected, rtol=1e-13, atol=1e-13)


def test_time_grid_returns_nan_for_nonuniform_runtime_times():
    matrix = jnp.array([[1.0, 3.0], [-2.0, 0.5]])
    vector = jnp.array([2.0, -1.0])

    @jax.jit
    def evaluate(times):
        action = expax.expm_multiply(
            _dense_matvec,
            matrix,
            times=times,
            v_like=jnp.zeros_like(vector),
            key=jax.random.key(0),
            trace_estimator=_known_trace,
            norm_estimator=(_exact_norm_estimator, 8),
            algorithm="time_grid",
        )
        return action(vector)

    received = evaluate(jnp.array([0.0, 0.1, 1.0]))

    assert jnp.all(jnp.isnan(received))


def test_time_grid_returns_nan_for_small_nonuniform_times():
    matrix = jnp.array([[1e16]])
    vector = jnp.ones(1)
    times = jnp.array([0.0, 1e-16, 1e-15])

    action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        times=times,
        v_like=jnp.zeros_like(vector),
        key=jax.random.key(0),
        trace_estimator=_known_trace,
        norm_estimator=(_exact_norm_estimator, 1),
        algorithm="time_grid",
    )
    received = action(vector)

    assert jnp.all(jnp.isnan(received))


@pytest.mark.parametrize(
    "times",
    [
        jnp.array([0.0, jnp.nan, 1.0]),
        jnp.array([0.0, 0.5, jnp.nan]),
    ],
)
def test_time_grid_returns_nan_for_nonfinite_time(times):
    matrix = jnp.array([[1.0, 3.0], [-2.0, 0.5]])
    vector = jnp.array([2.0, -1.0])

    action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        times=times,
        v_like=jnp.zeros_like(vector),
        key=jax.random.key(0),
        trace_estimator=_known_trace,
        norm_estimator=(_exact_norm_estimator, 8),
        algorithm="time_grid",
    )
    received = action(vector)

    assert jnp.all(jnp.isnan(received))


def test_time_grid_returns_nan_when_interval_exceeds_max_scaling():
    matrix = jnp.array([[0.0, 100.0], [-100.0, 0.0]])
    vector = jnp.array([2.0, -1.0])
    times = jnp.linspace(0.0, 1.0, 101)

    action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        times=times,
        v_like=jnp.zeros_like(vector),
        key=jax.random.key(0),
        trace_estimator=_known_trace,
        norm_estimator=(_exact_norm_estimator, 8),
        max_scaling=1,
        algorithm="time_grid",
    )
    received = action(vector)

    assert jnp.all(jnp.isnan(received))


def test_time_grid_uses_sequential_branch_for_large_scaling():
    matrix = jnp.array([[0.0, 100.0], [-100.0, 0.0]])
    vector = jnp.array([2.0, -1.0])
    times = jnp.linspace(0.0, 1.0, 3)

    action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        times=times,
        v_like=jnp.zeros_like(vector),
        key=jax.random.key(0),
        trace_estimator=_known_trace,
        norm_estimator=(_exact_norm_estimator, 8),
        algorithm="time_grid",
    )
    received = action(vector)
    expected = np.stack(
        [
            scipy.linalg.expm(float(time) * np.asarray(matrix)) @ np.asarray(vector)
            for time in times
        ]
    )

    np.testing.assert_allclose(received, expected, rtol=1e-12, atol=1e-12)


def test_time_grid_reuses_taylor_terms():
    matrix = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
    vector = jnp.array([2.0, -1.0])
    times = jnp.linspace(0.0, 1.0, 5)
    parallel_calls = []
    grid_calls = []

    def make_matvec(calls):
        def record(_):
            calls.append(None)

        def matvec(vector, matrix):
            jax.debug.callback(record, vector[0])
            return matrix @ vector

        return matvec

    def norm_estimator(*_):
        return jnp.array(1.0)

    options = {
        "times": times,
        "v_like": jnp.zeros_like(vector),
        "trace_estimator": lambda *_: jnp.array(0.0),
        "norm_estimator": (norm_estimator, 8),
    }
    parallel = expax.expm_multiply(
        make_matvec(parallel_calls),
        matrix,
        key=jax.random.key(0),
        algorithm="parallel",
        **options,
    )
    time_grid = expax.expm_multiply(
        make_matvec(grid_calls),
        matrix,
        key=jax.random.key(1),
        algorithm="time_grid",
        **options,
    )

    parallel_result = jax.jit(parallel)(vector)
    grid_result = jax.jit(time_grid)(vector)
    jax.block_until_ready((parallel_result, grid_result))

    np.testing.assert_allclose(grid_result, parallel_result, rtol=1e-14, atol=1e-14)
    assert len(grid_calls) < len(parallel_calls)


def test_time_grid_supports_pytree_vectors():
    matrix = jnp.array([[1.0, 3.0], [-2.0, 0.5]])
    vector = {"left": jnp.array([2.0]), "right": jnp.array([-1.0])}
    flat_vector, unravel = ravel_pytree(vector)
    times = jnp.linspace(-0.2, 0.8, 7)

    def matvec(x, matrix):
        flat, _ = ravel_pytree(x)
        return unravel(matrix @ flat)

    action = expax.expm_multiply(
        matvec,
        matrix,
        times=times,
        v_like=jax.tree.map(jnp.zeros_like, vector),
        key=jax.random.key(0),
        trace_estimator=_known_trace,
        norm_estimator=(_exact_norm_estimator, 8),
        algorithm="time_grid",
    )
    received = action(vector)
    received_flat = jnp.concatenate(
        (received["left"], received["right"]),
        axis=-1,
    )
    expected = np.stack(
        [
            scipy.linalg.expm(float(time) * np.asarray(matrix))
            @ np.asarray(flat_vector)
            for time in times
        ]
    )

    np.testing.assert_allclose(received_flat, expected, rtol=1e-13, atol=1e-13)


def test_time_grid_supports_forward_mode_differentiation():
    base = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
    vector = jnp.array([2.0, -1.0])
    times = jnp.linspace(0.0, 1.0, 5)

    def evaluate(scale):
        action = expax.expm_multiply(
            _dense_matvec,
            scale * base,
            times=times,
            v_like=jnp.zeros_like(vector),
            key=jax.random.key(0),
            trace_estimator=_known_trace,
            norm_estimator=(_exact_norm_estimator, 8),
            algorithm="time_grid",
        )
        return action(vector)

    scale = jnp.array(0.7)
    received, tangent = jax.jvp(evaluate, (scale,), (jnp.ones_like(scale),))
    expected = np.stack(
        [
            scipy.linalg.expm(float(time * scale) * np.asarray(base))
            @ np.asarray(vector)
            for time in times
        ]
    )
    expected_tangent = np.stack(
        [
            float(time) * np.asarray(base) @ value
            for time, value in zip(times, expected, strict=True)
        ]
    )

    np.testing.assert_allclose(received, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=1e-13, atol=1e-13)
