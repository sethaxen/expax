import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
from jax.flatten_util import ravel_pytree

import expax


class _IndexableBoolean:
    dtype = np.dtype(bool)

    def __index__(self):
        return 1


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


def test_expm_multiply_known_time_matches_dense_reference():
    matrix = jnp.array([[4.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, -2.0]])
    vector = jnp.array([2.0, -1.0, 0.5])
    time = jnp.array(0.8)

    action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        times=time,
        v_like=jnp.zeros_like(vector),
        key=jax.random.key(0),
        trace_estimator=_known_trace,
        norm_estimator=(_exact_norm_estimator, 8),
    )
    received = action(vector)
    expected = scipy.linalg.expm(float(time) * np.asarray(matrix)) @ np.asarray(vector)

    np.testing.assert_allclose(received, expected, rtol=1e-13, atol=1e-13)


def test_expm_multiply_parallel_times_have_leading_time_axis():
    matrix = jnp.array([[1.0, 3.0], [-2.0, 0.5]])
    vector = jnp.array([2.0, -1.0])
    times = jnp.array([0.0, 0.7, -0.4])

    action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        times=times,
        v_like=jnp.zeros_like(vector),
        key=jax.random.key(0),
        trace_estimator=_known_trace,
        norm_estimator=(_exact_norm_estimator, 8),
    )
    received = action(vector)
    expected = np.stack(
        [
            scipy.linalg.expm(float(time) * np.asarray(matrix)) @ np.asarray(vector)
            for time in times
        ]
    )

    assert received.shape == (3, 2)
    np.testing.assert_allclose(received, expected, rtol=1e-13, atol=1e-13)


def test_expm_multiply_actions_vmap_over_vectors():
    matrix = jnp.array([[1.0, 3.0], [-2.0, 0.5]])
    vectors = jnp.array([[2.0, -1.0], [0.5, 3.0], [-2.0, 4.0]])
    times = jnp.array([0.0, 0.4])

    action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        times=times,
        v_like=jnp.zeros(2),
        key=jax.random.key(0),
        trace_estimator=_known_trace,
        norm_estimator=(_exact_norm_estimator, 8),
    )
    received = jax.vmap(action)(vectors)
    expected = np.stack(
        [
            np.stack(
                [
                    scipy.linalg.expm(float(time) * np.asarray(matrix))
                    @ np.asarray(vector)
                    for time in times
                ]
            )
            for vector in vectors
        ]
    )

    assert received.shape == (3, 2, 2)
    np.testing.assert_allclose(received, expected, rtol=1e-13, atol=1e-13)


def test_expm_multiply_uses_default_norm_estimator_for_none():
    matrix = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    vector = jnp.array([2.0, -1.0, 0.5, 3.0, -2.0])
    time = jnp.array(0.2)

    action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        times=time,
        v_like=jnp.zeros_like(vector),
        key=jax.random.key(1),
        trace_estimator=_known_trace,
        norm_estimator=None,
    )
    received = action(vector)
    expected = scipy.linalg.expm(float(time) * np.asarray(matrix)) @ np.asarray(vector)

    np.testing.assert_allclose(received, expected, rtol=1e-13, atol=1e-13)


def test_expm_multiply_none_uses_onenormest_for_small_vector_spaces():
    matrix = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    vector = jnp.ones(2)
    options = {
        "times": jnp.array(22.0),
        "v_like": jnp.zeros(2),
        "key": jax.random.key(1),
        "max_scaling": 1,
    }

    default_action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        norm_estimator=None,
        **options,
    )
    explicit_action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        norm_estimator=expax.normest.onenormest(),
        **options,
    )

    np.testing.assert_array_equal(
        jnp.isnan(default_action(vector)),
        jnp.isnan(explicit_action(vector)),
    )


@pytest.mark.parametrize("dimension", [1, 2])
def test_expm_multiply_defaults_support_small_vector_spaces(dimension):
    matrix = jnp.diag(jnp.arange(1, dimension + 1, dtype=jnp.float64))
    vector = jnp.arange(1, dimension + 1, dtype=jnp.float64)
    time = jnp.array(0.2)

    action = expax.expm_multiply(
        _dense_matvec,
        matrix,
        times=time,
        v_like=jnp.zeros_like(vector),
        key=jax.random.key(1),
    )
    received = action(vector)
    expected = scipy.linalg.expm(float(time) * np.asarray(matrix)) @ np.asarray(vector)

    np.testing.assert_allclose(received, expected, rtol=1e-13, atol=1e-13)


def test_expm_multiply_deferred_time_matches_known_time():
    matrix = jnp.array([[2.0, -1.0, 0.0], [0.0, 3.0, 4.0], [1.0, 0.0, -2.0]])
    vector = jnp.array([1.0, 2.0, -1.0])
    time = jnp.array(-0.3)
    options = {
        "v_like": jnp.zeros_like(vector),
        "key": jax.random.key(2),
        "trace_estimator": _known_trace,
        "norm_estimator": (_exact_norm_estimator, 8),
    }

    known_action = expax.expm_multiply(_dense_matvec, matrix, times=time, **options)
    at_times = expax.expm_multiply(_dense_matvec, matrix, **options)

    np.testing.assert_allclose(
        at_times(time)(vector),
        known_action(vector),
        rtol=1e-14,
        atol=1e-14,
    )


def test_expm_multiply_ignores_v_like_values():
    matrix = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    vector = jnp.array([2.0, -1.0, 0.5])
    options = {
        "times": jnp.array(0.4),
        "key": jax.random.key(4),
        "trace_estimator": _known_trace,
        "norm_estimator": (_exact_norm_estimator, 8),
    }

    from_zeros = expax.expm_multiply(
        _dense_matvec, matrix, v_like=jnp.zeros(3), **options
    )(vector)
    from_values = expax.expm_multiply(
        _dense_matvec, matrix, v_like=jnp.array([8.0, -2.0, 5.0]), **options
    )(vector)

    np.testing.assert_array_equal(from_zeros, from_values)


def test_expm_multiply_plans_with_runtime_parameters_inside_jit():
    vector = jnp.array([1.0, -2.0, 0.5])
    time = jnp.array(0.2)

    @jax.jit
    def evaluate(matrix, vector, key):
        action = expax.expm_multiply(
            _dense_matvec,
            matrix,
            times=time,
            v_like=vector,
            key=key,
            trace_estimator=_known_trace,
            norm_estimator=(_exact_norm_estimator, 8),
        )
        return action(vector)

    first_matrix = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    second_matrix = jnp.array([[0.0, 1.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    first = evaluate(first_matrix, vector, jax.random.key(0))
    second = evaluate(second_matrix, vector, jax.random.key(1))

    first_expected = scipy.linalg.expm(
        float(time) * np.asarray(first_matrix)
    ) @ np.asarray(vector)
    second_expected = scipy.linalg.expm(
        float(time) * np.asarray(second_matrix)
    ) @ np.asarray(vector)
    np.testing.assert_allclose(first, first_expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(second, second_expected, rtol=1e-14, atol=1e-14)


def test_deferred_time_selection_reuses_operator_plan():
    estimator_calls = []

    def record(_):
        estimator_calls.append(None)

    def norm_estimator(*_):
        jax.debug.callback(record, jnp.array(0))
        return jnp.array(1.0)

    at_times = expax.expm_multiply(
        _dense_matvec,
        jnp.eye(3),
        v_like=jnp.zeros(3),
        key=jax.random.key(0),
        trace_estimator=lambda *_: jnp.array(0.0),
        norm_estimator=(norm_estimator, 8),
        max_degree=5,
    )
    jax.block_until_ready(at_times(jnp.array(0.5))(jnp.ones(3)))
    calls_after_first = len(estimator_calls)
    jax.block_until_ready(at_times(jnp.array(1.0))(jnp.ones(3)))

    assert calls_after_first == 4
    assert len(estimator_calls) == calls_after_first


@pytest.mark.parametrize(
    "max_degree",
    [0, 56, 1.5, True, np.bool_(True), jnp.array(True), _IndexableBoolean()],
)
def test_expm_multiply_rejects_invalid_max_degree(max_degree):
    with pytest.raises(ValueError, match="max_degree"):
        expax.expm_multiply(
            _dense_matvec,
            jnp.eye(3),
            times=jnp.array(0.5),
            v_like=jnp.zeros(3),
            key=jax.random.key(0),
            trace_estimator=_known_trace,
            norm_estimator=(_exact_norm_estimator, 8),
            max_degree=max_degree,
        )


@pytest.mark.parametrize(
    "max_scaling",
    [0, -1, 1.5, True, np.bool_(True), jnp.array(True), _IndexableBoolean()],
)
def test_expm_multiply_rejects_invalid_max_scaling(max_scaling):
    with pytest.raises(ValueError, match="max_scaling"):
        expax.expm_multiply(
            _dense_matvec,
            jnp.eye(3),
            times=jnp.array(0.5),
            v_like=jnp.zeros(3),
            key=jax.random.key(0),
            trace_estimator=_known_trace,
            norm_estimator=(_exact_norm_estimator, 8),
            max_scaling=max_scaling,
        )


def test_expm_multiply_accepts_numpy_integer_options():
    action = expax.expm_multiply(
        _dense_matvec,
        jnp.eye(3),
        times=jnp.array(0.5),
        v_like=jnp.zeros(3),
        key=jax.random.key(0),
        trace_estimator=_known_trace,
        norm_estimator=(_exact_norm_estimator, 8),
        max_degree=np.int64(5),
        max_scaling=np.int64(10),
    )

    np.testing.assert_allclose(action(jnp.ones(3)), jnp.exp(0.5) * jnp.ones(3))


@pytest.mark.parametrize("tol", [0.0, -1.0, jnp.inf, jnp.nan])
def test_expm_multiply_rejects_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tol"):
        expax.expm_multiply(
            _dense_matvec,
            jnp.eye(3),
            times=jnp.array(0.5),
            v_like=jnp.zeros(3),
            key=jax.random.key(0),
            trace_estimator=_known_trace,
            norm_estimator=(_exact_norm_estimator, 8),
            tol=tol,
        )


@pytest.mark.parametrize("times", [jnp.ones((2, 2)), jnp.array([])])
def test_expm_multiply_rejects_invalid_time_shapes(times):
    with pytest.raises(ValueError, match="times"):
        expax.expm_multiply(
            _dense_matvec,
            jnp.eye(3),
            times=times,
            v_like=jnp.zeros(3),
            key=jax.random.key(0),
            trace_estimator=_known_trace,
            norm_estimator=(_exact_norm_estimator, 8),
        )


def test_expm_multiply_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="algorithm"):
        expax.expm_multiply(
            _dense_matvec,
            jnp.eye(3),
            times=jnp.array(0.5),
            v_like=jnp.zeros(3),
            key=jax.random.key(0),
            trace_estimator=_known_trace,
            norm_estimator=(_exact_norm_estimator, 8),
            algorithm="sequential",
        )


@pytest.mark.parametrize("times", [jnp.array(0.5), jnp.array([0.5])])
def test_expm_multiply_time_grid_requires_an_interval(times):
    with pytest.raises(ValueError, match="time_grid"):
        expax.expm_multiply(
            _dense_matvec,
            jnp.eye(3),
            times=times,
            v_like=jnp.zeros(3),
            key=jax.random.key(0),
            trace_estimator=_known_trace,
            norm_estimator=(_exact_norm_estimator, 8),
            algorithm="time_grid",
        )


def test_expm_multiply_validates_time_grid_before_planning():
    def fail_if_called(*_):
        raise AssertionError("planning should not run")

    with pytest.raises(ValueError, match="time_grid"):
        expax.expm_multiply(
            _dense_matvec,
            jnp.eye(3),
            times=jnp.array(0.5),
            v_like=jnp.zeros(3),
            key=jax.random.key(0),
            trace_estimator=fail_if_called,
            norm_estimator=(fail_if_called, 8),
            algorithm="time_grid",
        )


def test_expm_multiply_skips_actions_above_max_scaling():
    calls = []

    def record(_):
        calls.append(None)

    def matvec(vector, matrix):
        jax.debug.callback(record, vector[0])
        return matrix @ vector

    def norm_estimator(*_):
        return jnp.array(1.0)

    action = expax.expm_multiply(
        matvec,
        jnp.eye(3),
        times=jnp.array(100.0),
        v_like=jnp.zeros(3),
        key=jax.random.key(0),
        trace_estimator=lambda *_: jnp.array(0.0),
        norm_estimator=(norm_estimator, 8),
        max_scaling=1,
    )
    received = jax.jit(action)(jnp.ones(3))
    jax.block_until_ready(received)

    assert jnp.all(jnp.isnan(received))
    assert not calls


def test_expm_multiply_uses_injected_while_loop():
    loop_calls = []

    def while_loop(cond_fun, body_fun, init_val):
        loop_calls.append(None)
        return jax.lax.while_loop(cond_fun, body_fun, init_val)

    action = expax.expm_multiply(
        _dense_matvec,
        jnp.eye(3),
        times=jnp.array(0.5),
        v_like=jnp.zeros(3),
        key=jax.random.key(0),
        trace_estimator=_known_trace,
        norm_estimator=(_exact_norm_estimator, 8),
        while_loop=while_loop,
    )
    received = action(jnp.ones(3))

    np.testing.assert_allclose(received, jnp.exp(0.5) * jnp.ones(3))
    assert len(loop_calls) == 1
