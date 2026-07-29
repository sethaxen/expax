import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from expax._planning import (
    _build_operator_plan,
    _default_trace_estimator,
    _make_selector,
)


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


@pytest.mark.parametrize(("dimension", "expected_calls"), [(3, 2), (4, 2), (5, 4)])
def test_default_trace_estimator_avoids_materialization(dimension, expected_calls):
    calls = []
    matrix = jnp.diag(jnp.arange(1, dimension + 1, dtype=jnp.float64))

    def record(_):
        calls.append(None)

    def matvec(x, matrix):
        jax.debug.callback(record, x[0])
        return matrix @ x

    received = _default_trace_estimator(
        matvec, jnp.zeros(dimension), jax.random.key(0), matrix
    )
    jax.block_until_ready(received)

    assert jnp.isfinite(received)
    assert len(calls) == expected_calls


@pytest.mark.parametrize("dimension", [1, 2])
def test_default_trace_estimator_rejects_too_small_spaces(dimension):
    with pytest.raises(ValueError, match="dimension"):
        _default_trace_estimator(
            _dense_matvec,
            jnp.zeros(dimension),
            jax.random.key(0),
            jnp.eye(dimension),
        )


def test_operator_plan_uses_finite_trace_shift():
    matrix = jnp.array([[4.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, -2.0]])
    theta = jnp.ones(5)

    mu, norm, _, _ = _build_operator_plan(
        None,
        _dense_matvec,
        (matrix,),
        v_like=jnp.zeros(3),
        key=jax.random.key(0),
        trace_estimator=lambda _matvec, _v_like, _key, matrix: jnp.trace(matrix),
        norm_estimator=(_exact_norm_estimator, 8),
        theta=theta,
        max_degree=5,
    )

    expected_mu = np.trace(np.asarray(matrix)) / 3
    expected_norm = np.linalg.norm(np.asarray(matrix) - expected_mu * np.eye(3), ord=1)
    np.testing.assert_allclose(mu, expected_mu)
    np.testing.assert_allclose(norm, expected_norm)


@pytest.mark.parametrize("trace", [jnp.nan, jnp.inf, -jnp.inf])
def test_operator_plan_uses_no_shift_for_nonfinite_trace(trace):
    matrix = jnp.diag(jnp.array([1.0, 2.0, 4.0]))

    mu, norm, _, _ = _build_operator_plan(
        None,
        _dense_matvec,
        (matrix,),
        v_like=jnp.zeros(3),
        key=jax.random.key(0),
        trace_estimator=lambda *_: trace,
        norm_estimator=(_exact_norm_estimator, 8),
        theta=jnp.ones(5),
        max_degree=5,
    )

    np.testing.assert_allclose(mu, 0.0)
    np.testing.assert_allclose(norm, 4.0)


def test_condition_3_13_skips_power_norm_estimates():
    calls = []

    def record(_):
        calls.append(None)

    def norm_estimator(*_):
        jax.debug.callback(record, jnp.array(0))
        return jnp.array(0.01)

    plan = _build_operator_plan(
        jnp.array(1.0),
        _dense_matvec,
        (jnp.eye(3),),
        v_like=jnp.zeros(3),
        key=jax.random.key(0),
        trace_estimator=lambda *_: jnp.array(0.0),
        norm_estimator=(norm_estimator, 8),
        theta=jnp.ones(5),
        max_degree=5,
    )
    jax.block_until_ready(plan)

    assert bool(plan[3])
    assert len(calls) == 1


def test_operator_plan_preserves_single_precision_across_condition():
    matrix = jnp.eye(3, dtype=jnp.float32)

    def norm_estimator(*_):
        return jnp.array(100.0, dtype=jnp.float32)

    _, _, power_matrix, use_norm = _build_operator_plan(
        jnp.array(1.0, dtype=jnp.float32),
        _dense_matvec,
        (matrix,),
        v_like=jnp.zeros(3, dtype=jnp.float32),
        key=jax.random.key(0),
        trace_estimator=lambda *_: jnp.array(0.0, dtype=jnp.float32),
        norm_estimator=(norm_estimator, 8),
        theta=(1.0,) * 5,
        max_degree=5,
    )

    assert not bool(use_norm)
    assert power_matrix.dtype == jnp.float32


def test_deferred_plan_builds_reusable_matrix_once():
    calls = []

    def record(_):
        calls.append(None)

    def norm_estimator(*_):
        jax.debug.callback(record, jnp.array(0))
        return jnp.array(2.0)

    _, norm, matrix, use_norm = _build_operator_plan(
        None,
        _dense_matvec,
        (jnp.eye(3),),
        v_like=jnp.zeros(3),
        key=jax.random.key(0),
        trace_estimator=lambda *_: jnp.array(0.0),
        norm_estimator=(norm_estimator, 8),
        theta=jnp.ones(5),
        max_degree=5,
    )
    jax.block_until_ready(matrix)
    calls_after_planning = len(calls)

    select = _make_selector(norm, matrix, use_norm, theta=jnp.ones(5), max_scaling=None)
    first = select(jnp.array(1.0))
    second = select(jnp.array([0.5, 2.0]))
    jax.block_until_ready((first, second))

    assert calls_after_planning == 4
    assert len(calls) == calls_after_planning
    assert first[0].shape == ()
    assert second[0].shape == (2,)


def test_selector_marks_scalings_above_cap_invalid():
    select = _make_selector(
        jnp.array(10.0),
        jnp.zeros((2, 5)),
        jnp.array(True),
        theta=jnp.ones(5),
        max_scaling=5,
    )

    degree, scaling, valid = select(jnp.array(1.0))

    assert int(degree) == 1
    assert float(scaling) == 10.0
    assert not bool(valid)


def test_operator_planning_stops_parameter_gradients():
    base = jnp.diag(jnp.array([1.0, 2.0, 4.0]))

    def objective(scale):
        matrix = scale * base
        mu, norm, power_matrix, use_norm = _build_operator_plan(
            None,
            _dense_matvec,
            (matrix,),
            v_like=jnp.zeros(3),
            key=jax.random.key(0),
            trace_estimator=lambda _matvec, _v_like, _key, matrix: jnp.trace(matrix),
            norm_estimator=(_exact_norm_estimator, 8),
            theta=jnp.ones(5),
            max_degree=5,
        )
        return jnp.real(mu) + norm + jnp.sum(power_matrix) + use_norm.astype(norm.dtype)

    np.testing.assert_allclose(jax.grad(objective)(jnp.array(2.0)), 0.0)
