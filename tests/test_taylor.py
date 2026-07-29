import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jax.flatten_util import ravel_pytree

from expax._taylor import _taylor_action


def _dense_matvec(x, matrix):
    return matrix @ x


def _action(
    time,
    matrix,
    *,
    degree=30,
    scaling=1,
    mu=0.0,
    tol=2.0**-53,
    max_degree=30,
):
    return _taylor_action(
        jnp.asarray(time),
        _dense_matvec,
        (matrix,),
        jnp.asarray(mu),
        jnp.asarray(degree),
        jnp.asarray(float(scaling)),
        jnp.asarray(True),
        tol=tol,
        max_degree=max_degree,
        while_loop=jax.lax.while_loop,
        scalar_time=True,
    )


def test_taylor_action_matches_dense_real_exponential():
    matrix = jnp.array([[1.0, 3.0], [-2.0, 0.5]])
    vector = jnp.array([2.0, -1.0])
    time = 0.7

    received = _action(time, matrix)(vector)
    expected = scipy.linalg.expm(time * np.asarray(matrix)) @ np.asarray(vector)

    np.testing.assert_allclose(received, expected, rtol=1e-14, atol=1e-14)


def test_taylor_action_matches_dense_complex_exponential():
    matrix = jnp.array([[1 + 2j, 3 - 1j], [-2j, 0.5 + 0.25j]], dtype=jnp.complex128)
    vector = jnp.array([2 - 1j, -1 + 3j], dtype=jnp.complex128)
    time = -0.4

    received = _action(time, matrix)(vector)
    expected = scipy.linalg.expm(time * np.asarray(matrix)) @ np.asarray(vector)

    np.testing.assert_allclose(received, expected, rtol=1e-14, atol=1e-14)


def test_taylor_action_supports_pytree_vectors_and_trace_shift():
    matrix = jnp.array([[4.0, 1.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, -2.0]])
    vector = {"left": jnp.array([2.0, -1.0]), "right": jnp.array([0.5])}
    _, unravel = ravel_pytree(vector)
    mu = jnp.trace(matrix) / 3

    def matvec(x, matrix):
        flat, _ = ravel_pytree(x)
        return unravel(matrix @ flat)

    action = _taylor_action(
        jnp.array(0.8),
        matvec,
        (matrix,),
        mu,
        jnp.array(30),
        jnp.array(1.0),
        jnp.array(True),
        tol=2.0**-53,
        max_degree=30,
        while_loop=jax.lax.while_loop,
        scalar_time=True,
    )
    received, _ = ravel_pytree(action(vector))
    expected = np.array([42.756249796797463, -1.5509927916598635, 0.1009482589973277])

    np.testing.assert_allclose(received, expected, rtol=1e-14, atol=1e-14)


def test_taylor_action_uses_runtime_scaling_inside_jit():
    matrix = jnp.array([[5.0, 4.0], [-2.0, 1.0]])
    vector = jnp.array([1.0, -3.0])
    time = jnp.array(2.0)

    @jax.jit
    def evaluate(matrix, vector, time, degree, scaling):
        return _taylor_action(
            time,
            _dense_matvec,
            (matrix,),
            jnp.array(0.0),
            degree,
            scaling,
            jnp.array(True),
            tol=2.0**-53,
            max_degree=30,
            while_loop=jax.lax.while_loop,
            scalar_time=True,
        )(vector)

    received = evaluate(matrix, vector, time, jnp.array(30), jnp.array(4.0))
    expected = scipy.linalg.expm(float(time) * np.asarray(matrix)) @ np.asarray(vector)

    np.testing.assert_allclose(received, expected, rtol=1e-13, atol=1e-12)


def test_taylor_action_returns_input_at_zero_time():
    vector = jnp.array([1.0, -2.0])

    received = _action(0.0, jnp.eye(2), degree=0, scaling=1, max_degree=5)(vector)

    np.testing.assert_array_equal(received, vector)


def test_taylor_action_skips_terms_beyond_runtime_degree():
    calls = []

    def record(_):
        calls.append(None)

    def matvec(x, matrix):
        jax.debug.callback(record, x[0])
        return matrix @ x

    @jax.jit
    def evaluate(vector, degree):
        action = _taylor_action(
            jnp.array(0.25),
            matvec,
            (jnp.eye(2),),
            jnp.array(0.0),
            degree,
            jnp.array(1.0),
            jnp.array(True),
            tol=0.0,
            max_degree=20,
            while_loop=jax.lax.while_loop,
            scalar_time=True,
        )
        return action(vector)

    received = evaluate(jnp.array([1.0, 2.0]), jnp.array(3))
    jax.block_until_ready(received)

    assert len(calls) == 3


def test_taylor_action_stops_after_convergence():
    calls = []

    def record(_):
        calls.append(None)

    def matvec(x, matrix):
        jax.debug.callback(record, x[0])
        return matrix @ x

    action = _taylor_action(
        jnp.array(0.25),
        matvec,
        (jnp.eye(2),),
        jnp.array(0.0),
        jnp.array(20),
        jnp.array(1.0),
        jnp.array(True),
        tol=1e6,
        max_degree=20,
        while_loop=jax.lax.while_loop,
        scalar_time=True,
    )
    received = jax.jit(action)(jnp.array([1.0, 2.0]))
    jax.block_until_ready(received)

    assert len(calls) == 1
