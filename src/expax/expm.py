"""Factories for matrix-exponential actions."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from expax._operator import _validate_vector_space
from expax._planning import (
    _build_operator_plan,
    _default_trace_estimator,
    _make_selector,
)
from expax._taylor import _taylor_action
from expax._theta import _theta
from expax.normest import onenormest


def expm_multiply(
    matvec,
    *parameters,
    times=None,
    v_like,
    key,
    trace_estimator=_default_trace_estimator,
    norm_estimator=None,
    max_degree=55,
    max_scaling=None,
    tol=None,
    while_loop=jax.lax.while_loop,
) -> Callable[..., Any]:
    """Construct an action of ``exp(t A)`` on one PyTree-valued vector.

    Supplying ``times`` returns the vector action directly. Omitting it returns
    a reusable factory that accepts times while retaining the operator-dependent
    plan.
    """
    flat_like, _ = _validate_vector_space(v_like)
    if norm_estimator is None:
        norm_estimator = onenormest()
    tolerance = _resolve_tolerance(flat_like.dtype, tol)
    theta = _theta(flat_like.dtype, tolerance, max_degree)
    mu, norm, matrix, use_norm = _build_operator_plan(
        times,
        matvec,
        parameters,
        v_like=v_like,
        key=key,
        trace_estimator=trace_estimator,
        norm_estimator=norm_estimator,
        theta=theta,
        max_degree=max_degree,
    )
    select = _make_selector(
        norm,
        matrix,
        use_norm,
        theta=theta,
        max_scaling=max_scaling,
    )

    def at_times(action_times):
        action_times = jnp.asarray(action_times)
        scalar_time = action_times.ndim == 0
        degrees, scalings, valid = select(action_times)
        return _taylor_action(
            action_times,
            matvec,
            parameters,
            mu,
            degrees,
            scalings,
            valid,
            tol=tolerance,
            max_degree=max_degree,
            while_loop=while_loop,
            scalar_time=scalar_time,
        )

    if times is None:
        return at_times
    return at_times(times)


def _resolve_tolerance(dtype, tol):
    if tol is not None:
        return float(tol)
    real_dtype = jnp.real(jnp.zeros((), dtype=dtype)).dtype
    return float(jnp.finfo(real_dtype).eps / 2)
