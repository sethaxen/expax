"""Factories for matrix-exponential actions."""

import math
from collections.abc import Callable
from operator import index
from typing import Any, SupportsIndex

import jax
import jax.numpy as jnp

from expax._operator import _validate_vector_space
from expax._planning import (
    _build_operator_plan,
    _default_trace_estimator,
    _exact_1_norm_estimator,
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
    max_degree: SupportsIndex = 55,
    max_scaling: SupportsIndex | None = None,
    tol=None,
    while_loop=jax.lax.while_loop,
) -> Callable[..., Any]:
    """Construct an action of ``exp(t A)`` on one PyTree-valued vector.

    Supplying ``times`` returns the vector action directly. Omitting it returns
    a reusable factory that accepts times while retaining the operator-dependent
    plan.
    """
    max_degree, max_scaling = _validate_static_options(max_degree, max_scaling)
    flat_like, _ = _validate_vector_space(v_like)
    if norm_estimator is None:
        norm_estimator = (
            (_exact_1_norm_estimator, flat_like.size)
            if flat_like.size < 3
            else onenormest()
        )
    tolerance = _resolve_tolerance(flat_like.dtype, tol)
    planned_times = None if times is None else _validate_times(times)
    theta = _theta(flat_like.dtype, tolerance, max_degree)
    mu, norm, matrix, use_norm = _build_operator_plan(
        planned_times,
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
        action_times = _validate_times(action_times)
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
    return at_times(planned_times)


def _resolve_tolerance(dtype, tol):
    if tol is None:
        real_dtype = jnp.real(jnp.zeros((), dtype=dtype)).dtype
        return float(jnp.finfo(real_dtype).eps / 2)
    tolerance = float(tol)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tol must be finite and positive")
    return tolerance


def _validate_static_options(max_degree, max_scaling):
    max_degree = _as_integer(max_degree, "max_degree")
    if not 1 <= max_degree <= 55:
        raise ValueError("max_degree must be an integer between one and 55")
    if max_scaling is not None:
        max_scaling = _as_integer(max_scaling, "max_scaling")
        if max_scaling < 1:
            raise ValueError("max_scaling must be a positive integer or None")
    return max_degree, max_scaling


def _as_integer(value, name):
    dtype = getattr(value, "dtype", None)
    if isinstance(value, bool) or (
        dtype is not None and jnp.issubdtype(dtype, jnp.bool_)
    ):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(index(value))
    except TypeError:
        raise ValueError(f"{name} must be an integer") from None


def _validate_times(times):
    times = jnp.asarray(times)
    if times.ndim > 1:
        raise ValueError("times must be a scalar or one-dimensional array")
    if times.size == 0:
        raise ValueError("times must contain at least one time point")
    return times
