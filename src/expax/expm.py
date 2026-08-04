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
    _make_selector,
)
from expax._taylor import _taylor_action
from expax._theta import _theta
from expax._time_grid import _time_grid_action
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
    algorithm="parallel",
) -> Callable[..., Any]:
    """Construct a matrix-exponential action without materializing a matrix.

    Args:
        matvec: Callable ``matvec(vector, *parameters)`` representing the linear
            operator. Its input and output are PyTrees with the structure described
            by ``v_like``.
        *parameters: Differentiable runtime parameters passed to ``matvec``.
        times: A scalar time or a one-dimensional array of times. Supplying times
            returns the vector action directly. Omitting them returns a reusable
            factory ``times -> action`` that retains the operator-dependent plan.
        v_like: A PyTree of JAX arrays describing the vector-space structure,
            shapes, and common inexact dtype. Its values are ignored.
        key: Random key used by stochastic planning estimators.
        trace_estimator: Callable
            ``estimator(matvec, v_like, key, *parameters) -> trace``. The default
            uses two-sample XTrace when the dimension permits it.
        norm_estimator: Pair ``(estimator, cost)``. The estimator has the same
            calling convention as ``trace_estimator`` and estimates the operator
            1-norm; ``cost`` models its work in scalar ``matvec`` equivalents for
            the planning criterion. ``None`` selects
            :func:`expax.normest.onenormest`, whose ``4 * block_size`` model is
            approximately four batches of ``block_size`` parallel applications,
            and which evaluates every basis vector exactly whenever the dimension
            does not exceed that cost.
        max_degree: Maximum Taylor degree, between 1 and 55.
        max_scaling: Optional upper bound on a selected scaling count. An action
            whose plan exceeds the bound returns NaN leaves without entering its
            scaling loop.
        tol: Positive finite Taylor tolerance. The default is half the machine
            epsilon of the vector-space dtype.
        while_loop: Callable with the
            ``while_loop(cond_fun, body_fun, init_val) -> final_val`` contract.
            It implements runtime-dependent scaling and time-grid loops.
        algorithm: ``"parallel"`` evaluates arbitrary times with synchronized
            lanes. ``"time_grid"`` uses Algorithm 5.2 and requires the supplied
            times to be equally spaced. Invalid grids produce NaN result leaves.

    Returns:
        If ``times`` is supplied, a callable ``vector -> result``. Otherwise, a
        callable ``times -> (vector -> result)``. Scalar-time results have the
        vector's PyTree shapes. Array-time results add a leading time axis to every
        leaf.

    Note:
        Planning is stopped from differentiation. Only the returned Taylor action
        is differentiated. JAX's default dynamic ``while_loop`` supports forward
        mode but not reverse mode; inject a reverse-mode-compatible implementation
        when reverse-mode differentiation is required.
    """
    max_degree, max_scaling = _validate_static_options(
        max_degree, max_scaling, algorithm
    )
    flat_like, _ = _validate_vector_space(v_like)
    if norm_estimator is None:
        norm_estimator = onenormest()
    tolerance = _resolve_tolerance(flat_like.dtype, tol)
    planned_times = None if times is None else _validate_times(times, algorithm)
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
        action_times = _validate_times(action_times, algorithm)
        if algorithm == "time_grid":
            return _time_grid_action(
                action_times,
                matvec,
                parameters,
                mu,
                select,
                tol=tolerance,
                max_degree=max_degree,
                while_loop=while_loop,
            )
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


def _validate_static_options(max_degree, max_scaling, algorithm):
    max_degree = _as_integer(max_degree, "max_degree")
    if not 1 <= max_degree <= 55:
        raise ValueError("max_degree must be an integer between one and 55")
    if max_scaling is not None:
        max_scaling = _as_integer(max_scaling, "max_scaling")
        if max_scaling < 1:
            raise ValueError("max_scaling must be a positive integer or None")
    if algorithm not in ("parallel", "time_grid"):
        raise ValueError("algorithm must be 'parallel' or 'time_grid'")
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


def _validate_times(times, algorithm="parallel"):
    times = jnp.asarray(times)
    if times.ndim > 1:
        raise ValueError("times must be a scalar or one-dimensional array")
    if times.size == 0:
        raise ValueError("times must contain at least one time point")
    if algorithm == "time_grid" and (times.ndim != 1 or times.size < 2):
        raise ValueError("time_grid requires at least two time points")
    return times
