from functools import cache

import jax.numpy as jnp
import numpy as np

from ._generated_theta_values import _THETA_VALUES_BY_TOLERANCE

_REAL_DTYPE_BY_DTYPE = {
    jnp.dtype(jnp.float16): np.dtype(jnp.float16),
    jnp.dtype(jnp.bfloat16): np.dtype(jnp.bfloat16),
    jnp.dtype(jnp.float32): np.dtype(jnp.float32),
    jnp.dtype(jnp.float64): np.dtype(jnp.float64),
    jnp.dtype(jnp.complex64): np.dtype(jnp.float32),
    jnp.dtype(jnp.complex128): np.dtype(jnp.float64),
}

_DEFAULT_TOLERANCE_BY_REAL_DTYPE = {
    np.dtype(jnp.float16): 2.0**-11,
    np.dtype(jnp.bfloat16): 2.0**-8,
    np.dtype(jnp.float32): 2.0**-24,
    np.dtype(jnp.float64): 2.0**-53,
}
_SUPPORTED_TOLERANCES = tuple(sorted(_THETA_VALUES_BY_TOLERANCE))


def _real_dtype(dtype):
    dtype = jnp.dtype(dtype)
    if dtype in _REAL_DTYPE_BY_DTYPE:
        return _REAL_DTYPE_BY_DTYPE[dtype]
    if dtype.kind == "c":
        return np.dtype(f"float{4 * np.dtype(dtype).itemsize}")
    if dtype.kind == "f":
        return np.dtype(dtype)
    else:
        raise TypeError(f"unsupported vector-space dtype: {dtype}") from None


def _dtype_tolerance(dtype):
    real_dtype = _real_dtype(dtype)
    if real_dtype in _DEFAULT_TOLERANCE_BY_REAL_DTYPE:
        return _DEFAULT_TOLERANCE_BY_REAL_DTYPE[real_dtype]
    return float(np.finfo(real_dtype).eps / 2)


def _select_tolerance(dtype, tol):
    dtype_tolerance = _dtype_tolerance(dtype)
    requested_tolerance = dtype_tolerance if tol is None else float(tol)
    effective_tolerance = max(requested_tolerance, dtype_tolerance)
    if effective_tolerance < _SUPPORTED_TOLERANCES[0]:
        raise ValueError(
            "requested tolerance is below the minimum supported tolerance: "
            f"{_SUPPORTED_TOLERANCES[0]}"
        )
    return next(
        table_tolerance
        for table_tolerance in reversed(_SUPPORTED_TOLERANCES)
        if table_tolerance <= effective_tolerance
    )


def _project_down(value, real_dtype):
    candidate = np.asarray(value, dtype=real_dtype)[()]
    if float(candidate) > value:
        candidate = np.nextafter(
            candidate,
            np.asarray(0, dtype=real_dtype)[()],
        )
    return float(candidate)


@cache
def _theta(dtype, tol, max_degree):
    real_dtype = _real_dtype(dtype)
    tolerance = _select_tolerance(dtype, tol)
    values = _THETA_VALUES_BY_TOLERANCE[tolerance]
    return tuple(_project_down(value, real_dtype) for value in values[:max_degree])
