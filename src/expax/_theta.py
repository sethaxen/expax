import math
from decimal import Decimal, localcontext
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


@cache
def _compute_theta(degree, tol):
    limit = max(200, 4 * degree + 80)
    with localcontext() as context:
        context.prec = 100
        factorials = [Decimal(1)]
        for k in range(1, limit + 1):
            factorials.append(factorials[-1] * k)

        product_coefficients = [Decimal(0)] * (limit + 1)
        product_coefficients[0] = Decimal(1)
        for n in range(degree + 1, limit + 1):
            product_coefficients[n] = sum(
                (
                    (Decimal(-1) if (n - j) % 2 else Decimal(1))
                    / (factorials[n - j] * factorials[j])
                    for j in range(degree + 1)
                ),
                start=Decimal(0),
            )

        log_coefficients = [Decimal(0)] * (limit + 1)
        for n in range(degree + 1, limit + 1):
            convolution = sum(
                Decimal(k) * log_coefficients[k] * product_coefficients[n - k]
                for k in range(degree + 1, n)
            )
            log_coefficients[n] = product_coefficients[n] - convolution / Decimal(n)

        target = Decimal.from_float(float(tol))

        def backward_error_bound(x):
            value = Decimal(0)
            for n in range(limit, degree, -1):
                value = value * x + abs(log_coefficients[n])
            return value * x**degree

        lower = Decimal(0)
        upper = Decimal(1)
        while backward_error_bound(upper) < target:
            upper *= 2
        for _ in range(200):
            midpoint = (lower + upper) / 2
            if backward_error_bound(midpoint) <= target:
                lower = midpoint
            else:
                upper = midpoint

        candidate = float(lower)
        if Decimal.from_float(candidate) > lower:
            candidate = math.nextafter(candidate, 0.0)
        return candidate


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
    dtype = jnp.dtype(dtype)
    try:
        real_dtype = _REAL_DTYPE_BY_DTYPE[dtype]
    except KeyError:
        raise TypeError(f"unsupported vector-space dtype: {dtype}") from None

    tolerance = (
        _DEFAULT_TOLERANCE_BY_REAL_DTYPE[real_dtype] if tol is None else float(tol)
    )
    values = _THETA_VALUES_BY_TOLERANCE.get(tolerance)
    if values is None:
        values = tuple(_compute_theta(m, tolerance) for m in range(1, max_degree + 1))
    return tuple(_project_down(value, real_dtype) for value in values[:max_degree])
