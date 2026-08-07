import jax.numpy as jnp
import numpy as np

from ._generated_theta_values import _THETA_VALUES_BY_TOLERANCE


def _real_dtype(dtype):
    return jnp.dtype(dtype).type(0).real.dtype


def _unit_roundoff(dtype):
    real_dtype = _real_dtype(dtype)
    return np.finfo(real_dtype).eps / 2


def _select_tolerance(tol: float):
    tol_min = min(_THETA_VALUES_BY_TOLERANCE.keys())
    if tol < tol_min:
        raise ValueError(
            f"requested tolerance {tol} is below the minimum supported tolerance: "
            f"{tol_min}"
        )
    return next(
        table_tolerance
        for table_tolerance in sorted(_THETA_VALUES_BY_TOLERANCE.keys(), reverse=True)
        if table_tolerance <= tol
    )


def _project_down(vals, dtype):
    vals = np.asarray(vals)
    vals_dtype = vals.astype(dtype)
    return np.where(
        vals_dtype > vals,
        np.nextafter(vals_dtype, np.array(-np.inf, dtype=dtype)),
        vals_dtype,
    )


def _theta(dtype, tol: float, max_degree: int):
    real_dtype = _real_dtype(dtype)
    tolerance = _select_tolerance(tol)
    values = _THETA_VALUES_BY_TOLERANCE[tolerance]
    return _project_down(values[:max_degree], real_dtype)
