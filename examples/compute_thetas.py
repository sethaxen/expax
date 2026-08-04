"""Independently derive and compare Al-Mohy--Higham theta values.

The symbolic recurrence is a Python/SymPy port of ``compute_thetas.jl`` in this
directory. Reference comparisons use commit-pinned data from
https://github.com/higham/expmv, distributed there under BSD-2-Clause:
https://github.com/higham/expmv/blob/779f27e81afba16b9b2454ef0460ecf7bad23988/license.txt
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import ml_dtypes
import mpmath as mp
import numpy as np
import sympy as sp

DEFAULT_MAX_DEGREE = 55
DEFAULT_SERIES_DEGREE = 200
DEFAULT_PRECISION_BITS = 512
VERIFY_SERIES_DEGREE = 400
VERIFY_PRECISION_BITS = 1024
AUTHORITATIVE_SERIES_DEGREE = 800
AUTHORITATIVE_PRECISION_BITS = 2048
ROOT_DECIMAL_DIGITS = 170
MPFloat: TypeAlias = Any


def _require_rational(value: object) -> sp.Rational:
    """Narrow SymPy's conservative arithmetic types to exact rationals."""
    if not isinstance(value, sp.Rational):
        raise ArithmeticError(f"expected an exact rational, received {value!r}")
    return value


@dataclass
class TaylorLogState:
    """Exact coefficients of ``exp(-x) * T_m(x)`` through a fixed degree."""

    m: int
    g: list[sp.Rational]

    @classmethod
    def create(cls, series_degree: int) -> TaylorLogState:
        if series_degree < 2:
            raise ValueError("series_degree must be at least two")
        coefficients = [_require_rational(sp.Rational(1))]
        coefficients.extend(
            _require_rational(sp.Rational(-1 if k % 2 else 1, math.factorial(k)))
            for k in range(1, series_degree + 1)
        )
        return cls(m=0, g=coefficients)

    def advance(self) -> None:
        series_degree = len(self.g) - 1
        next_m = self.m + 1
        if next_m >= series_degree:
            raise ValueError("cannot advance beyond series_degree - 1")
        self.m = next_m
        self.g[self.m] = _require_rational(sp.Rational(0))
        denominator_left = math.factorial(self.m)
        for k in range(self.m + 1, series_degree + 1):
            remainder = k - self.m
            sign = -1 if remainder % 2 else 1
            self.g[k] = _require_rational(
                self.g[k]
                + sp.Rational(
                    sign,
                    denominator_left * math.factorial(remainder),
                )
            )


def log_coefficients(state: TaylorLogState) -> tuple[sp.Rational, ...]:
    """Return exact ascending coefficients of ``log(g_m(x))``."""
    series_degree = len(state.g) - 1
    coefficients = [_require_rational(sp.Rational(0))] * (series_degree + 1)
    for n in range(state.m + 1, series_degree + 1):
        convolution = sum(
            (sp.Rational(j) * coefficients[j] * state.g[n - j] for j in range(1, n)),
            start=sp.Rational(0),
        )
        coefficients[n] = _require_rational(state.g[n] - convolution / n)
    return tuple(coefficients)


def build_majorants(
    max_degree: int,
    series_degree: int,
) -> tuple[tuple[sp.Rational, ...], ...]:
    """Build exact ascending coefficients of each ``abs(h_m(x)) / x``."""
    if not 1 <= max_degree < series_degree:
        raise ValueError("max_degree must be positive and below series_degree")
    state = TaylorLogState.create(series_degree)
    majorants = []
    for _ in range(max_degree):
        state.advance()
        coefficients = log_coefficients(state)
        majorants.append(
            tuple(_require_rational(abs(value)) for value in coefficients[1:])
        )
    return tuple(majorants)


@dataclass(frozen=True)
class PrecisionMode:
    name: str
    scalar_type: type[Any]
    tolerance: sp.Rational
    complex_type: str | None


NATIVE_MODES = (
    PrecisionMode(
        "float16", np.float16, _require_rational(sp.Rational(1, 2**11)), None
    ),
    PrecisionMode(
        "bfloat16", ml_dtypes.bfloat16, _require_rational(sp.Rational(1, 2**8)), None
    ),
    PrecisionMode(
        "float32",
        np.float32,
        _require_rational(sp.Rational(1, 2**24)),
        "complex64",
    ),
    PrecisionMode(
        "float64",
        np.float64,
        _require_rational(sp.Rational(1, 2**53)),
        "complex128",
    ),
)
EXPMV_HALF_MODE = PrecisionMode(
    "float16-expmv-tolerance",
    np.float16,
    _require_rational(sp.Rational(1, 2**10)),
    None,
)


def _as_mpf_rational(value: sp.Rational) -> MPFloat:
    return mp.mpf(int(value.p)) / int(value.q)


def _evaluate_majorant(
    x: MPFloat,
    coefficients: Sequence[MPFloat],
) -> MPFloat:
    value = mp.mpf(0)
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def compute_theta_roots(
    tolerance: sp.Rational,
    majorants: Sequence[Sequence[sp.Rational]],
    precision_bits: int,
) -> tuple[MPFloat, ...]:
    """Return safe lower endpoints of high-precision theta root brackets."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if precision_bits < 64:
        raise ValueError("precision_bits must be at least 64")
    roots = []
    with mp.workprec(precision_bits):
        target = _as_mpf_rational(tolerance)
        lower = mp.mpf(0)
        for degree, coefficients in enumerate(majorants, start=1):
            numeric_coefficients = tuple(
                _as_mpf_rational(coefficient) for coefficient in coefficients
            )
            upper = mp.root(target * math.factorial(degree + 1), degree)
            while _evaluate_majorant(upper, numeric_coefficients) <= target:
                upper *= 2
            if _evaluate_majorant(lower, numeric_coefficients) > target:
                raise ArithmeticError("theta sequence lost monotonicity")
            for _ in range(precision_bits + 64):
                midpoint = (lower + upper) / 2
                if midpoint == lower or midpoint == upper:
                    break
                if _evaluate_majorant(midpoint, numeric_coefficients) <= target:
                    lower = midpoint
                else:
                    upper = midpoint
            roots.append(+lower)
    return tuple(roots)


def as_mpf(value: Any) -> MPFloat:
    """Convert a supported target scalar to its exact binary value in mpmath."""
    return mp.mpf(float(value))


def _target_scalar(value: MPFloat, mode: PrecisionMode) -> Any:
    text = mp.nstr(value, n=ROOT_DECIMAL_DIGITS)
    return mode.scalar_type(text)


def _nextafter(value: Any, target: Any, mode: PrecisionMode) -> Any:
    source_array = np.asarray(value, dtype=mode.scalar_type)
    target_array = np.asarray(target, dtype=mode.scalar_type)
    return np.nextafter(source_array, target_array)[()]


def round_down(value: MPFloat, mode: PrecisionMode) -> Any:
    candidate = _target_scalar(value, mode)
    if as_mpf(candidate) > value:
        candidate = _nextafter(candidate, mode.scalar_type(0), mode)
    if not as_mpf(candidate) <= value:
        raise ArithmeticError("target value exceeds high-precision root")
    if not value < as_mpf(next_up(candidate, mode)):
        raise ArithmeticError("target value is not the greatest safe value")
    return candidate


def next_up(value: Any, mode: PrecisionMode) -> Any:
    return _nextafter(value, mode.scalar_type(np.inf), mode)


def _positive_bits(value: Any, mode: PrecisionMode) -> int:
    dtype = np.dtype(mode.scalar_type)
    unsigned_type = {2: np.uint16, 4: np.uint32, 8: np.uint64}[dtype.itemsize]
    return int(np.asarray(value, dtype=dtype).view(unsigned_type).item())


def ulp_distance(left: Any, right: Any, mode: PrecisionMode) -> int:
    if as_mpf(left) < 0 or as_mpf(right) < 0:
        raise ValueError(
            "ULP distance is defined here only for nonnegative theta values"
        )
    return abs(_positive_bits(left, mode) - _positive_bits(right, mode))
