"""Independently derive and compare Al-Mohy--Higham theta values.

The symbolic recurrence is a Python/SymPy port of ``compute_thetas.jl`` in this
directory. Reference comparisons use commit-pinned data from
https://github.com/higham/expmv, distributed there under BSD-2-Clause:
https://github.com/higham/expmv/blob/779f27e81afba16b9b2454ef0460ecf7bad23988/license.txt
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp

DEFAULT_MAX_DEGREE = 55
DEFAULT_SERIES_DEGREE = 200
DEFAULT_PRECISION_BITS = 512
VERIFY_SERIES_DEGREE = 400
VERIFY_PRECISION_BITS = 1024
AUTHORITATIVE_SERIES_DEGREE = 800
AUTHORITATIVE_PRECISION_BITS = 2048


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
