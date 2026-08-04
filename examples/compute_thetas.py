"""Independently derive and compare Al-Mohy--Higham theta values.

The independent symbolic recurrence is a Python/SymPy port of
``compute_thetas.jl`` in this directory. Full tables use exact FLINT rational
power series and rigorous Arb root enclosures. Reference comparisons use
commit-pinned data from
https://github.com/higham/expmv, distributed there under BSD-2-Clause:
https://github.com/higham/expmv/blob/779f27e81afba16b9b2454ef0460ecf7bad23988/license.txt
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Any, Literal, TypeAlias
from urllib.request import urlopen

import ml_dtypes
import mpmath as mp
import numpy as np
import numpy.typing as npt
import scipy.io
import sympy as sp
from flint import arb, arb_poly, fmpq, fmpq_series
from flint import ctx as flint_ctx

DEFAULT_MAX_DEGREE = 55
DEFAULT_SERIES_DEGREE = 200
VERIFY_SERIES_DEGREE = 1050
AUTHORITATIVE_SERIES_DEGREE = 1200
ROOT_ACCURACY_BITS = 128
MIN_ROOT_ACCURACY_BITS = 64
ARB_GUARD_BITS = 128
COMPARISON_PRECISION_BITS = 256
TARGET_CONVERSION_DIGITS = 40
MPFloat: TypeAlias = Any
FloatValues: TypeAlias = Sequence[float] | npt.NDArray[np.float64]
EXPMV_COMMIT = "779f27e81afba16b9b2454ef0460ecf7bad23988"
EXPMV_REPOSITORY = "https://github.com/higham/expmv"
EXPMV_LICENSE = f"{EXPMV_REPOSITORY}/blob/{EXPMV_COMMIT}/license.txt"
EXPMV_RAW_ROOT = f"https://raw.githubusercontent.com/higham/expmv/{EXPMV_COMMIT}"


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


def build_flint_majorants(
    max_degree: int,
    series_degree: int,
) -> tuple[tuple[Any, ...], ...]:
    """Build the majorants exactly using FLINT rational power series."""
    if not 1 <= max_degree < series_degree:
        raise ValueError("max_degree must be positive and below series_degree")

    old_cap = flint_ctx.cap
    flint_ctx.cap = series_degree + 1
    try:
        factorials = [math.factorial(k) for k in range(series_degree + 1)]
        coefficients = [fmpq(1)]
        coefficients.extend(
            fmpq(-1 if k % 2 else 1, factorials[k]) for k in range(1, series_degree + 1)
        )
        majorants = []
        for degree in range(1, max_degree + 1):
            coefficients[degree] = fmpq(0)
            denominator_left = factorials[degree]
            for k in range(degree + 1, series_degree + 1):
                remainder = k - degree
                sign = -1 if remainder % 2 else 1
                coefficients[k] += fmpq(
                    sign,
                    denominator_left * factorials[remainder],
                )
            logarithm = fmpq_series(
                coefficients,
                prec=series_degree + 1,
            ).log()
            log_coefficients = logarithm.coeffs()
            log_coefficients.extend(
                [fmpq(0)] * (series_degree + 1 - len(log_coefficients))
            )
            majorants.append(tuple(abs(value) for value in log_coefficients[1:]))
    finally:
        flint_ctx.cap = old_cap
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


def _as_mpf_rational(
    value: sp.Rational,
    *,
    rounding: Literal["d", "n", "u"] = "n",
) -> MPFloat:
    return mp.fdiv(int(value.p), int(value.q), rounding=rounding)


def _evaluate_majorant(
    x: MPFloat,
    coefficients: Sequence[MPFloat],
    *,
    rounding: Literal["d", "n", "u"] = "n",
) -> MPFloat:
    value = mp.mpf(0)
    for coefficient in reversed(coefficients):
        product = mp.fmul(value, x, rounding=rounding)
        value = mp.fadd(product, coefficient, rounding=rounding)
    return value


def _majorant_definitely_below(
    value: MPFloat,
    upper_coefficients: Sequence[MPFloat],
    target_lower: MPFloat,
) -> bool:
    upper_value = _evaluate_majorant(value, upper_coefficients, rounding="u")
    return bool(upper_value <= target_lower)


def _majorant_definitely_above(
    value: MPFloat,
    lower_coefficients: Sequence[MPFloat],
    target_upper: MPFloat,
) -> bool:
    lower_value = _evaluate_majorant(value, lower_coefficients, rounding="d")
    return bool(lower_value > target_upper)


def compute_theta_roots_reference(
    tolerance: sp.Rational,
    majorants: Sequence[Sequence[sp.Rational]],
    precision_bits: int,
) -> tuple[MPFloat, ...]:
    """Return safe roots using the independent directed-mpmath solver."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if precision_bits < 64:
        raise ValueError("precision_bits must be at least 64")
    roots = []
    with mp.workprec(precision_bits):
        target_lower = _as_mpf_rational(tolerance, rounding="d")
        target_upper = _as_mpf_rational(tolerance, rounding="u")
        lower = mp.mpf(0)
        for degree, coefficients in enumerate(majorants, start=1):
            lower_coefficients = tuple(
                _as_mpf_rational(coefficient, rounding="d")
                for coefficient in coefficients
            )
            upper_coefficients = tuple(
                _as_mpf_rational(coefficient, rounding="u")
                for coefficient in coefficients
            )
            upper = mp.root(target_lower * math.factorial(degree + 1), degree)
            while not _majorant_definitely_above(
                upper, lower_coefficients, target_upper
            ):
                upper *= 2
            if not _majorant_definitely_below(lower, upper_coefficients, target_lower):
                lower = mp.mpf(0)
            for _ in range(precision_bits + 64):
                midpoint = (lower + upper) / 2
                if midpoint == lower or midpoint == upper:
                    break
                if _majorant_definitely_below(
                    midpoint, upper_coefficients, target_lower
                ):
                    lower = midpoint
                elif _majorant_definitely_above(
                    midpoint, lower_coefficients, target_upper
                ):
                    upper = midpoint
                else:
                    break
            roots.append(+lower)
    return tuple(roots)


def _as_fmpq_rational(value: Any) -> Any:
    """Convert a SymPy or FLINT rational to an exact FLINT rational."""
    if isinstance(value, fmpq):
        return value
    return fmpq(int(value.p), int(value.q))


def _as_mpf_fmpq(value: Any) -> MPFloat:
    """Convert a FLINT rational to an mpmath value rounded downward."""
    return mp.fdiv(int(value.p), int(value.q), rounding="d")


def _arb_polynomials(
    majorants: Sequence[Sequence[Any]],
) -> tuple[Any, ...]:
    return tuple(
        arb_poly([_as_fmpq_rational(value) for value in coefficients])
        for coefficients in majorants
    )


def _compute_theta_roots_from_arb(
    tolerance: sp.Rational,
    polynomials: Sequence[Any],
    accuracy_bits: int,
) -> tuple[MPFloat, ...]:
    target = arb(_as_fmpq_rational(tolerance))
    target_lower = target.lower()
    target_upper = target.upper()
    lower = fmpq(0)
    exact_roots = []
    relative_tolerance = fmpq(1, 2**accuracy_bits)

    def classify(polynomial: Any, point: Any) -> int:
        value = polynomial(arb(point))
        if value.upper() <= target_lower:
            return -1
        if value.lower() > target_upper:
            return 1
        raise ArithmeticError(
            "Arb enclosure cannot classify a root bracket endpoint; "
            "increase precision_bits"
        )

    for polynomial in polynomials:
        upper = fmpq(1)
        while classify(polynomial, upper) < 0:
            upper *= 2
        if classify(polynomial, lower) > 0:
            lower = fmpq(0)
        while lower == 0 or upper - lower > lower * relative_tolerance:
            midpoint = (lower + upper) / 2
            if classify(polynomial, midpoint) < 0:
                lower = midpoint
            else:
                upper = midpoint
        exact_roots.append(lower)

    with mp.workprec(accuracy_bits + ARB_GUARD_BITS):
        return tuple(+_as_mpf_fmpq(value) for value in exact_roots)


def compute_theta_roots(
    tolerance: sp.Rational,
    majorants: Sequence[Sequence[Any]],
    accuracy_bits: int,
) -> tuple[MPFloat, ...]:
    """Return root lower bounds with the requested relative accuracy."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if accuracy_bits < MIN_ROOT_ACCURACY_BITS:
        raise ValueError(f"accuracy_bits must be at least {MIN_ROOT_ACCURACY_BITS}")

    old_precision = flint_ctx.prec
    flint_ctx.prec = accuracy_bits + ARB_GUARD_BITS
    try:
        return _compute_theta_roots_from_arb(
            tolerance,
            _arb_polynomials(majorants),
            accuracy_bits,
        )
    finally:
        flint_ctx.prec = old_precision


def as_mpf(value: Any) -> MPFloat:
    """Convert a supported target scalar to its exact binary value in mpmath."""
    return mp.mpf(float(value))


def _target_scalar(value: MPFloat, mode: PrecisionMode) -> Any:
    text = mp.nstr(value, n=TARGET_CONVERSION_DIGITS)
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


@dataclass(frozen=True)
class UpstreamSource:
    filename: str
    sha256: str
    expected_length: int

    @property
    def url(self) -> str:
        return f"{EXPMV_RAW_ROOT}/{self.filename}"


UPSTREAM_SOURCES = {
    "half": UpstreamSource(
        "theta_taylor_half.mat",
        "1d4940fda288f74f9f5a1dde7d6186990ab3157feddf0d8a0218973bab39adfd",
        100,
    ),
    "single": UpstreamSource(
        "theta_taylor_single.mat",
        "e1fb7a81fb68fbbb08fa0f7668a014f4c973ff8ffdad1470b632578137a0723e",
        60,
    ),
    "double": UpstreamSource(
        "theta_taylor.mat",
        "29d37b1c66d424f911b8c821190d3345ad89570c0a006a7ecb472d46e248fcac",
        100,
    ),
}


def _download(url: str) -> bytes:
    with urlopen(url, timeout=30) as response:
        return response.read()


def load_upstream(
    source: UpstreamSource,
    *,
    fetch: Callable[[str], bytes] | None = None,
) -> npt.NDArray[np.float64]:
    payload = (fetch or _download)(source.url)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != source.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {source.filename}: {digest} != {source.sha256}"
        )
    contents = scipy.io.loadmat(io.BytesIO(payload))
    if "theta" not in contents:
        raise ValueError(f"{source.filename} has no theta variable")
    raw_theta = np.asarray(contents["theta"])
    expected_shape = (1, source.expected_length)
    if raw_theta.shape != expected_shape:
        raise ValueError(
            f"expected shape {expected_shape} in {source.filename}, "
            f"received {raw_theta.shape}"
        )
    theta = np.asarray(raw_theta, dtype=np.float64).ravel()
    if not np.all(np.isfinite(theta)) or not np.all(theta > 0):
        raise ValueError(f"{source.filename} theta values must be finite positive")
    return theta


@dataclass(frozen=True)
class ComparisonRow:
    degree: int
    root: MPFloat
    target: Any
    upstream: float | None
    upstream_target: Any | None
    relative_difference: MPFloat | None
    ulp_distance: int | None


@dataclass(frozen=True)
class Comparison:
    label: str
    mode: PrecisionMode
    tolerance_match: bool | None
    rows: tuple[ComparisonRow, ...]
    exact_binary64_matches: int | None
    target_matches: int | None
    common_significant_digits: int | None

    @property
    def upstream_available(self) -> bool:
        return all(row.upstream is not None for row in self.rows)


def _round_significant(value: Decimal, digits: int) -> Decimal:
    quantum = Decimal(1).scaleb(value.adjusted() - digits + 1)
    return value.quantize(quantum, rounding=ROUND_HALF_EVEN)


def _decimal_from_mpf(value: MPFloat, digits: int) -> Decimal:
    text = mp.nstr(value, n=digits)
    if not isinstance(text, str):
        raise ArithmeticError(f"expected scalar decimal text, received {text!r}")
    return Decimal(text)


def maximum_common_significant_digits(
    roots: Sequence[MPFloat],
    upstream: FloatValues,
) -> int | None:
    if len(roots) != len(upstream):
        raise ValueError("computed and upstream tables must have equal lengths")
    with localcontext() as context:
        context.prec = 200
        computed_decimal = [_decimal_from_mpf(value, 180) for value in roots]
        upstream_decimal = [Decimal.from_float(float(value)) for value in upstream]
        matching = [
            digits
            for digits in range(1, 18)
            if all(
                _round_significant(left, digits) == _round_significant(right, digits)
                for left, right in zip(
                    computed_decimal,
                    upstream_decimal,
                    strict=True,
                )
            )
        ]
    return max(matching, default=None)


def compare_values(
    label: str,
    mode: PrecisionMode,
    roots: Sequence[MPFloat],
    upstream: FloatValues | None,
    *,
    tolerance_match: bool | None,
) -> Comparison:
    if upstream is not None and len(upstream) < len(roots):
        raise ValueError("upstream table must contain at least as many values as roots")
    upstream_prefix = None if upstream is None else upstream[: len(roots)]
    rows = []
    for degree, root in enumerate(roots, start=1):
        target = round_down(root, mode)
        reference = (
            None if upstream_prefix is None else float(upstream_prefix[degree - 1])
        )
        reference_target = None
        relative = None
        distance = None
        if reference is not None:
            with mp.workprec(COMPARISON_PRECISION_BITS):
                reference_mpf = mp.mpf(reference)
                relative = +(abs(root - reference_mpf) / abs(reference_mpf))
            reference_target = round_down(reference_mpf, mode)
            distance = ulp_distance(target, reference_target, mode)
        rows.append(
            ComparisonRow(
                degree,
                root,
                target,
                reference,
                reference_target,
                relative,
                distance,
            )
        )
    if upstream_prefix is None:
        exact_matches = target_matches = common_digits = None
    else:
        exact_matches = sum(
            float(root) == float(reference)
            for root, reference in zip(roots, upstream_prefix, strict=True)
        )
        target_matches = sum(row.ulp_distance == 0 for row in rows)
        common_digits = maximum_common_significant_digits(roots, upstream_prefix)
    return Comparison(
        label,
        mode,
        tolerance_match,
        tuple(rows),
        exact_matches,
        target_matches,
        common_digits,
    )


def _format_mpf(value: MPFloat | None, digits: int = 17) -> str:
    if value is None:
        return "N/A"
    text = mp.nstr(value, n=digits)
    if not isinstance(text, str):
        raise ArithmeticError(f"expected scalar decimal text, received {text!r}")
    return text


def verify_root_convergence(
    coarse: dict[str, tuple[MPFloat, ...]],
    fine: dict[str, tuple[MPFloat, ...]],
    modes: Sequence[PrecisionMode],
) -> None:
    with mp.workprec(COMPARISON_PRECISION_BITS):
        for mode in modes:
            for degree, (left, right) in enumerate(
                zip(coarse[mode.name], fine[mode.name], strict=True),
                start=1,
            ):
                relative = +(abs(left - right) / abs(right))
                decimal_projections_match = all(
                    _round_significant(_decimal_from_mpf(left, 180), digits)
                    == _round_significant(_decimal_from_mpf(right, 180), digits)
                    for digits in range(1, 18)
                )
                converged = (
                    relative <= mp.mpf("1e-20")
                    and float(left) == float(right)
                    and round_down(left, mode) == round_down(right, mode)
                    and decimal_projections_match
                    and _format_mpf(left) == _format_mpf(right)
                )
                if not converged:
                    raise ArithmeticError(
                        f"roots did not converge for {mode.name} degree {degree}: "
                        f"relative difference {_format_mpf(relative, 8)}"
                    )


def _compute_modes(
    modes: Sequence[PrecisionMode],
    *,
    series_degree: int,
    accuracy_bits: int,
) -> dict[str, tuple[MPFloat, ...]]:
    majorants = build_flint_majorants(DEFAULT_MAX_DEGREE, series_degree)
    return _compute_modes_from_majorants(modes, majorants, accuracy_bits)


def _compute_modes_from_majorants(
    modes: Sequence[PrecisionMode],
    majorants: Sequence[Sequence[Any]],
    accuracy_bits: int,
) -> dict[str, tuple[MPFloat, ...]]:
    if accuracy_bits < MIN_ROOT_ACCURACY_BITS:
        raise ValueError(f"accuracy_bits must be at least {MIN_ROOT_ACCURACY_BITS}")
    old_precision = flint_ctx.prec
    flint_ctx.prec = accuracy_bits + ARB_GUARD_BITS
    try:
        polynomials = _arb_polynomials(majorants)
        return {
            mode.name: _compute_theta_roots_from_arb(
                mode.tolerance,
                polynomials,
                accuracy_bits,
            )
            for mode in modes
        }
    finally:
        flint_ctx.prec = old_precision


def _majorant_prefixes(
    majorants: Sequence[Sequence[Any]],
    series_degree: int,
) -> tuple[tuple[Any, ...], ...]:
    if series_degree < 1 or any(
        len(coefficients) < series_degree for coefficients in majorants
    ):
        raise ValueError("series_degree must be a positive available prefix")
    return tuple(tuple(coefficients[:series_degree]) for coefficients in majorants)


def _make_comparisons(
    roots: dict[str, tuple[MPFloat, ...]],
    upstream: dict[str, npt.NDArray[np.float64]],
) -> tuple[Comparison, ...]:
    modes = (*NATIVE_MODES, EXPMV_HALF_MODE)
    mode_by_name = {mode.name: mode for mode in modes}
    return (
        compare_values(
            "float16 native tolerance versus expmv half",
            mode_by_name["float16"],
            roots["float16"],
            upstream["half"],
            tolerance_match=False,
        ),
        compare_values(
            "float16 matched expmv tolerance",
            EXPMV_HALF_MODE,
            roots[EXPMV_HALF_MODE.name],
            upstream["half"],
            tolerance_match=True,
        ),
        compare_values(
            "bfloat16 native tolerance",
            mode_by_name["bfloat16"],
            roots["bfloat16"],
            None,
            tolerance_match=None,
        ),
        compare_values(
            "float32/complex64 versus expmv single",
            mode_by_name["float32"],
            roots["float32"],
            upstream["single"],
            tolerance_match=True,
        ),
        compare_values(
            "float64/complex128 versus expmv double",
            mode_by_name["float64"],
            roots["float64"],
            upstream["double"],
            tolerance_match=True,
        ),
    )


def run_experiment(
    *,
    verify: bool = False,
    fetch: Callable[[str], bytes] | None = None,
) -> tuple[Comparison, ...]:
    modes = (*NATIVE_MODES, EXPMV_HALF_MODE)
    upstream = {
        name: load_upstream(source, fetch=fetch)
        for name, source in UPSTREAM_SOURCES.items()
    }
    if verify:
        authoritative_majorants = build_flint_majorants(
            DEFAULT_MAX_DEGREE, AUTHORITATIVE_SERIES_DEGREE
        )
        checked_majorants = _majorant_prefixes(
            authoritative_majorants, VERIFY_SERIES_DEGREE
        )
        checked = _compute_modes_from_majorants(
            modes,
            checked_majorants,
            ROOT_ACCURACY_BITS,
        )
        roots = _compute_modes_from_majorants(
            modes,
            authoritative_majorants,
            ROOT_ACCURACY_BITS,
        )
        verify_root_convergence(checked, roots, modes)
    else:
        checked = None
        roots = _compute_modes(
            modes,
            series_degree=DEFAULT_SERIES_DEGREE,
            accuracy_bits=ROOT_ACCURACY_BITS,
        )
    comparisons = _make_comparisons(roots, upstream)
    if checked is not None:
        checked_comparisons = _make_comparisons(checked, upstream)
        verify_report_stability(checked_comparisons, comparisons)
    return comparisons


def _format_scalar(value: Any | None) -> str:
    return "N/A" if value is None else repr(float(value))


def _format_optional_integer(value: int | None) -> str:
    return "N/A" if value is None else str(value)


def _format_tolerance_match(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "yes" if value else "no"


def _maximum_row(
    comparison: Comparison,
    attribute: str,
) -> tuple[str, str]:
    available = [row for row in comparison.rows if getattr(row, attribute) is not None]
    if not available:
        return "N/A", "N/A"
    row = max(available, key=lambda candidate: getattr(candidate, attribute))
    value = getattr(row, attribute)
    return _format_mpf(value) if isinstance(value, mp.mpf) else str(value), str(
        row.degree
    )


def render_report(comparisons: Sequence[Comparison]) -> str:
    lines = [
        "# Theta comparison",
        "",
        f"Source: [{EXPMV_REPOSITORY}]({EXPMV_REPOSITORY}) at `{EXPMV_COMMIT}`.",
        f"Upstream license: [BSD-2-Clause]({EXPMV_LICENSE}).",
        "Complex reuse: `complex64 -> float32`; `complex128 -> float64`.",
        "",
        "## Summary",
        "",
        "| Mode | Tolerances match | Binary64 matches | Target matches | "
        "Common digits | Max relative (degree) | Max ULP (degree) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in comparisons:
        max_relative, relative_degree = _maximum_row(comparison, "relative_difference")
        max_ulp, ulp_degree = _maximum_row(comparison, "ulp_distance")
        exact_matches = _format_optional_integer(comparison.exact_binary64_matches)
        target_matches = _format_optional_integer(comparison.target_matches)
        common_digits = _format_optional_integer(comparison.common_significant_digits)
        lines.append(
            f"| {comparison.label} | "
            f"{_format_tolerance_match(comparison.tolerance_match)} "
            f"| {exact_matches} | {target_matches} | {common_digits} "
            f"| {max_relative} ({relative_degree}) | {max_ulp} ({ulp_degree}) |"
        )
    for comparison in comparisons:
        lines.extend(
            [
                "",
                f"## {comparison.label}",
                "",
                "| Degree | Computed root | Computed target | Upstream binary64 | "
                "Upstream target | Relative difference | ULP distance |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in comparison.rows:
            lines.append(
                f"| {row.degree} | {_format_mpf(row.root)} | "
                f"{_format_scalar(row.target)} | "
                f"{repr(row.upstream) if row.upstream is not None else 'N/A'} | "
                f"{_format_scalar(row.upstream_target)} | "
                f"{_format_mpf(row.relative_difference)} | "
                f"{row.ulp_distance if row.ulp_distance is not None else 'N/A'} |"
            )
    return "\n".join(lines)


def verify_report_stability(
    coarse: Sequence[Comparison],
    fine: Sequence[Comparison],
) -> None:
    if render_report(coarse) != render_report(fine):
        raise ArithmeticError("comparison report metrics did not converge")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify 1050-term roots against authoritative 1200-term roots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the Markdown report to this path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = render_report(run_experiment(verify=args.verify))
    if args.output is not None:
        args.output.write_text(f"{report}\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
