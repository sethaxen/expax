"""Generate the theta values imported by ``expax._theta``.

Usage, from the repository root:

    uv run --project tools/generate_theta_values --locked python \\
        tools/generate_theta_values/generate_theta_values.py

For the degree-m Taylor polynomial ``T_m(x)``, define the backward-error series

    h_m(x) = log(exp(-x) T_m(x)) = sum(c_k x**k, k=m+1,...)

and its absolute-coefficient majorant

    h_tilde_m(x) = sum(abs(c_k) x**k, k=m+1,...).

The generated threshold is the largest theta for which

    h_tilde_m(theta) / theta <= tolerance.

This is the Taylor specialization in Appendix A, especially equation (A.3), of
the backward-error analysis in Section 5 of

Higham NJ, Al-Mohy AH. Computing matrix functions. Acta Numerica. 2010;19:159-208.
doi: 10.1017/S0962492910000036.
eprint: https://eprints.maths.manchester.ac.uk/1406/.

The paper uses unit roundoff as the tolerance. This script applies the same
construction to every tolerance supported by Expax. It builds ``h_m`` exactly
over the rationals, encloses each theta with Arb, and rounds the result downward
to binary64. A longer series must select the same binary64 value before the table
is written.
"""

import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from itertools import accumulate
from operator import mul
from pathlib import Path

import flint
from flint import arb, arb_poly, fmpq, fmpq_series

MAX_TAYLOR_DEGREE = 55
BACKWARD_ERROR_TRUNCATION = 1050
BACKWARD_ERROR_VERIFICATION_TRUNCATION = 1200
ROOT_ACCURACY_BITS = 128
ARB_GUARD_BITS = 128
TOLERANCE_EXPONENTS = (11, 8, 24, 53)
DTYPES_BY_TOLERANCE_EXPONENT = {
    11: "float16",
    8: "bfloat16",
    24: "float32 / complex64",
    53: "float64 / complex128",
}
OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "expax" / "_generated_theta_values.py"
)


@contextmanager
def _workcap(cap: int) -> Iterator[None]:
    """Temporarily set FLINT's global power-series precision."""
    old_cap = flint.ctx.cap
    flint.ctx.cap = cap
    try:
        yield
    finally:
        flint.ctx.cap = old_cap


# Approximation-specific layer: derive h_m for the exponential Taylor polynomial.


def exponential_taylor_backward_error_series(
    *, max_degree: int, series_degree: int
) -> Iterator[fmpq_series]:
    """Yield the exact series ``h_m = log(exp(-x) T_m(x))``.

    This is ``h_m`` from Appendix A immediately before equation (A.3). The
    recurrence below avoids multiplying two long series afresh for every degree:

        exp(-x) T_m(x)
            = exp(-x) T_(m-1)(x) + x**m exp(-x) / m!.

    A different approximation or matrix function can reuse the rest of this
    script by supplying its own exact backward-error series in place of these.
    """
    series_precision = series_degree + 1
    factorials = list(accumulate(range(1, series_precision), mul, initial=1))
    exp_minus_x = [
        fmpq(-1 if power % 2 else 1, factorials[power])
        for power in range(series_precision)
    ]

    # At the start of degree m, this contains exp(-x) T_(m-1)(x).
    scaled_approximant = exp_minus_x.copy()
    for degree in range(1, max_degree + 1):
        degree_factorial = factorials[degree]
        for power in range(degree, series_precision):
            remainder = power - degree
            scaled_approximant[power] += fmpq(
                -1 if remainder % 2 else 1,
                degree_factorial * factorials[remainder],
            )

        with _workcap(series_precision):
            backward_error = fmpq_series(
                scaled_approximant,
                prec=series_precision,
            ).log()
        yield backward_error


# Generic layer: turn exact h_m coefficients into certified theta values.


def relative_backward_error_majorant(
    backward_error: fmpq_series, *, max_power: int
) -> arb_poly:
    """Return the truncated majorant ``h_tilde_m(x) / x``.

    Dividing by ``x`` makes theta the positive solution of
    ``majorant(theta) = tolerance``, exactly as in equation (A.3).
    """
    if backward_error.prec <= max_power:
        raise ValueError("backward-error series is shorter than max_power")
    return arb_poly([abs(backward_error[power]) for power in range(1, max_power + 1)])


def build_relative_backward_error_majorants(
    backward_errors: Iterable[fmpq_series],
    *,
    max_power: int,
    extended_max_power: int,
) -> tuple[tuple[arb_poly, ...], tuple[arb_poly, ...]]:
    """Build primary and extended majorants while consuming each series once."""
    with flint.ctx.workprec(ROOT_ACCURACY_BITS + ARB_GUARD_BITS):
        majorants = []
        extended_majorants = []
        for backward_error in backward_errors:
            majorants.append(
                relative_backward_error_majorant(
                    backward_error,
                    max_power=max_power,
                )
            )
            extended_majorants.append(
                relative_backward_error_majorant(
                    backward_error,
                    max_power=extended_max_power,
                )
            )
        return tuple(majorants), tuple(extended_majorants)


def _enclosure_is_at_most(value: arb, tolerance: fmpq) -> bool:
    """Classify an Arb enclosure without making an uncertified comparison."""
    target = arb(tolerance)
    if value.upper() <= target:
        return True
    if value.lower() > target:
        return False
    raise ArithmeticError("unable to classify Arb polynomial enclosure")


def bracket_theta(majorant: arb_poly, tolerance: fmpq) -> tuple[fmpq, fmpq]:
    """Enclose the positive solution of ``majorant(theta) = tolerance``."""
    lower = fmpq(0)
    upper = fmpq(1)
    while _enclosure_is_at_most(majorant(arb(upper)), tolerance):
        upper *= 2
    while upper - lower > lower * fmpq(1, 2**ROOT_ACCURACY_BITS):
        midpoint = (lower + upper) / 2
        if _enclosure_is_at_most(majorant(arb(midpoint)), tolerance):
            lower = midpoint
        else:
            upper = midpoint
    return lower, upper


def _as_fmpq(value: float) -> fmpq:
    numerator, denominator = value.as_integer_ratio()
    return fmpq(numerator, denominator)


def downward_binary64(lower: fmpq, upper: fmpq) -> float:
    """Return the greatest binary64 value proved to lie below the root."""
    candidate = float(lower)
    if _as_fmpq(candidate) > lower:
        candidate = math.nextafter(candidate, 0.0)
    successor = math.nextafter(candidate, math.inf)
    if not _as_fmpq(candidate) <= lower < upper < _as_fmpq(successor):
        raise ArithmeticError("root bracket does not determine one binary64 value")
    return candidate


def verify_extended_majorants(
    values: tuple[float, ...],
    majorants: tuple[arb_poly, ...],
    tolerance: fmpq,
) -> None:
    """Require longer majorants to select the same downward binary64 values."""
    for value, majorant in zip(values, majorants, strict=True):
        lower = _as_fmpq(value)
        upper = _as_fmpq(math.nextafter(value, math.inf))
        if not _enclosure_is_at_most(majorant(arb(lower)), tolerance):
            raise ArithmeticError("theta values did not converge")
        if _enclosure_is_at_most(majorant(arb(upper)), tolerance):
            raise ArithmeticError("theta values did not converge")


def compute_theta_values(
    majorants: tuple[arb_poly, ...],
    extended_majorants: tuple[arb_poly, ...],
    tolerance: fmpq,
) -> tuple[float, ...]:
    """Solve, round, and verify theta for every supplied approximation degree."""
    with flint.ctx.workprec(ROOT_ACCURACY_BITS + ARB_GUARD_BITS):
        brackets = tuple(bracket_theta(majorant, tolerance) for majorant in majorants)
        values = tuple(downward_binary64(*bracket) for bracket in brackets)
        verify_extended_majorants(values, extended_majorants, tolerance)
        return values


# Output-specific layer: render the table consumed by Expax at runtime.


def render_module(values: dict[float, tuple[float, ...]]) -> str:
    """Render the generated table with exact binary64 decimal spellings."""
    lines = [
        '"""Generated theta values; do not edit this file by hand.',
        "",
        "Generate from the repository root with:",
        "uv run --project tools/generate_theta_values --locked python \\",
        "    tools/generate_theta_values/generate_theta_values.py",
        "",
        "Higham NJ, Al-Mohy AH. Computing matrix functions. Acta Numerica.",
        "2010;19:159-208. doi:10.1017/S0962492910000036.",
        "Taylor construction: Appendix A, equation (A.3).",
        '"""',
        "",
        "_THETA_VALUES_BY_TOLERANCE: dict[float, tuple[float, ...]] = {",
    ]
    for exponent in TOLERANCE_EXPONENTS:
        dtype_names = DTYPES_BY_TOLERANCE_EXPONENT[exponent]
        lines.append(f"    2.0**-{exponent}: (  # {dtype_names}")
        lines.extend(f"        {value!r}," for value in values[2.0**-exponent])
        lines.append("    ),")
    lines.extend(("}", ""))
    return "\n".join(lines)


def main() -> None:
    majorants, extended_majorants = build_relative_backward_error_majorants(
        exponential_taylor_backward_error_series(
            max_degree=MAX_TAYLOR_DEGREE,
            series_degree=BACKWARD_ERROR_VERIFICATION_TRUNCATION,
        ),
        max_power=BACKWARD_ERROR_TRUNCATION,
        extended_max_power=BACKWARD_ERROR_VERIFICATION_TRUNCATION,
    )

    values = {}
    for exponent in TOLERANCE_EXPONENTS:
        tolerance = fmpq(1, 2**exponent)
        values[2.0**-exponent] = compute_theta_values(
            majorants,
            extended_majorants,
            tolerance,
        )

    rendered = render_module(values)
    if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
        OUTPUT_PATH.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
