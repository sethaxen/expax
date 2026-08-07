"""Generate the theta values imported by ``expax._theta``.

Usage, from the repository root:

    uv run --project tools/generate_theta_values --locked python \\
        tools/generate_theta_values/generate_theta_values.py

The generator builds its coefficients exactly as rational numbers and uses Arb
to enclose the positive roots of absolute-coefficient majorants. It solves a
1,050-term majorant and requires a 1,200-term extension to select the same
downward-rounded binary64 value before emitting it.

Based on the approach used in
Higham NJ, Al-Mohy AH. Computing matrix functions. Acta Numerica.
2010;19:159-208. doi:10.1017/S0962492910000036.
"""

import math
from pathlib import Path

from flint import arb, arb_poly, fmpq, fmpq_series
from flint import ctx as flint_ctx

MAX_DEGREE = 55
SERIES_DEGREE = 1050
VERIFICATION_SERIES_DEGREE = 1200
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


def build_majorants() -> tuple[tuple[arb_poly, ...], tuple[arb_poly, ...]]:
    """Build primary and extended majorants from one exact rational series."""
    old_cap = flint_ctx.cap
    old_prec = flint_ctx.prec
    flint_ctx.cap = VERIFICATION_SERIES_DEGREE + 1
    flint_ctx.prec = ROOT_ACCURACY_BITS + ARB_GUARD_BITS
    try:
        factorials = [math.factorial(k) for k in range(VERIFICATION_SERIES_DEGREE + 1)]
        coefficients = [fmpq(1)]
        coefficients.extend(
            fmpq(-1 if k % 2 else 1, factorials[k])
            for k in range(1, VERIFICATION_SERIES_DEGREE + 1)
        )
        majorants = []
        verification_majorants = []
        for degree in range(1, MAX_DEGREE + 1):
            coefficients[degree] = fmpq(0)
            denominator_left = factorials[degree]
            for k in range(degree + 1, VERIFICATION_SERIES_DEGREE + 1):
                remainder = k - degree
                coefficients[k] += fmpq(
                    -1 if remainder % 2 else 1,
                    denominator_left * factorials[remainder],
                )
            logarithm = fmpq_series(
                coefficients,
                prec=VERIFICATION_SERIES_DEGREE + 1,
            ).log()
            absolute_coefficients = [abs(value) for value in logarithm.coeffs()[1:]]
            majorants.append(arb_poly(absolute_coefficients[:SERIES_DEGREE]))
            verification_majorants.append(arb_poly(absolute_coefficients))
        return tuple(majorants), tuple(verification_majorants)
    finally:
        flint_ctx.prec = old_prec
        flint_ctx.cap = old_cap


def _classify_enclosure(value: arb, tolerance: fmpq) -> bool:
    """Return whether an Arb enclosure is rigorously at or below tolerance."""
    target = arb(tolerance)
    if value.upper() <= target:
        return True
    if value.lower() > target:
        return False
    raise ArithmeticError("unable to classify Arb polynomial enclosure")


def bracket_theta_values(
    majorants: tuple[arb_poly, ...], tolerance: fmpq
) -> tuple[tuple[fmpq, fmpq], ...]:
    """Enclose every positive majorant root in a sufficiently narrow interval."""
    old_prec = flint_ctx.prec
    flint_ctx.prec = ROOT_ACCURACY_BITS + ARB_GUARD_BITS
    try:
        brackets = []
        for majorant in majorants:
            lower = fmpq(0)
            upper = fmpq(1)
            while _classify_enclosure(majorant(arb(upper)), tolerance):
                upper *= 2
            while upper - lower > lower * fmpq(1, 2**ROOT_ACCURACY_BITS):
                midpoint = (lower + upper) / 2
                if _classify_enclosure(majorant(arb(midpoint)), tolerance):
                    lower = midpoint
                else:
                    upper = midpoint
            brackets.append((lower, upper))
        return tuple(brackets)
    finally:
        flint_ctx.prec = old_prec


def _as_fmpq(value: float) -> fmpq:
    numerator, denominator = value.as_integer_ratio()
    return fmpq(numerator, denominator)


def binary64_below(lower: fmpq, upper: fmpq) -> float:
    candidate = float(lower)
    if _as_fmpq(candidate) > lower:
        candidate = math.nextafter(candidate, 0.0)
    successor = math.nextafter(candidate, math.inf)
    if not _as_fmpq(candidate) <= lower < upper < _as_fmpq(successor):
        raise ArithmeticError("root bracket does not determine one binary64 value")
    return candidate


def verify_binary64_convergence(
    values: tuple[float, ...],
    majorants: tuple[arb_poly, ...],
    tolerance: fmpq,
) -> None:
    """Require the extended majorants to select the same binary64 values."""
    old_prec = flint_ctx.prec
    flint_ctx.prec = ROOT_ACCURACY_BITS + ARB_GUARD_BITS
    try:
        for value, majorant in zip(values, majorants, strict=True):
            lower = _as_fmpq(value)
            upper = _as_fmpq(math.nextafter(value, math.inf))
            if not _classify_enclosure(majorant(arb(lower)), tolerance):
                raise ArithmeticError("theta values did not converge")
            if _classify_enclosure(majorant(arb(upper)), tolerance):
                raise ArithmeticError("theta values did not converge")
    finally:
        flint_ctx.prec = old_prec


def render_module(values: dict[float, tuple[float, ...]]) -> str:
    """Render the generated table with exact binary64 spellings."""
    lines = [
        '"""Generated theta values; do not edit this file by hand.',
        "",
        "Generate from the repository root with:",
        "uv run --project tools/generate_theta_values --locked python \\",
        "    tools/generate_theta_values/generate_theta_values.py",
        "",
        "Higham NJ, Al-Mohy AH. Computing matrix functions. Acta Numerica.",
        "2010;19:159-208. doi:10.1017/S0962492910000036.",
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
    majorants, verification_majorants = build_majorants()
    values = {}
    for exponent in TOLERANCE_EXPONENTS:
        tolerance = fmpq(1, 2**exponent)
        brackets = bracket_theta_values(majorants, tolerance)
        candidates = tuple(binary64_below(lower, upper) for lower, upper in brackets)
        verify_binary64_convergence(candidates, verification_majorants, tolerance)
        values[2.0**-exponent] = candidates
    rendered = render_module(values)
    if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(encoding="utf-8") != rendered:
        OUTPUT_PATH.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
