import importlib.util
import sys
from itertools import pairwise
from pathlib import Path

import mpmath as mp
import pytest
import sympy as sp

MODULE_PATH = Path(__file__).parents[1] / "examples" / "compute_thetas.py"
SPEC = importlib.util.spec_from_file_location("compute_thetas", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compute_thetas = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compute_thetas
SPEC.loader.exec_module(compute_thetas)


def test_degree_one_majorant_coefficients_are_exact():
    (majorant,) = compute_thetas.build_majorants(1, 5)

    assert majorant == (
        sp.Rational(0),
        sp.Rational(1, 2),
        sp.Rational(1, 3),
        sp.Rational(1, 4),
        sp.Rational(1, 5),
    )


def test_precision_modes_have_exact_tolerances_and_complex_reuse():
    modes = {mode.name: mode for mode in compute_thetas.NATIVE_MODES}

    assert modes["float16"].tolerance == sp.Rational(1, 2**11)
    assert modes["bfloat16"].tolerance == sp.Rational(1, 2**8)
    assert modes["float32"].tolerance == sp.Rational(1, 2**24)
    assert modes["float64"].tolerance == sp.Rational(1, 2**53)
    assert modes["float32"].complex_type == "complex64"
    assert modes["float64"].complex_type == "complex128"
    assert modes["float16"].complex_type is None
    assert modes["bfloat16"].complex_type is None
    assert compute_thetas.EXPMV_HALF_MODE.tolerance == sp.Rational(1, 2**10)


@pytest.mark.parametrize(
    "mode", compute_thetas.NATIVE_MODES, ids=lambda mode: mode.name
)
def test_round_down_brackets_high_precision_value(mode):
    value = mp.mpf("1.1")

    rounded = compute_thetas.round_down(value, mode)

    assert compute_thetas.as_mpf(rounded) <= value
    assert value < compute_thetas.as_mpf(compute_thetas.next_up(rounded, mode))
    assert (
        compute_thetas.ulp_distance(
            rounded,
            compute_thetas.next_up(rounded, mode),
            mode,
        )
        == 1
    )


def test_computed_roots_are_positive_monotone_and_stable_when_rounded():
    mode = next(mode for mode in compute_thetas.NATIVE_MODES if mode.name == "float32")
    ordinary = compute_thetas.compute_theta_roots(
        mode.tolerance,
        compute_thetas.build_majorants(6, 40),
        128,
    )
    checked = compute_thetas.compute_theta_roots(
        mode.tolerance,
        compute_thetas.build_majorants(6, 80),
        256,
    )

    assert all(value > 0 for value in ordinary)
    assert all(left < right for left, right in pairwise(ordinary))
    assert [compute_thetas.round_down(value, mode) for value in ordinary] == [
        compute_thetas.round_down(value, mode) for value in checked
    ]
