import hashlib
import importlib.util
import io
import sys
from itertools import pairwise
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
import scipy.io
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


def _mat_payload(values):
    buffer = io.BytesIO()
    scipy.io.savemat(buffer, {"theta": np.asarray(values, dtype=np.float64)})
    return buffer.getvalue()


def test_load_upstream_verifies_hash_shape_and_values():
    payload = _mat_payload([1.0, 2.0, 3.0])
    source = compute_thetas.UpstreamSource(
        filename="theta.mat",
        sha256=hashlib.sha256(payload).hexdigest(),
        expected_length=3,
    )

    received = compute_thetas.load_upstream(source, fetch=lambda _url: payload)

    np.testing.assert_array_equal(received, [1.0, 2.0, 3.0])


def test_load_upstream_rejects_hash_mismatch():
    payload = _mat_payload([1.0])
    source = compute_thetas.UpstreamSource("theta.mat", "0" * 64, 1)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        compute_thetas.load_upstream(source, fetch=lambda _url: payload)


def test_load_upstream_rejects_nonfinite_values():
    payload = _mat_payload([1.0, np.nan])
    source = compute_thetas.UpstreamSource(
        "theta.mat",
        hashlib.sha256(payload).hexdigest(),
        2,
    )

    with pytest.raises(ValueError, match="finite positive"):
        compute_thetas.load_upstream(source, fetch=lambda _url: payload)


def test_load_upstream_rejects_unexpected_length_and_shape():
    short_payload = _mat_payload([1.0])
    short_source = compute_thetas.UpstreamSource(
        "theta.mat",
        hashlib.sha256(short_payload).hexdigest(),
        2,
    )
    column_buffer = io.BytesIO()
    scipy.io.savemat(column_buffer, {"theta": np.asarray([[1.0], [2.0]])})
    column_payload = column_buffer.getvalue()
    column_source = compute_thetas.UpstreamSource(
        "theta.mat",
        hashlib.sha256(column_payload).hexdigest(),
        2,
    )

    with pytest.raises(ValueError, match="expected shape"):
        compute_thetas.load_upstream(short_source, fetch=lambda _url: short_payload)
    with pytest.raises(ValueError, match="expected shape"):
        compute_thetas.load_upstream(column_source, fetch=lambda _url: column_payload)


def test_load_upstream_rejects_missing_theta_variable():
    buffer = io.BytesIO()
    scipy.io.savemat(buffer, {"other": np.asarray([1.0])})
    payload = buffer.getvalue()
    source = compute_thetas.UpstreamSource(
        "theta.mat",
        hashlib.sha256(payload).hexdigest(),
        1,
    )

    with pytest.raises(ValueError, match="no theta variable"):
        compute_thetas.load_upstream(source, fetch=lambda _url: payload)


def test_comparison_reports_target_matches_and_decimal_agreement():
    mode = next(mode for mode in compute_thetas.NATIVE_MODES if mode.name == "float32")
    roots = (mp.mpf("1.25"),)
    upstream = np.asarray([1.25], dtype=np.float64)

    comparison = compute_thetas.compare_values(
        "single",
        mode,
        roots,
        upstream,
        tolerance_match=True,
    )

    assert comparison.exact_binary64_matches == 1
    assert comparison.target_matches == 1
    assert comparison.common_significant_digits == 17
    assert comparison.rows[0].degree == 1
    assert comparison.rows[0].ulp_distance == 0


def test_decimal_agreement_finds_greatest_common_precision():
    received = compute_thetas.maximum_common_significant_digits(
        (mp.mpf("1.23456"),),
        (1.23457,),
    )

    assert received == 5


def test_comparison_supports_a_mode_without_upstream_data():
    mode = next(mode for mode in compute_thetas.NATIVE_MODES if mode.name == "bfloat16")

    comparison = compute_thetas.compare_values(
        "bfloat16",
        mode,
        (mp.mpf("1.0"),),
        None,
        tolerance_match=True,
    )

    assert comparison.upstream_available is False
    assert comparison.rows[0].upstream is None
    assert comparison.common_significant_digits is None
