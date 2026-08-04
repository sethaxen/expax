import importlib.util
import sys
from pathlib import Path

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
