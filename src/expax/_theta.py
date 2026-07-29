from decimal import Decimal, localcontext
from functools import cache

import jax.numpy as jnp

_THETA_DOUBLE = (
    2.220446049250313e-16,
    2.5809568029946243e-08,
    1.3863478661191185e-05,
    0.00033971688399768305,
    0.0024008763578872742,
    0.009065656407595102,
    0.023844555325002736,
    0.049912288711153226,
    0.08957760203223343,
    0.1441829761614378,
    0.21423580684517107,
    0.299615891381158,
    0.3997775336316795,
    0.5139146936124294,
    0.6410835233041199,
    0.7802874256626576,
    0.9305328460786568,
    1.0908637192900361,
    1.2603810606426387,
    1.438252596804337,
    1.6237159502358214,
    1.8160778162150852,
    2.014710780944616,
    2.2190488693650896,
    2.4285825244428265,
    2.6428534574594353,
    2.861449633934264,
    3.084000544989162,
    3.310172839890271,
    3.5396663487436895,
    3.772210495681751,
    4.00756108611804,
    4.245497442579696,
    4.485819859447369,
    4.728347345793539,
    4.972915626191981,
    5.219375371084058,
    5.467590630524544,
    5.717437447572013,
    5.968802630041849,
    6.221582661689891,
    6.4756827360799845,
    6.731015898381024,
    6.98750228213063,
    7.245068429597953,
    7.503646685788864,
    7.763174657377987,
    8.02359472893998,
    8.284853629803918,
    8.546902045684932,
    8.809694269971322,
    9.073187890176145,
    9.337343505612013,
    9.602124472826557,
    9.8674966757534,
)

_THETA_SINGLE = (
    1.1920927998154304e-07,
    0.0005978858893805217,
    0.011233864735286708,
    0.05116619363445085,
    0.13084871645994703,
    0.24952893228466977,
    0.40145824235104804,
    0.5800524627688768,
    0.779511337435803,
    0.9951840790004457,
    1.2234795424241427,
    1.4616615072090335,
    1.7076485296087012,
    1.959850585959898,
    2.2170443949747205,
    2.4782808775219713,
    2.7428171126987797,
    3.0100663628176343,
    3.279561212635997,
    3.5509262147064953,
    3.823857425450966,
    4.098106972191506,
    4.3734713118405,
    4.649782224100758,
    4.926899843755911,
    5.204707228012361,
    5.483106087658634,
    5.762013408447769,
    6.041358758192571,
    6.321082126301961,
    6.601132179501162,
    6.8814648452097185,
    7.162042154487758,
    7.4428312919365975,
    7.723803811553991,
    8.004934986436286,
    8.286203267002167,
    8.567589827662578,
    8.849078185923954,
    9.13065388109011,
    9.412304202219431,
    9.694017956963032,
    9.975785274470699,
    10.257597436797516,
    10.539446734242194,
    10.821326340852181,
    11.103230206980713,
    11.385152966309168,
    11.667089855178881,
    11.949036642429446,
    12.230989568231498,
    12.51294529064492,
    12.794900838842899,
    13.07685357212606,
    13.35880114398722,
)


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

        return float((lower + upper) / 2)


@cache
def _theta(dtype, tol, max_degree):
    dtype = jnp.dtype(dtype)
    if tol is None:
        if dtype in (jnp.dtype(jnp.float32), jnp.dtype(jnp.complex64)):
            values = _THETA_SINGLE
        elif dtype in (jnp.dtype(jnp.float64), jnp.dtype(jnp.complex128)):
            values = _THETA_DOUBLE
        else:
            raise TypeError(f"unsupported vector-space dtype: {dtype}")
    elif float(tol) == 2.0**-24:
        values = _THETA_SINGLE
    elif float(tol) == 2.0**-53:
        values = _THETA_DOUBLE
    else:
        values = tuple(_compute_theta(m, float(tol)) for m in range(1, max_degree + 1))
    return values[:max_degree]
