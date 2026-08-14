# Generate theta values

This tool generates the theta values imported by `expax._theta`. It is an
isolated, locked project because none of its arbitrary-precision dependencies
are needed when using Expax.

From the repository root, run:

```console
uv run --project tools/generate_theta_values --locked python tools/generate_theta_values/generate_theta_values.py
```

The command writes `src/expax/_generated_theta_values.py`. Inspect its diff
before accepting regenerated values.

## Mathematical construction

For a Taylor degree `m`, Expax approximates the exponential with

\[
T_m(x) = \sum_{j=0}^{m} \frac{x^j}{j!}.
\]

Section 5 of Higham and Al-Mohy derives approximation thresholds from relative
backward error. Appendix A applies that construction to Taylor approximants.
Immediately before equation (A.3), the paper defines

\[
h_m(x)
  = \log\!\left(e^{-x}T_m(x)\right)
  = \sum_{k=m+1}^{\infty} c_k x^k
\]

and the absolute-coefficient majorant

\[
\widetilde h_m(x) = \sum_{k=m+1}^{\infty} |c_k|x^k.
\]

The definition of `h_m` means that `T_m(B) = exp(B + h_m(B))`: the Taylor
approximation is the exact exponential of a nearby argument. Bounding `h_m`
therefore bounds the backward error.

The theta value for degree `m` and tolerance `tol` is therefore

\[
\theta_m
  = \max\left\{\theta:
      \frac{\widetilde h_m(\theta)}{\theta} \leq \mathrm{tol}
    \right\}.
\]

This is equation (A.3), with the paper's double-precision unit roundoff
replaced by each tolerance supported by Expax.

The threshold applies to Expax's scaled action as well as to the paper's
scaling-and-squaring setting. With Expax's trace shift `mu`, let
`B = t(A - mu I) / s`. Expax applies the scalar factor `exp(t mu / s)`
separately at each step, so formally

\[
\left(e^{t\mu/s}T_m(B)\right)^s
  = \exp\!\left(tA + s h_m(B)\right).
\]

Consequently, the relative backward error is controlled by the same ratio
`h_tilde_m(||B||) / ||B||`. Theta tells the planner how large a scaled step can
be for a selected Taylor degree and tolerance.

## Script structure

The generator follows the mathematical construction in three layers:

1. `exponential_taylor_backward_error_series` constructs each `h_m` exactly
   over the rationals. It updates `exp(-x) T_m(x)` incrementally using

   \[
   e^{-x}T_m(x)
     = e^{-x}T_{m-1}(x) + \frac{x^m}{m!}e^{-x},
   \]

   which avoids recomputing a long product for every degree.
2. The generic solver forms `h_tilde_m(x) / x`, encloses the positive solution
   where it equals the tolerance, and rounds that solution downward to
   binary64.
3. The output layer labels the tolerance tables and renders the Python module
   used by Expax.

To adapt the tool to another approximation or matrix function, implement a
replacement for `exponential_taylor_backward_error_series` that returns its
exact backward-error series `h_m`. The majorant construction, root enclosure,
rounding, and rendering do not depend on the Taylor exponential.

The main calculation truncates each backward-error series after 1,050 powers.
During the same run, a 1,200-power extension must put the root in the same
binary64 interval. This inexpensive check guards against a truncation that is
too short to determine the emitted value.

## Reference

Higham NJ, Al-Mohy AH. Computing matrix functions. *Acta Numerica*.
2010;19:159–208. [doi:10.1017/S0962492910000036](https://doi.org/10.1017/S0962492910000036).
See Section 5—particularly Lemma 5.1 and the discussion following equation
(5.4)—and Appendix A, especially equation (A.3). A freely available
[author eprint](https://eprints.maths.manchester.ac.uk/1406/) is also
available.
