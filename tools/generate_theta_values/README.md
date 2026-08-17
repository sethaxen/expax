# Generate theta values

This tool generates the $\theta$ values imported by `expax._theta`.

## Execution

From the repository root, run:

```console
uv run --project tools/generate_theta_values --locked python tools/generate_theta_values/generate_theta_values.py
```

This command overwrites `src/expax/_generated_theta_values.py`, and `git diff` can be used to inspect any changes to the values.

## Mathematical background

`expax` approximates the exponential with the degree $m$ Taylor polynomial

\[
\exp(x) \approx T_m(x) := \sum_{j=0}^{m} \frac{x^j}{j!}
\]

The following is based on Section 5 and the appendix of [^1], which derives approximation thresholds from relative backward error.
Let $h_m$ denote the backward error of the Taylor approximation, defined by

\[
h_m(x)
  := \log\!\left(\exp(-x)T_m(x)\right)
  = \sum_{k=m+1}^{\infty} c_k x^k.
\]

For a scalar or matrix argument $B$ within the convergence domain of this power series, the Taylor approximation is the exact exponential of the perturbed argument $B + h_m(B)$.
Bounding $h_m(B)$ therefore certifies the backward error of the approximation.
Let $\widetilde h_m$ denote the majorant of $h_m$ for positive scalar argument, defined by

\[
\widetilde h_m(x) := \sum_{k=m+1}^{\infty} |c_k|x^k, \quad x > 0.
\]

For a square matrix $B$ and any submultiplicative matrix norm $\lVert \cdot \rVert$,

\[
\begin{aligned}
\lVert h_m(B)\rVert
&\leq \sum_{k=m+1}^{\infty}|c_k|\,\lVert B^k\rVert \\
&\leq \sum_{k=m+1}^{\infty}|c_k|\,\lVert B\rVert^k
 = \widetilde h_m(\lVert B\rVert),
\end{aligned}
\]

so the relative backward error is bounded by
\[
\frac{\lVert h_m(B)\rVert}{\lVert B\rVert} \leq \frac{\widetilde h_m(\lVert B\rVert)}{\lVert B\rVert}.
\]

$\theta_m$ is the largest scalar norm threshold for which the relative backward error of the degree-$m$ approximation does not exceed a given tolerance `tol`:

\[
\theta_m
  = \max\left\{\theta:
      \frac{\widetilde h_m(\theta)}{\theta} \leq \mathrm{tol}, \;\theta > 0
    \right\}.
\]

Because $\widetilde h_m(r) / r$ has nonnegative coefficients, it is nondecreasing for $r \geq 0$, so

\[
\lVert B\rVert \leq \theta_m
\quad\Longrightarrow\quad
\frac{\lVert h_m(B)\rVert}{\lVert B\rVert}\leq\mathrm{tol}.
\]

## Script structure

The script has three main components:

1. `exponential_taylor_backward_error_series` yields each $h_m$ exactly over the rationals.
   It updates $e^{-x} T_m(x)$ incrementally using

   \[
   e^{-x}T_m(x)
     = e^{-x}T_{m-1}(x) + \frac{x^m}{m!}e^{-x},
   \]

   which avoids recomputing a long product for every degree.
   The solver builds both majorants immediately, so only one large rational series is retained.
2. The generic solver forms $\tilde{h}_m(x) / x$, encloses the positive solution where it equals the tolerance, and rounds that solution downward to float64.
3. The output layer labels the tolerance tables and renders the Python module used by `expax`.

> [!NOTE]
> It is straightforward to adapt this tool to generate $\theta$ values for a different approximation or matrix function.
> Just replace `exponential_taylor_backward_error_series` with a generator that yields the exact backward-error series $h_m$ for the corresponding function/approximation.

In practice, we must truncate $h_m$.
We use a 1,024-term truncation to calculate the $\theta_m$ values and validate that expanding the truncation by 200 additional terms produces the exact same values.
This inexpensive check guards against a truncation that is too short to determine the emitted value.

## References

[^1]: Higham NJ, Al-Mohy AH. Computing matrix functions. *Acta Numerica*. 2010;19:159–208. doi:[10.1017/S0962492910000036](https://doi.org/10.1017/S0962492910000036). [eprint](https://eprints.maths.manchester.ac.uk/1406/).
