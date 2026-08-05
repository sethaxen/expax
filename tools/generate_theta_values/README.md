# Generate theta values

Generate the theta values imported by `expax._theta`.

From the repository root, run:

```console
uv run --project tools/generate_theta_values --locked python tools/generate_theta_values/generate_theta_values.py
```

This writes `src/expax/_generated_theta_values.py`. Inspect `git diff` before
accepting regenerated values.

The generator builds its coefficients exactly as rational numbers and uses Arb
to enclose the positive roots of absolute-coefficient majorants. It solves a
1,050-term majorant and requires a 1,200-term extension to select the same
downward-rounded binary64 value before emitting it.

Higham NJ, Al-Mohy AH. Computing matrix functions. Acta Numerica.
2010;19:159-208. doi:10.1017/S0962492910000036.
