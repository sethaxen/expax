# Generate theta values

Generate the theta values imported by `expax._theta`.

From the repository root, run:

```console
uv run --project tools/generate_theta_values --locked python tools/generate_theta_values/generate_theta_values.py
```

This writes `src/expax/_generated_theta_values.py`. Inspect `git diff` before
accepting regenerated values.

The generator builds its coefficients exactly as rational numbers, takes an
absolute-coefficient majorant, and uses Arb to enclose every positive root.
Each enclosure is then converted downward to binary64, so the emitted value is
the largest binary64 value safely below the exact root.

Higham NJ, Al-Mohy AH. Computing matrix functions. Acta Numerica.
2010;19:159-208. doi:10.1017/S0962492910000036.
