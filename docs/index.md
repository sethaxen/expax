# expax

`expax` computes matrix-exponential actions

\[
\exp(tA)v
\]

in JAX without forming either \(A\) or \(\exp(tA)\). Linear operators and vectors
may be arbitrary PyTrees of JAX arrays, and operator parameters remain
differentiable runtime values.

```{toctree}
:maxdepth: 2

usage
api
```
