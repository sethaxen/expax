# expax

[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange)](#development-status)

> [!WARNING]
> **Development status:** `expax` is under active development and is not yet ready for production or research-critical use.
> The API, numerical methods, and implementation may change substantially, and correctness and performance are in the process of being validated across the full intended range of problems.
> Please independently verify results before relying on them in scientific work.

`expax` computes matrix-exponential actions `exp(t A) x` in JAX without materializing either the linear operator or its exponential.

```python
import jax
import jax.numpy as jnp

import expax


# action of an upper bidiagonal matrix
def matvec(vector, diag, sup_diag):
    return (diag * vector).at[:-1].add(sup_diag * vector[1:])

dim = 10_000
diag = jnp.ones(dim)
sup_diag = jnp.ones(dim - 1)
x = jnp.arange(dim, dtype=float)

# construct exponential action operator
action = expax.expm_multiply(
    matvec,
    diag,
    sup_diag,
    times=jnp.array(0.5),
    x_like=jnp.zeros_like(x),
    key=jax.random.key(0),
)

# call it on an arbitrary vector
result = action(x)
```

Operators and vectors may be arbitrary PyTrees of JAX arrays.
Planning uses runtime operator parameters inside `jax.jit`, and arbitrary time points use synchronized actions.
