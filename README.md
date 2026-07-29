# expax

`expax` computes matrix-exponential actions `exp(t A) v` in JAX without materializing either the linear operator or its exponential.

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
vector = jnp.arange(dim, dtype=float)

# construct exponential action operator
action = expax.expm_multiply(
    matvec,
    diag,
    sup_diag,
    times=jnp.array(0.5),
    v_like=vector,
    key=jax.random.key(0),
)

# call it on an arbitrary vector
result = action(vector)
```

Operators and vectors may be arbitrary PyTrees of JAX arrays.
Planning uses runtime operator parameters inside `jax.jit`, arbitrary time points use synchronized parallel actions, and equally spaced points can opt into the specialized Al-Mohy--Higham interval algorithm.
