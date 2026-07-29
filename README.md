# expax

`expax` computes matrix-exponential actions `exp(t A) v` in JAX without
materializing either the linear operator or its exponential.

```python
import jax
import jax.numpy as jnp

import expax


def matvec(vector, matrix):
    return matrix @ vector


matrix = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
vector = jnp.array([1.0, 0.0])
action = expax.expm_multiply(
    matvec,
    matrix,
    times=jnp.array(0.5),
    v_like=vector,
    key=jax.random.key(0),
)
result = action(vector)
```

Operators and vectors may be arbitrary PyTrees of JAX arrays. Planning uses
runtime operator parameters inside `jax.jit`, arbitrary time points use
synchronized parallel actions, and equally spaced points can opt into the
specialized Al-Mohy--Higham interval algorithm.

See the [usage guide](docs/usage.md) and [API reference](docs/api.md).
