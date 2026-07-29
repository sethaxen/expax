# Usage

## Plan and apply

`expm_multiply` is a callable factory.
It plans from the current matvec operator parameters and returns a matvec operator that computes the action of the matrix exponential of the original matvec.

```python
import jax
import jax.numpy as jnp

import expax


def matvec(vector, matrix):
    return matrix @ vector


@jax.jit
def evaluate(matrix, time, vector, key):
    action = expax.expm_multiply(
        matvec,
        matrix,
        times=time,
        v_like=vector,
        key=key,
    )
    return action(vector)
```

Planning—including trace estimation, norm estimation, and selection of the Taylor degree and scaling count—calls `matvec` multiple times using the runtime value of `matrix`, and the returned action is only valid for that `matrix`.

The action accepts one vector. Apply it to multiple vectors with `jax.vmap`:

```python
results = jax.vmap(action)(vectors)
```

## Reuse a fixed operator plan

When operator parameters are fixed but time change between serial calls, omit `times`.
This may perform more `matvec` calls at planning time, but returns a callable that can be applied to arbitrary time points without re-planning.

```python
at_times = expax.expm_multiply(
    matvec,
    matrix,
    v_like=vector,
    key=key,
)

action_at_half = at_times(jnp.array(0.5))
result = action_at_half(vector)
```

The random key belongs to planning, not application, so returned actions accept only their vector.

## Several times

The default `algorithm="parallel"` accepts arbitrary time points.
It returns a PyTree whose leaves have a leading time axis and synchronizes the runtime Taylor loops across time lanes.

```python
times = jnp.array([-0.2, 0.0, 0.7])
action = expax.expm_multiply(
    matvec,
    matrix,
    times=times,
    v_like=vector,
    key=key,
)
trajectory = action(vector)
```

For a one-dimensional array of equally spaced points, opt into `algorithm="time_grid"`.
This reuses Taylor terms across grid points so can perform fewe `matvec` calls than the parallel algorithm; however, it is sequential and may be less efficient on GPU and TPU.

```python
times = jnp.linspace(0.0, 1.0, 101)
action = expax.expm_multiply(
    matvec,
    matrix,
    times=times,
    v_like=vector,
    key=key,
    algorithm="time_grid",
)
trajectory = action(vector)
```

`time_grid` requires at least two equally spaced, finite points at runtime; invalid grids produce NaN result leaves.
The default parallel algorithm can be faster on accelerators, so the specialized grid algorithm is opt-in.

## Estimators

A trace estimator has the compositional signature

```python
trace_estimator(matvec, v_like, key, *parameters) -> scalar
```

The default uses XTrace with two samples when the vector-space dimension permits it, and exact basis actions for dimensions one and two.
A known trace can be supplied with the same interface.

```python
def known_trace(_matvec, _v_like, _key, matrix):
    return jnp.trace(matrix)
```

A norm estimator is an `(estimator, cost)` pair. Its callable uses the same arguments and returns an operator 1-norm estimate.
`cost` is the work model in scalar `matvec` equivalents used by the planning criterion, rather than a promise of the estimator's exact adaptive runtime count;
power costs are derived by repeated operator application.
The default is `expax.normest.onenormest()`, whose model is four applications per block column, when the dimension permits it and exact basis actions for dimensions one and two.

## Control flow and autodiff

`max_scaling` can bound runtime work.
If the selected scaling count is non-finite or exceeds this bound, the affected result leaves are NaN and the unbounded scaling recurrence is skipped.

Runtime-dependent loops use the injected `while_loop`, whose default is `jax.lax.while_loop`.
This default supports forward-mode differentiation through dynamic loops but not reverse mode.
A checkpointed or otherwise reverse-mode-compatible implementation can be supplied without adding a control-flow dependency to `expax`.
For example, `equinox.internal.while_loop` supports checkpointed reverse-mode differentiation but not forward-mode differentiation:

```python
from functools import partial
import equinox as eqx

while_loop = partial(eqx.internal.while_loop, kind="checkpointed", checkpoints=20)
action = expax.expm_multiply(
    matvec,
    matrix,
    times=times,
    v_like=vector,
    key=key,
    while_loop=while_loop,
)

def loss(vector):
    result = action(vector)
    return jnp.sum(result**2)

grad_loss = jax.jit(jax.grad(loss))
grad_loss(vector)
```
