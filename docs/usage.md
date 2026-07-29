# Usage

## Plan and apply

`expm_multiply` is a callable factory. It plans from the current operator
parameters and returns an action on one vector.

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

Planning—including trace estimation, norm estimation, and selection of the
Taylor degree and scaling count—uses the runtime value of `matrix`. Gradients
through planning are stopped; the returned Taylor action remains differentiable.

The action accepts one vector. Apply it to multiple vectors with `jax.vmap`:

```python
results = jax.vmap(action)(vectors)
```

## Reuse a fixed operator plan

When operator parameters are fixed but time changes, omit `times`. This computes
the operator-dependent power-norm matrix once and returns a time factory.

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

The random key belongs to planning, not application, so returned actions accept
only their vector.

## Several times

The default `algorithm="parallel"` accepts arbitrary time points. It returns a
PyTree whose leaves have a leading time axis and synchronizes the runtime Taylor
loops across time lanes.

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

For a one-dimensional array of equally spaced points, opt into
`algorithm="time_grid"`. This uses Al-Mohy--Higham Algorithm 5.2 and reuses
Taylor terms across grid points.

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

`time_grid` is an algorithm selection, not a runtime grid detector. The caller
must supply at least two equally spaced points. The default parallel algorithm
can be faster on accelerators, so the specialized grid algorithm is opt-in.

## Estimators

A trace estimator has the compositional signature

```python
trace_estimator(matvec, v_like, key, *parameters) -> scalar
```

The default uses XTrace with two samples when the vector-space dimension permits
it, and exact basis actions for dimensions one and two. A known trace can be
supplied with the same interface.

```python
def known_trace(_matvec, _v_like, _key, matrix):
    return jnp.trace(matrix)
```

A norm estimator is an `(estimator, cost)` pair. Its callable uses the same
arguments and returns an operator 1-norm estimate. `cost` is the scalar number of
`matvec` applications needed for one estimate; power costs are derived by
repeated operator application. The default is `expax.normest.onenormest()` when
the dimension permits it and exact basis actions for dimensions one and two.

Adjoint actions are derived internally with JAX linear transposition. Estimators
never require a separate adjoint argument.

## Control flow and autodiff

`max_scaling` can bound runtime work. If the selected scaling count is
non-finite or exceeds this bound, the affected result leaves are NaN and the
unbounded scaling recurrence is skipped.

Runtime-dependent loops use the injected `while_loop`, whose default is
`jax.lax.while_loop`. The default supports forward-mode differentiation through
dynamic loops but not reverse mode. A checkpointed or otherwise
reverse-mode-compatible implementation can be supplied without adding a
control-flow dependency to `expax`.
