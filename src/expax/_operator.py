import math

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def _validate_vector_space(v_like):
    leaves = jax.tree.leaves(v_like)
    if not leaves:
        raise ValueError("v_like must be a nonempty PyTree")

    for leaf in leaves:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            raise TypeError("every v_like leaf must be a JAX array")

    dtypes = {jnp.dtype(leaf.dtype) for leaf in leaves}
    if len(dtypes) != 1:
        raise TypeError("all v_like leaves must have the same dtype")

    (dtype,) = dtypes
    if not jnp.issubdtype(dtype, jnp.inexact):
        raise TypeError("v_like leaves must have an inexact dtype")
    if sum(math.prod(leaf.shape) for leaf in leaves) == 0:
        raise ValueError("v_like must describe a positive-dimensional vector space")

    return ravel_pytree(v_like)


def _linear_adjoint(matvec, v_like):
    def adjoint(y, *parameters):
        transpose = jax.linear_transpose(lambda x: matvec(x, *parameters), v_like)
        y_conjugate = jax.tree.map(jnp.conj, y)
        (result,) = transpose(y_conjugate)
        return jax.tree.map(jnp.conj, result)

    return adjoint
