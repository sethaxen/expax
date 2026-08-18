import math

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def _validate_vector_space(x_like):
    leaves = jax.tree.leaves(x_like)
    if not leaves:
        raise ValueError("x_like must be a nonempty PyTree")

    for leaf in leaves:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            raise TypeError("every x_like leaf must be a JAX array")

    dtypes = {jnp.dtype(leaf.dtype) for leaf in leaves}
    if len(dtypes) != 1:
        raise TypeError("all x_like leaves must have the same dtype")

    (dtype,) = dtypes
    if not jnp.issubdtype(dtype, jnp.inexact):
        raise TypeError("x_like leaves must have an inexact dtype")
    if sum(math.prod(leaf.shape) for leaf in leaves) == 0:
        raise ValueError("x_like must describe a positive-dimensional vector space")

    return ravel_pytree(x_like)


def _linear_adjoint(func, *primals):
    transpose = jax.linear_transpose(func, *primals)

    def tree_conj(x):
        return jax.tree.map(jnp.conj, x)

    def adjoint(*primal_results):
        return tree_conj(transpose(*tree_conj(primal_results)))

    return adjoint
