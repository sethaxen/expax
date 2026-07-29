"""Taylor recurrences for matrix-exponential actions."""

import jax
import jax.numpy as jnp


def _taylor_action(
    times,
    matvec,
    parameters,
    mu,
    degrees,
    scalings,
    valid,
    *,
    tol,
    max_degree,
    while_loop,
    scalar_time,
):
    del scalar_time

    def action(vector):
        dtype = jnp.result_type(
            times, mu, *(leaf.dtype for leaf in jax.tree.leaves(vector))
        )
        vector = jax.tree.map(lambda leaf: leaf.astype(dtype), vector)
        scaling_count = scalings.astype(jnp.int32)
        eta = jnp.exp(times * mu / scalings)

        def evaluate(initial_vector):
            def cond_fun(state):
                iteration, _ = state
                return iteration < scaling_count

            def body_fun(state):
                iteration, base = state
                previous_norm = _tree_infinity_norm(base)
                converged = jnp.array(False)
                inner_state = (base, base, previous_norm, converged)

                def taylor_term(degree, state):
                    converged = state[-1]
                    active = (degree <= degrees) & ~converged

                    def apply_term(state):
                        term, total, previous_norm, _ = state
                        image = matvec(term, *parameters)
                        image = jax.tree.map(
                            lambda image_leaf, term_leaf: image_leaf - mu * term_leaf,
                            image,
                            term,
                        )
                        coefficient = times / (scalings * degree)
                        next_term = _tree_scale(coefficient, image)
                        next_total = _tree_add(total, next_term)
                        current_norm = _tree_infinity_norm(next_term)
                        converged = previous_norm + current_norm <= (
                            tol * _tree_infinity_norm(next_total)
                        )
                        return (
                            next_term,
                            next_total,
                            current_norm,
                            converged,
                        )

                    return jax.lax.cond(active, apply_term, lambda state: state, state)

                # Algorithm 3.2, lines 12--17, with the equation (3.15) test.
                _, total, _, _ = jax.lax.fori_loop(
                    1, max_degree + 1, taylor_term, inner_state
                )
                next_base = _tree_scale(eta, total)
                return iteration + 1, next_base

            # Algorithm 3.2, lines 10--19.
            return while_loop(
                cond_fun, body_fun, (jnp.array(0, dtype=jnp.int32), initial_vector)
            )[1]

        return jax.lax.cond(
            valid,
            evaluate,
            lambda initial_vector: _tree_nan_like(initial_vector),
            vector,
        )

    return action


def _tree_add(left, right):
    return jax.tree.map(lambda x, y: x + y, left, right)


def _tree_scale(scalar, tree):
    return jax.tree.map(lambda leaf: scalar * leaf, tree)


def _tree_infinity_norm(tree):
    norms = [jnp.max(jnp.abs(leaf)) for leaf in jax.tree.leaves(tree) if leaf.size > 0]
    return jnp.max(jnp.stack(norms))


def _tree_nan_like(tree):
    return jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.nan), tree)
