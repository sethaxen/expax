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
    times = jnp.atleast_1d(times)
    degrees = jnp.atleast_1d(degrees)
    scalings = jnp.atleast_1d(scalings)
    valid = jnp.atleast_1d(valid)
    num_times = times.shape[0]

    def action(vector):
        dtype = jnp.result_type(
            times, mu, *(leaf.dtype for leaf in jax.tree.leaves(vector))
        )
        vector = jax.tree.map(lambda leaf: leaf.astype(dtype), vector)
        base = _tree_broadcast(vector, num_times)
        safe_scalings = jnp.where(valid, scalings, 1)
        scaling_counts = safe_scalings.astype(jnp.int32)
        eta = jnp.exp(times * mu / safe_scalings)

        def cond_fun(state):
            iteration, _ = state
            return jnp.any(valid & (iteration < scaling_counts))

        def body_fun(state):
            iteration, base = state
            active_outer = valid & (iteration < scaling_counts)
            previous_norm = _tree_infinity_norm_batched(base)
            converged = ~active_outer
            inner_state = (base, base, previous_norm, converged)

            def taylor_term(degree, state):
                converged = state[-1]
                active = active_outer & (degree <= degrees) & ~converged

                def apply_term(state):
                    term, total, previous_norm, converged = state
                    image = jax.vmap(lambda x: matvec(x, *parameters))(term)
                    image = jax.tree.map(
                        lambda image_leaf, term_leaf: image_leaf - mu * term_leaf,
                        image,
                        term,
                    )
                    coefficient = times / (safe_scalings * degree)
                    candidate_term = _tree_scale_batched(coefficient, image)
                    candidate_total = _tree_add(total, candidate_term)
                    current_norm = _tree_infinity_norm_batched(candidate_term)
                    candidate_converged = previous_norm + current_norm <= (
                        tol * _tree_infinity_norm_batched(candidate_total)
                    )
                    return (
                        _tree_where_batched(active, candidate_term, term),
                        _tree_where_batched(active, candidate_total, total),
                        jnp.where(active, current_norm, previous_norm),
                        converged | (active & candidate_converged),
                    )

                return jax.lax.cond(
                    jnp.any(active), apply_term, lambda state: state, state
                )

            # Algorithm 3.2, lines 12--17, with the equation (3.15) test.
            _, total, _, _ = jax.lax.fori_loop(
                1, max_degree + 1, taylor_term, inner_state
            )
            candidate_base = _tree_scale_batched(eta, total)
            next_base = _tree_where_batched(active_outer, candidate_base, base)
            return iteration + 1, next_base

        # The shared loop synchronizes arbitrary time points at max(s), while the
        # lane mask preserves each point's selected recurrence from Algorithm 3.2.
        result = while_loop(cond_fun, body_fun, (jnp.array(0, dtype=jnp.int32), base))[
            1
        ]
        result = _tree_where_batched(valid, result, _tree_nan_like(result))
        if scalar_time:
            return jax.tree.map(lambda leaf: leaf[0], result)
        return result

    return action


def _tree_broadcast(tree, size):
    return jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (size, *leaf.shape)),
        tree,
    )


def _tree_where_batched(mask, left, right):
    return jax.tree.map(
        lambda x, y: jnp.where(
            jnp.reshape(mask, (mask.shape[0],) + (1,) * (x.ndim - 1)),
            x,
            y,
        ),
        left,
        right,
    )


def _tree_add(left, right):
    return jax.tree.map(lambda x, y: x + y, left, right)


def _tree_scale_batched(scalars, tree):
    return jax.tree.map(
        lambda leaf: (
            jnp.reshape(scalars, (scalars.shape[0],) + (1,) * (leaf.ndim - 1)) * leaf
        ),
        tree,
    )


def _tree_infinity_norm_batched(tree):
    norms = [
        jnp.max(jnp.abs(leaf), axis=tuple(range(1, leaf.ndim)))
        for leaf in jax.tree.leaves(tree)
        if leaf.size > 0
    ]
    return jnp.max(jnp.stack(norms), axis=0)


def _tree_nan_like(tree):
    return jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.nan), tree)
