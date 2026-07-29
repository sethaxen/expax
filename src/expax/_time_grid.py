"""Taylor actions specialized for equally spaced time points."""

import jax
import jax.numpy as jnp

from expax._taylor import _taylor_action


def _time_grid_action(
    times,
    matvec,
    parameters,
    mu,
    select,
    *,
    tol,
    max_degree,
    while_loop,
):
    """Construct the equally spaced action from Algorithm 5.2."""
    num_steps = times.shape[0] - 1
    initial_time = times[0]
    interval = times[-1] - initial_time
    step_size = interval / num_steps
    interval_degree, interval_scaling, interval_valid = select(interval)
    initial_degree, initial_scaling, initial_valid = select(initial_time)
    step_degree, step_scaling, step_valid = select(step_size)
    grid_valid = _is_evenly_spaced(times, step_size)
    initial_action = _taylor_action(
        initial_time,
        matvec,
        parameters,
        mu,
        initial_degree,
        initial_scaling,
        initial_valid,
        tol=tol,
        max_degree=max_degree,
        while_loop=while_loop,
        scalar_time=True,
    )
    step_action = _taylor_action(
        step_size,
        matvec,
        parameters,
        mu,
        step_degree,
        step_scaling,
        step_valid,
        tol=tol,
        max_degree=max_degree,
        while_loop=while_loop,
        scalar_time=True,
    )

    def action(vector):
        dtype = jnp.result_type(
            times, mu, *(leaf.dtype for leaf in jax.tree.leaves(vector))
        )
        vector = jax.tree.map(lambda leaf: leaf.astype(dtype), vector)

        def evaluate(vector):
            initial = initial_action(vector)

            def sequential(initial):
                def advance(carry, _):
                    result = step_action(carry)
                    return result, result

                _, rest = jax.lax.scan(advance, initial, xs=None, length=num_steps)
                return _tree_prepend(initial, rest)

            def reuse_terms(initial):
                return _reuse_taylor_terms(
                    initial,
                    num_steps,
                    step_size,
                    matvec,
                    parameters,
                    mu,
                    interval_degree,
                    interval_scaling,
                    tol=tol,
                    max_degree=max_degree,
                    while_loop=while_loop,
                )

            # Algorithm 5.2, lines 8--14, avoids overscaling when q <= s.
            return jax.lax.cond(
                num_steps <= interval_scaling,
                sequential,
                reuse_terms,
                initial,
            )

        return jax.lax.cond(
            initial_valid & interval_valid & grid_valid,
            evaluate,
            lambda vector: _tree_nan_times(vector, num_steps + 1),
            vector,
        )

    return action


def _is_evenly_spaced(times, step_size):
    real_dtype = jnp.real(jnp.asarray(step_size)).dtype
    scale = jnp.maximum(1, jnp.max(jnp.abs(times)))
    tolerance = 32 * jnp.finfo(real_dtype).eps * scale
    differences = jnp.diff(times)
    return jnp.all(jnp.isfinite(times)) & jnp.all(
        jnp.abs(differences - step_size) <= tolerance
    )


def _reuse_taylor_terms(
    initial,
    num_steps,
    step_size,
    matvec,
    parameters,
    mu,
    interval_degree,
    interval_scaling,
    *,
    tol,
    max_degree,
    while_loop,
):
    scaling = interval_scaling.astype(jnp.int32)
    group_size = num_steps // scaling
    outputs = _tree_output_buffer(initial, num_steps + 1)
    outputs = _tree_set(outputs, 0, initial)

    def group_cond(state):
        offset, _, _ = state
        return offset < num_steps

    def group_body(state):
        offset, base, outputs = state
        steps_in_group = jnp.minimum(group_size, num_steps - offset)
        terms = _tree_term_buffer(base, max_degree + 1)
        terms = _tree_set(terms, 0, base)

        def step_cond(state):
            step, _, _, _ = state
            return step <= steps_in_group

        def step_body(state):
            step, terms, formed_degree, outputs = state
            total = base
            previous_norm = _tree_infinity_norm(base)
            converged = jnp.array(False)
            inner_state = (
                terms,
                total,
                previous_norm,
                converged,
                jnp.zeros_like(interval_degree),
            )

            def taylor_term(degree, state):
                converged = state[3]
                active = (degree <= interval_degree) & ~converged

                def apply_term(state):
                    terms, total, previous_norm, converged, _ = state

                    def form_term(terms):
                        previous = _tree_take(terms, degree - 1)
                        image = matvec(previous, *parameters)
                        image = jax.tree.map(
                            lambda image_leaf, previous_leaf: (
                                image_leaf - mu * previous_leaf
                            ),
                            image,
                            previous,
                        )
                        term = _tree_scale(step_size / degree, image)
                        return _tree_set(terms, degree, term)

                    terms = jax.lax.cond(
                        degree > formed_degree,
                        form_term,
                        lambda terms: terms,
                        terms,
                    )
                    power = jnp.asarray(step, dtype=step_size.dtype) ** degree
                    term = _tree_scale(power, _tree_take(terms, degree))
                    total = _tree_add(total, term)
                    current_norm = _tree_infinity_norm(term)
                    converged = previous_norm + current_norm <= (
                        tol * _tree_infinity_norm(total)
                    )
                    return terms, total, current_norm, converged, degree

                return jax.lax.cond(
                    active,
                    apply_term,
                    lambda state: state,
                    state,
                )

            # Algorithm 5.2, lines 23--33, incrementally builds equation (5.3).
            terms, total, _, _, last_degree = jax.lax.fori_loop(
                1,
                max_degree + 1,
                taylor_term,
                inner_state,
            )
            formed_degree = jnp.maximum(formed_degree, last_degree)
            total = _tree_scale(jnp.exp(step * step_size * mu), total)
            outputs = _tree_set(outputs, offset + step, total)
            return step + 1, terms, formed_degree, outputs

        step_state = (
            jnp.array(1, dtype=jnp.int32),
            terms,
            jnp.zeros_like(interval_degree),
            outputs,
        )
        _, _, _, outputs = while_loop(step_cond, step_body, step_state)
        next_offset = offset + steps_in_group
        next_base = _tree_take(outputs, next_offset)
        return next_offset, next_base, outputs

    group_state = (jnp.array(0, dtype=jnp.int32), initial, outputs)
    return while_loop(group_cond, group_body, group_state)[2]


def _tree_add(left, right):
    return jax.tree.map(lambda x, y: x + y, left, right)


def _tree_scale(scalar, tree):
    return jax.tree.map(lambda leaf: scalar * leaf, tree)


def _tree_infinity_norm(tree):
    norms = [jnp.max(jnp.abs(leaf)) for leaf in jax.tree.leaves(tree) if leaf.size > 0]
    return jnp.max(jnp.stack(norms))


def _tree_term_buffer(tree, size):
    return jax.tree.map(lambda leaf: jnp.zeros((size, *leaf.shape), leaf.dtype), tree)


def _tree_output_buffer(tree, size):
    return jax.tree.map(lambda leaf: jnp.zeros((size, *leaf.shape), leaf.dtype), tree)


def _tree_set(tree, index, value):
    return jax.tree.map(lambda leaves, leaf: leaves.at[index].set(leaf), tree, value)


def _tree_take(tree, index):
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _tree_prepend(first, rest):
    return jax.tree.map(
        lambda first_leaf, rest_leaf: jnp.concatenate(
            (first_leaf[jnp.newaxis], rest_leaf),
            axis=0,
        ),
        first,
        rest,
    )


def _tree_nan_times(tree, size):
    return jax.tree.map(
        lambda leaf: jnp.full((size, *leaf.shape), jnp.nan, leaf.dtype),
        tree,
    )
