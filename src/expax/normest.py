"""Composable estimators for operator norms."""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from expax._operator import _linear_adjoint, _validate_vector_space


def onenormest(*, block_size=2, max_steps=5):
    """Construct a block 1-norm estimator and its scalar-matvec cost model.

    The estimator implements Higham--Tisseur Algorithm 2.4. Its reported cost is
    the ``4 * block_size`` model used in Al-Mohy--Higham equation (3.12).
    """
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 2:
        raise ValueError("max_steps must be an integer of at least two")

    return _estimate_norm_factory(block_size, max_steps), 4 * block_size


def _sign_round_up(values):
    magnitudes = jnp.abs(values)
    sign = values / magnitudes
    return jnp.where(magnitudes == 0, jnp.ones_like(sign), sign)


def _is_parallel_to_any(vector, others):
    inner_products = jax.vmap(lambda v: jnp.vdot(v, vector), in_axes=0)(others)
    return jnp.any(jnp.isclose(jnp.abs(inner_products), vector.size))


def _all_columns_parallel(left, right):
    inner_products = jnp.abs(jnp.conj(left) @ right.T)
    return jnp.all(jnp.any(jnp.isclose(inner_products, left.shape[-1]), axis=-1))


def _sample_sign_vector(key, size, dtype):
    key, sample_key = jax.random.split(key)
    real_dtype = jnp.real(jnp.zeros((), dtype=dtype)).dtype
    sample = jax.random.rademacher(sample_key, (size,), dtype=real_dtype)
    return key, sample.astype(dtype)


def _resample_until_independent(key, vector, others):
    def cond_fun(state):
        _, candidate = state
        return _is_parallel_to_any(candidate, others)

    def body_fun(state):
        next_key, _ = state
        return _sample_sign_vector(next_key, vector.size, vector.dtype)

    return jax.lax.while_loop(cond_fun, body_fun, (key, vector))


def _initial_block(key, size, block_size, dtype):
    block = jnp.ones((block_size, size), dtype=dtype)
    for column in range(1, block_size):
        key, candidate = _sample_sign_vector(key, size, dtype)
        key, candidate = _resample_until_independent(key, candidate, block[:column])
        block = block.at[column].set(candidate)
    return key, block / size


def _resample_parallel_columns(key, block, previous):
    for column in range(block.shape[0]):
        forbidden = jnp.concatenate((block[:column], previous), axis=0)
        key, candidate = _resample_until_independent(key, block[column], forbidden)
        block = block.at[column].set(candidate)
    return key, block


def _estimate_norm_factory(block_size, max_steps):
    def estimate_norm(matvec, v_like, key, *parameters):
        flat_like, unravel = _validate_vector_space(v_like)
        size = flat_like.size
        if block_size >= size:
            raise ValueError(
                "block_size must be smaller than the vector-space dimension"
            )

        def matvec_flat(vector):
            return ravel_pytree(matvec(unravel(vector), *parameters))[0]

        vecmat_flat_adjoint = _linear_adjoint(matvec_flat, flat_like)
        matmat = jax.vmap(matvec_flat)
        matmat_adjoint = jax.vmap(lambda v: vecmat_flat_adjoint(v)[0])

        key, block = _initial_block(key, size, block_size, flat_like.dtype)
        real_dtype = jnp.real(flat_like).dtype
        estimate = jnp.zeros((), dtype=real_dtype)
        previous_signs = jnp.zeros_like(block)
        indices = jnp.zeros((block_size,), dtype=jnp.int32)
        visited = jnp.zeros((size,), dtype=bool)
        done = jnp.array(False)

        def iteration(step, state):
            done = state[-1]

            def active_iteration(state):
                (
                    block,
                    estimate,
                    previous_signs,
                    indices,
                    visited,
                    key,
                    _,
                ) = state
                image = matmat(block)
                column_norms = jnp.sum(jnp.abs(image), axis=-1)
                candidate = jnp.max(column_norms)
                no_improvement = (step > 0) & (candidate <= estimate)

                def stop_without_adjoint(_):
                    return (
                        block,
                        estimate,
                        previous_signs,
                        indices,
                        visited,
                        key,
                        jnp.array(True),
                    )

                def continue_iteration(_):
                    next_estimate = jnp.maximum(estimate, candidate)

                    def stop_after_final_forward(_):
                        return (
                            block,
                            next_estimate,
                            previous_signs,
                            indices,
                            visited,
                            key,
                            jnp.array(True),
                        )

                    def process_signs(_):
                        best_column = jnp.argmax(column_norms)
                        best_index = indices[best_column]
                        signs = _sign_round_up(image)
                        signs_repeated = (step > 0) & _all_columns_parallel(
                            signs, previous_signs
                        )

                        def stop_on_repeated_signs(_):
                            return (
                                block,
                                next_estimate,
                                previous_signs,
                                indices,
                                visited,
                                key,
                                jnp.array(True),
                            )

                        def continue_with_adjoint(_):
                            next_key, signs_unique = _resample_parallel_columns(
                                key, signs, previous_signs
                            )
                            adjoint_image = matmat_adjoint(signs_unique)
                            row_norms = jnp.max(jnp.abs(adjoint_image), axis=0)
                            ranked = jnp.argsort(-row_norms, stable=True)
                            repeated_best = (step > 0) & jnp.isclose(
                                jnp.max(row_norms), row_norms[best_index]
                            )
                            top_already_visited = jnp.all(visited[ranked[:block_size]])
                            should_stop = repeated_best | top_already_visited

                            ranked_visited = visited[ranked]
                            unseen_first = jnp.argsort(ranked_visited, stable=True)
                            selected = ranked[unseen_first[:block_size]].astype(
                                indices.dtype
                            )
                            next_block = jax.nn.one_hot(
                                selected, size, dtype=flat_like.dtype
                            )
                            next_visited = visited.at[selected].set(True)
                            return (
                                next_block,
                                next_estimate,
                                signs_unique,
                                selected,
                                next_visited,
                                next_key,
                                should_stop,
                            )

                        return jax.lax.cond(
                            signs_repeated,
                            stop_on_repeated_signs,
                            continue_with_adjoint,
                            operand=None,
                        )

                    return jax.lax.cond(
                        step == max_steps - 1,
                        stop_after_final_forward,
                        process_signs,
                        operand=None,
                    )

                return jax.lax.cond(
                    no_improvement,
                    stop_without_adjoint,
                    continue_iteration,
                    operand=None,
                )

            return jax.lax.cond(done, lambda state: state, active_iteration, state)

        state = (
            block,
            estimate,
            previous_signs,
            indices,
            visited,
            key,
            done,
        )
        _, estimate, _, _, _, _, _ = jax.lax.fori_loop(0, max_steps, iteration, state)
        return jax.lax.stop_gradient(jnp.maximum(estimate, 0))

    return estimate_norm
