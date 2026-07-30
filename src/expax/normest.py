"""Composable estimators for operator norms."""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from expax._operator import _linear_adjoint, _validate_vector_space


def onenormest(*, block_size=2, max_steps=5):
    """Construct a block 1-norm estimator and its scalar-matvec cost model.

    The estimator implements Higham--Tisseur Algorithm 2.4. Each batch applies
    ``block_size`` vectors in parallel, allowing matrix-backed operators to expose
    matrix--matrix kernels analogous to level-3 BLAS. Its reported scalar-matvec
    cost is the ``4 * block_size`` model used in Al-Mohy--Higham equation (3.12).
    """
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 2:
        raise ValueError("max_steps must be an integer of at least two")

    return _onenormest(block_size, max_steps), 4 * block_size


def _sign_round_up(values):
    magnitudes = jnp.abs(values)
    return jnp.where(magnitudes == 0, 1.0, values / magnitudes)


def _parallel_vectors(left, right):
    return jnp.isclose(jnp.abs(jnp.inner(left, right)), left.shape[-1])


def _all_vectors_parallel(left, right):
    return jnp.all(jnp.any(_parallel_vectors(left, right), axis=-1))


def _vectors_needing_resampling(block, previous):
    block_size = block.shape[0]
    vector = jnp.arange(block_size)
    earlier = vector < vector[:, None]
    comparisons = jnp.concatenate(
        (
            earlier,
            jnp.ones((block_size, previous.shape[0]), dtype=bool),
        ),
        axis=1,
    )
    others = jnp.concatenate((block, previous), axis=0)
    return jnp.any(_parallel_vectors(block, others) & comparisons, axis=-1)


def _sample_real_signs(key, size, dtype):
    rdtype = jnp.dtype(dtype).type(0).real.dtype
    sample = jax.random.rademacher(key, size, dtype=rdtype)
    return sample.astype(dtype)


def _initial_block(key, size, block_size, dtype):
    key, sample_key, resample_key = jax.random.split(key, 3)
    samples = _sample_real_signs(sample_key, (block_size - 1, size), dtype=dtype)
    block = jnp.concatenate((jnp.ones((1, size), dtype=dtype), samples), axis=0)
    previous = jnp.empty((0, size), dtype=dtype)
    block = _resample_parallel_vectors(resample_key, block, previous)
    return key, block / size


def _resample_parallel_vectors(key, block, previous):
    needs_resampling = _vectors_needing_resampling(block, previous)

    def cond_fun(state):
        _, _, needs_resampling = state
        return jnp.any(needs_resampling)

    def body_fun(state):
        next_key, current, needs_resampling = state
        next_key, sample_key = jax.random.split(next_key)
        samples = _sample_real_signs(sample_key, current.shape, current.dtype)
        current = jnp.where(needs_resampling[:, None], samples, current)
        return (
            next_key,
            current,
            _vectors_needing_resampling(current, previous),
        )

    _, block, _ = jax.lax.while_loop(cond_fun, body_fun, (key, block, needs_resampling))
    return block


def _onenormest(block_size, max_steps):
    def estimate_norm(matvec, v_like, key, *parameters):
        flat_like, unravel = _validate_vector_space(v_like)
        real_dtype = jnp.real(flat_like).dtype
        check_sign_parallelism = not jnp.issubdtype(
            flat_like.dtype, jnp.complexfloating
        )
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
                vector_norms = jnp.sum(jnp.abs(image), axis=-1)
                candidate = jnp.max(vector_norms)
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
                        best_vector = jnp.argmax(vector_norms)
                        best_index = indices[best_vector]
                        signs = _sign_round_up(image)
                        if check_sign_parallelism:
                            signs_repeated = (step > 0) & _all_vectors_parallel(
                                signs, previous_signs
                            )
                        else:
                            signs_repeated = jnp.array(False)

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
                            if check_sign_parallelism:
                                next_key, resample_key = jax.random.split(key)
                                signs_unique = _resample_parallel_vectors(
                                    resample_key, signs, previous_signs
                                )
                            else:
                                next_key, signs_unique = key, signs
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
