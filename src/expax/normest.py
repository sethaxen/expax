"""Composable estimators for operator norms."""

from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, DTypeLike, Inexact, Int, PRNGKeyArray, PyTree, Real

from expax._operator import _linear_adjoint, _validate_vector_space

_RealScalar: TypeAlias = Real[Array, ""]
_PyTreeVector: TypeAlias = PyTree[Inexact[Array, "..."]]
_VectorBatch: TypeAlias = Inexact[Array, "block dim"]
_BatchedMatvec: TypeAlias = Callable[[_VectorBatch], _VectorBatch]


def onenormest(
    *, block_size: int = 2, max_steps: int = 5
) -> tuple[Callable[..., _RealScalar], int]:
    """Construct a block 1-norm estimator and its scalar-matvec cost model.

    The estimator implements Higham--Tisseur Algorithm 2.4. Each batch applies
    ``block_size`` vectors in parallel, allowing matrix-backed operators to expose
    matrix--matrix kernels analogous to level-3 BLAS. Whenever the dimension does
    not exceed its reported cost, it instead evaluates every basis vector exactly.
    Its reported scalar-matvec cost is the ``4 * block_size`` model used in
    Al-Mohy--Higham equation (3.12).
    """
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 2:
        raise ValueError("max_steps must be an integer of at least two")

    estimator_cost = 4 * block_size
    return _onenormest(block_size, max_steps, estimator_cost), estimator_cost


def _sign_round_up(
    values: _VectorBatch,
) -> _VectorBatch:
    magnitudes = jnp.abs(values)
    return jnp.where(magnitudes == 0, 1.0, values / magnitudes)


def _parallel_vectors(
    left: Inexact[Array, "left dim"],
    right: Inexact[Array, "right dim"],
) -> Bool[Array, "left right"]:
    return jnp.isclose(jnp.abs(jnp.inner(left, right)), left.shape[-1])


def _all_vectors_parallel(
    left: Inexact[Array, "left dim"],
    right: Inexact[Array, "right dim"],
) -> Bool[Array, ""]:
    return jnp.all(jnp.any(_parallel_vectors(left, right), axis=-1))


def _vectors_needing_resampling(
    block: _VectorBatch,
    previous: Inexact[Array, "prev dim"],
) -> Bool[Array, " block"]:
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


def _sample_real_signs(
    key: PRNGKeyArray,
    size: tuple[int, ...],
    dtype: DTypeLike,
) -> Inexact[Array, "*shape"]:
    rdtype = jnp.dtype(dtype).type(0).real.dtype
    sample = jax.random.rademacher(key, size, dtype=rdtype)
    return sample.astype(dtype)


def _initial_block(
    key: PRNGKeyArray,
    size: int,
    block_size: int,
    dtype: DTypeLike,
) -> _VectorBatch:
    sample_key, resample_key = jax.random.split(key, 2)
    samples = _sample_real_signs(sample_key, (block_size - 1, size), dtype=dtype)
    block = jnp.concatenate((jnp.ones((1, size), dtype=dtype), samples), axis=0)
    previous = jnp.empty((0, size), dtype=dtype)
    block = _resample_parallel_vectors(resample_key, block, previous)
    return block / size


def _resample_parallel_vectors(
    key: PRNGKeyArray,
    block: _VectorBatch,
    previous: Inexact[Array, "prev dim"],
) -> _VectorBatch:
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


def _all_sign_vectors_parallel_to_previous(
    step: int | Int[Array, ""],
    sign_vectors: _VectorBatch,
    previous_sign_vectors: _VectorBatch,
    *,
    check_parallel: bool,
) -> Bool[Array, ""]:
    if not check_parallel:
        return jnp.array(False)
    return (step > 0) & _all_vectors_parallel(
        sign_vectors,
        previous_sign_vectors,
    )


def _resample_parallel_sign_vectors(
    key: PRNGKeyArray,
    sign_vectors: _VectorBatch,
    previous_sign_vectors: _VectorBatch,
    *,
    check_parallel: bool,
) -> tuple[PRNGKeyArray, _VectorBatch]:
    if not check_parallel:
        return key, sign_vectors
    next_key, resample_key = jax.random.split(key)
    sign_vectors = _resample_parallel_vectors(
        resample_key,
        sign_vectors,
        previous_sign_vectors,
    )
    return next_key, sign_vectors


class _Block1NormEstimatorState(NamedTuple):
    test_vectors: _VectorBatch
    estimate: _RealScalar
    previous_sign_vectors: _VectorBatch
    test_vector_indices: Int[Array, " block"]
    visited_basis_indices: Bool[Array, " dim"]
    key: PRNGKeyArray
    done: Bool[Array, ""]


class _ForwardNormEstimate(NamedTuple):
    step: int | Int[Array, ""]
    state: _Block1NormEstimatorState
    response_vectors: _VectorBatch
    response_onenorms: Real[Array, " block"]


class _SignVectorIteration(NamedTuple):
    step: int | Int[Array, ""]
    state: _Block1NormEstimatorState
    sign_vectors: _VectorBatch
    best_basis_index: Int[Array, ""]


def _finish_norm_estimation(
    state: _Block1NormEstimatorState,
) -> _Block1NormEstimatorState:
    return state._replace(done=jnp.array(True))


def _select_unvisited_basis_indices(
    basis_scores: Real[Array, " dim"],
    visited_basis_indices: Bool[Array, " dim"],
    num_test_vectors: int,
    index_dtype: DTypeLike,
) -> tuple[Int[Array, " block"], Bool[Array, ""]]:
    ranked = jnp.argsort(-basis_scores, stable=True)
    top_basis_indices_visited = jnp.all(
        visited_basis_indices[ranked[:num_test_vectors]]
    )
    unseen_first = jnp.argsort(
        visited_basis_indices[ranked],
        stable=True,
    )
    selected_basis_indices = ranked[unseen_first[:num_test_vectors]].astype(index_dtype)
    return selected_basis_indices, top_basis_indices_visited


def _score_basis_vectors_with_adjoint(
    sign_vectors: _VectorBatch,
    matvec_batch_adjoint: _BatchedMatvec,
) -> Real[Array, " dim"]:
    adjoint_products = matvec_batch_adjoint(sign_vectors)
    return jnp.max(jnp.abs(adjoint_products), axis=0)


def _build_basis_test_vectors(
    basis_indices: Int[Array, " block"],
    size: int,
    dtype: DTypeLike,
) -> _VectorBatch:
    return jax.nn.one_hot(basis_indices, size, dtype=dtype)


def _choose_test_vectors_from_adjoint(
    sign_iteration: _SignVectorIteration,
    *,
    matvec_batch_adjoint: _BatchedMatvec,
    check_sign_parallelism: bool,
) -> _Block1NormEstimatorState:
    state = sign_iteration.state
    next_key, sign_vectors = _resample_parallel_sign_vectors(
        state.key,
        sign_iteration.sign_vectors,
        state.previous_sign_vectors,
        check_parallel=check_sign_parallelism,
    )
    basis_scores = _score_basis_vectors_with_adjoint(
        sign_vectors,
        matvec_batch_adjoint,
    )
    selected_basis_indices, top_basis_indices_visited = _select_unvisited_basis_indices(
        basis_scores,
        state.visited_basis_indices,
        state.test_vectors.shape[0],
        state.test_vector_indices.dtype,
    )
    best_basis_still_optimal = (sign_iteration.step > 0) & jnp.isclose(
        jnp.max(basis_scores),
        basis_scores[sign_iteration.best_basis_index],
    )
    next_visited_basis_indices = state.visited_basis_indices.at[
        selected_basis_indices
    ].set(True)
    next_test_vectors = _build_basis_test_vectors(
        selected_basis_indices,
        state.visited_basis_indices.size,
        state.test_vectors.dtype,
    )
    return state._replace(
        test_vectors=next_test_vectors,
        previous_sign_vectors=sign_vectors,
        test_vector_indices=selected_basis_indices,
        visited_basis_indices=next_visited_basis_indices,
        key=next_key,
        done=best_basis_still_optimal | top_basis_indices_visited,
    )


def _choose_next_test_vectors(
    forward_estimate: _ForwardNormEstimate,
    *,
    matvec_batch_adjoint: _BatchedMatvec,
    check_sign_parallelism: bool,
) -> _Block1NormEstimatorState:
    sign_vectors = _sign_round_up(forward_estimate.response_vectors)
    best_test_vector = jnp.argmax(forward_estimate.response_onenorms)
    best_basis_index = forward_estimate.state.test_vector_indices[best_test_vector]
    sign_iteration = _SignVectorIteration(
        step=forward_estimate.step,
        state=forward_estimate.state,
        sign_vectors=sign_vectors,
        best_basis_index=best_basis_index,
    )
    choose_from_adjoint = partial(
        _choose_test_vectors_from_adjoint,
        matvec_batch_adjoint=matvec_batch_adjoint,
        check_sign_parallelism=check_sign_parallelism,
    )
    all_sign_vectors_parallel = _all_sign_vectors_parallel_to_previous(
        forward_estimate.step,
        sign_vectors,
        forward_estimate.state.previous_sign_vectors,
        check_parallel=check_sign_parallelism,
    )
    return jax.lax.cond(
        all_sign_vectors_parallel,
        lambda iter: _finish_norm_estimation(iter.state),
        choose_from_adjoint,
        sign_iteration,
    )


def _estimate_onenorm_from_test_vectors(
    state: _Block1NormEstimatorState,
    *,
    step: int | Int[Array, ""],
    matvec_batch: _BatchedMatvec,
    matvec_batch_adjoint: _BatchedMatvec,
    max_steps: int,
    check_sign_parallelism: bool,
) -> _Block1NormEstimatorState:
    response_vectors = matvec_batch(state.test_vectors)
    response_onenorms = jnp.linalg.norm(response_vectors, ord=1, axis=-1)
    estimate = jnp.max(response_onenorms)
    no_improvement = (step > 0) & (estimate <= state.estimate)
    forward_estimate = _ForwardNormEstimate(
        step=step,
        state=state._replace(estimate=jnp.maximum(state.estimate, estimate)),
        response_vectors=response_vectors,
        response_onenorms=response_onenorms,
    )
    choose_next_test_vectors = partial(
        _choose_next_test_vectors,
        matvec_batch_adjoint=matvec_batch_adjoint,
        check_sign_parallelism=check_sign_parallelism,
    )
    stop_after_forward = no_improvement | (step == max_steps - 1)
    return jax.lax.cond(
        stop_after_forward,
        lambda est: _finish_norm_estimation(est.state),
        choose_next_test_vectors,
        forward_estimate,
    )


def _block_onenorm_power_iteration_step(
    step: int | Int[Array, ""],
    state: _Block1NormEstimatorState,
    *,
    matvec_batch: _BatchedMatvec,
    matvec_batch_adjoint: _BatchedMatvec,
    max_steps: int,
    check_sign_parallelism: bool,
) -> _Block1NormEstimatorState:
    estimate_from_test_vectors = partial(
        _estimate_onenorm_from_test_vectors,
        step=step,
        matvec_batch=matvec_batch,
        matvec_batch_adjoint=matvec_batch_adjoint,
        max_steps=max_steps,
        check_sign_parallelism=check_sign_parallelism,
    )
    return jax.lax.cond(
        state.done,
        lambda state: state,
        estimate_from_test_vectors,
        state,
    )


def _materialize_operator(
    matvec_batch: _BatchedMatvec,
    v: Inexact[Array, " dim"],
) -> Inexact[Array, "dim dim"]:
    """Materialize the operator defined by a batched matvec and a vector."""
    basis_vecs = jnp.eye(v.size, dtype=v.dtype)
    return matvec_batch(basis_vecs).T


def _onenorm_exact(
    matvec_batch: _BatchedMatvec, v: Inexact[Array, " dim"]
) -> _RealScalar:
    mat = _materialize_operator(matvec_batch, v)
    return jnp.linalg.norm(mat, ord=1)


def _onenormest(
    block_size: int,
    max_steps: int,
    estimator_cost: int,
) -> Callable[..., _RealScalar]:
    def estimate_norm(
        matvec: Callable[..., _PyTreeVector],
        v_like: _PyTreeVector,
        key: PRNGKeyArray,
        *parameters: Any,
    ) -> _RealScalar:
        flat_like, unravel = _validate_vector_space(v_like)
        size = flat_like.size

        def matvec_flat(
            vector: Inexact[Array, " dim"],
        ) -> Inexact[Array, " dim"]:
            return ravel_pytree(matvec(unravel(vector), *parameters))[0]

        matvec_batch = jax.vmap(matvec_flat)

        if size <= estimator_cost:
            return _onenorm_exact(matvec_batch, flat_like)

        matvec_flat_adjoint = _linear_adjoint(matvec_flat, flat_like)
        matvec_batch_adjoint = jax.vmap(
            lambda sign_vector: matvec_flat_adjoint(sign_vector)[0]
        )

        real_dtype = jnp.real(flat_like).dtype
        check_sign_parallelism = not jnp.issubdtype(
            flat_like.dtype, jnp.complexfloating
        )

        initial_key, resample_key = jax.random.split(key)

        test_vectors = _initial_block(
            initial_key,
            size,
            block_size,
            flat_like.dtype,
        )
        state = _Block1NormEstimatorState(
            test_vectors=test_vectors,
            estimate=jnp.zeros((), dtype=real_dtype),
            previous_sign_vectors=jnp.zeros_like(test_vectors),
            test_vector_indices=jnp.zeros((block_size,), dtype=jnp.int32),
            visited_basis_indices=jnp.zeros((size,), dtype=bool),
            key=resample_key,
            done=jnp.array(False),
        )
        power_iteration_step = partial(
            _block_onenorm_power_iteration_step,
            matvec_batch=matvec_batch,
            matvec_batch_adjoint=matvec_batch_adjoint,
            max_steps=max_steps,
            check_sign_parallelism=check_sign_parallelism,
        )
        state = jax.lax.fori_loop(0, max_steps, power_iteration_step, state)
        return state.estimate

    return estimate_norm
