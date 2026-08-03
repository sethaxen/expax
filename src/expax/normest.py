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
    """Construct a block 1-norm estimator for linear operators.

    The returned estimator computes a lower bound for the induced 1-norm of a
    linear operator. It processes a block of `block_size` test vectors
    together to improve reliability.

    ```{note}
    If the average cost of the estimator would exceed the cost of materializing
    the operator and computing its exact 1-norm, the exact 1-norm is computed
    instead.
    ```

    **Parameters**

    - `block_size`: Number of test vectors processed in parallel during each
      operator or adjoint application (default: 2). Must be a positive integer.
      On the GPU, increasing the block size may provide significantly better
      estimates with very little increase in runtime.
    - `max_steps`: Maximum number of block power-iteration steps (default: 5).
      Must be an integer greater than 1 but typically should not be changed.

    **Returns**

    A pair `(estimate_norm, cost)`. `cost` of a random matrix for `max_steps=5`
    is typically `4 * block_size` scalar `matvec` operations. The estimator is
    called as `estimate_norm(matvec, v_like, key, *parameters)` and accepts:

    - `matvec`: A callable `matvec(vector, *parameters)` representing a linear
      operator. Its input and output are PyTrees with the structure described by
      `v_like`.
    - `v_like`: A nonempty PyTree of JAX arrays describing the vector-space
      structure, leaf shapes, and common inexact dtype. Its values are ignored.
    - `key`: A JAX random key for (re)sampling test vectors.
    - `*parameters`: Runtime arguments forwarded unchanged to `matvec` after the
      vector argument.

    `estimate_norm` returns the estimated operator 1-norm as a scalar JAX array.

    **References**

    Nicholas J. Higham and Françoise Tisseur, "A Block Algorithm for Matrix
    1-Norm Estimation, with an Application to 1-Norm Pseudospectra," *SIAM
    Journal on Matrix Analysis and Applications*, 21(4), 1185--1201, 2000.
    [10.1137/S0895479899356080](https://doi.org/10.1137/S0895479899356080)
    [eprint](https://eprints.maths.manchester.ac.uk/id/eprint/321)
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


def _sign_round_up(values: _VectorBatch) -> _VectorBatch:
    """Return elementwise signs, assigning a sign of one to zero entries."""
    magnitudes = jnp.abs(values)
    return jnp.where(magnitudes == 0, 1.0, values / magnitudes)


def _parallel_sign_vectors(
    left: Inexact[Array, "left dim"],
    right: Inexact[Array, "right dim"],
) -> Bool[Array, "left right"]:
    """Identify parallel pairs of real-valued sign vectors."""
    return jnp.isclose(jnp.abs(jnp.inner(left, right)), left.shape[-1])


def _all_sign_vectors_parallel(
    left: Inexact[Array, "left dim"],
    right: Inexact[Array, "right dim"],
) -> Bool[Array, ""]:
    """Return whether every left sign vector is parallel to a right sign vector."""
    return jnp.all(jnp.any(_parallel_sign_vectors(left, right), axis=-1))


def _sign_vectors_to_resample(
    block: _VectorBatch,
    previous: Inexact[Array, "prev dim"],
) -> Bool[Array, " block"]:
    """Flag sign vectors parallel to an earlier current or previous sign vector."""
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
    return jnp.any(_parallel_sign_vectors(block, others) & comparisons, axis=-1)


def _sample_real_signs(
    key: PRNGKeyArray,
    size: tuple[int, ...],
    dtype: DTypeLike,
) -> Inexact[Array, "*shape"]:
    """Sample real Rademacher signs represented in the requested dtype."""
    rdtype = jnp.dtype(dtype).type(0).real.dtype
    sample = jax.random.rademacher(key, size, dtype=rdtype)
    return sample.astype(dtype)


def _initial_block(
    key: PRNGKeyArray,
    size: int,
    block_size: int,
    dtype: DTypeLike,
) -> _VectorBatch:
    """Construct the normalized initial batch of nonparallel test vectors."""
    sample_key, resample_key = jax.random.split(key, 2)
    samples = _sample_real_signs(sample_key, (block_size - 1, size), dtype=dtype)
    block = jnp.concatenate((jnp.ones((1, size), dtype=dtype), samples), axis=0)
    previous = jnp.empty((0, size), dtype=dtype)
    block = _resample_parallel_sign_vectors(resample_key, block, previous)
    return block / size


def _resample_parallel_sign_vectors(
    key: PRNGKeyArray,
    block: _VectorBatch,
    previous: Inexact[Array, "prev dim"],
) -> _VectorBatch:
    """Resample vectors until none is parallel to an earlier or previous vector."""
    needs_resampling = _sign_vectors_to_resample(block, previous)

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
            _sign_vectors_to_resample(current, previous),
        )

    _, block, _ = jax.lax.while_loop(cond_fun, body_fun, (key, block, needs_resampling))
    return block


class _Block1NormEstimatorState(NamedTuple):
    """Bundle the complete loop-carried state across power-iteration steps.

    Attributes:
        test_vectors: Vectors to which the matvec will be applied next.
        estimate: Running 1-norm estimate.
        previous_result_probes: Result probes retained for the next parallelism check.
        test_vector_indices: Basis indices represented by ``test_vectors``, with
            placeholder values for the initial non-basis batch.
        visited_basis_indices: Mask of basis vectors selected in prior steps.
        key: Random key reserved for resampling parallel result probes.
        done: Whether subsequent steps should leave the state unchanged.
    """

    test_vectors: _VectorBatch
    estimate: _RealScalar
    previous_result_probes: _VectorBatch
    test_vector_indices: Int[Array, " block"]
    visited_basis_indices: Bool[Array, " dim"]
    key: PRNGKeyArray
    done: Bool[Array, ""]


class _ForwardNormEstimate(NamedTuple):
    """Bundle a result-vector batch with the estimator state it updated.

    Attributes:
        step: Zero-based index of the current power-iteration step.
        state: Estimator state whose running estimate includes this batch.
        result_vectors: Forward-operator values at the current test vectors.
        result_onenorms: One-norm of each result vector.
    """

    step: Int[Array, ""]
    state: _Block1NormEstimatorState
    result_vectors: _VectorBatch
    result_onenorms: Real[Array, " block"]


class _ResultProbeIteration(NamedTuple):
    """Bundle result probes used for parallelism and test-vector selection.

    Attributes:
        step: Zero-based index of the current power-iteration step.
        state: Estimator state produced by the forward-operator batch.
        result_probes: Norming probes for the forward result vectors, possibly
            replaced by random probes to avoid repeated pullbacks.
        best_basis_index: Basis index associated with the strongest result,
            with a placeholder value during the initial non-basis step.
    """

    step: Int[Array, ""]
    state: _Block1NormEstimatorState
    result_probes: _VectorBatch
    best_basis_index: Int[Array, ""]


def _all_result_probes_parallel_to_previous(
    probe_iteration: _ResultProbeIteration,
) -> Bool[Array, ""]:
    """Check whether every result probe repeats a previous direction."""
    return (probe_iteration.step > 0) & _all_sign_vectors_parallel(
        probe_iteration.result_probes,
        probe_iteration.state.previous_result_probes,
    )


def _resample_parallel_result_probes(
    probe_iteration: _ResultProbeIteration,
) -> _ResultProbeIteration:
    """Advance the key and replace result probes that repeat another direction."""
    state = probe_iteration.state
    next_key, resample_key = jax.random.split(state.key)
    result_probes = _resample_parallel_sign_vectors(
        resample_key,
        probe_iteration.result_probes,
        state.previous_result_probes,
    )
    return probe_iteration._replace(
        state=state._replace(key=next_key),
        result_probes=result_probes,
    )


def _finish_norm_estimation(
    state: _Block1NormEstimatorState,
) -> _Block1NormEstimatorState:
    """Mark a block 1-norm estimator state as finished."""
    return state._replace(done=jnp.array(True))


def _select_unvisited_basis_indices(
    basis_scores: Real[Array, " dim"],
    visited_basis_indices: Bool[Array, " dim"],
    num_test_vectors: int,
    index_dtype: DTypeLike,
) -> tuple[Int[Array, " block"], Bool[Array, ""]]:
    """Select top unvisited basis indices and flag when every top choice was visited."""
    ranked = jnp.argsort(basis_scores, stable=True, descending=True)
    top_basis_indices_visited = jnp.all(
        visited_basis_indices[ranked[:num_test_vectors]]
    )
    unseen_first = jnp.argsort(
        visited_basis_indices[ranked],
        stable=True,
    )
    selected_basis_indices = ranked[unseen_first[:num_test_vectors]].astype(index_dtype)
    return selected_basis_indices, top_basis_indices_visited


def _score_basis_vectors(
    result_probes: _VectorBatch,
    matvec_batch_adjoint: _BatchedMatvec,
) -> Real[Array, " dim"]:
    """Pull result probes back to test probes and score each basis vector."""
    test_probes = matvec_batch_adjoint(result_probes)
    return jnp.max(jnp.abs(test_probes), axis=0)


def _build_basis_test_vectors(
    basis_indices: Int[Array, " block"],
    size: int,
    dtype: DTypeLike,
) -> _VectorBatch:
    """Construct one-hot test vectors for selected basis indices."""
    return jax.nn.one_hot(basis_indices, size, dtype=dtype)


def _select_test_vectors(
    probe_iteration: _ResultProbeIteration,
    *,
    matvec_batch_adjoint: _BatchedMatvec,
) -> _Block1NormEstimatorState:
    """Choose the next basis test vectors from result-probe pullbacks."""
    state = probe_iteration.state
    basis_scores = _score_basis_vectors(
        probe_iteration.result_probes,
        matvec_batch_adjoint,
    )
    selected_basis_indices, top_basis_indices_visited = _select_unvisited_basis_indices(
        basis_scores,
        state.visited_basis_indices,
        state.test_vectors.shape[0],
        state.test_vector_indices.dtype,
    )
    best_basis_still_optimal = (probe_iteration.step > 0) & jnp.isclose(
        jnp.max(basis_scores),
        basis_scores[probe_iteration.best_basis_index],
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
        previous_result_probes=probe_iteration.result_probes,
        test_vector_indices=selected_basis_indices,
        visited_basis_indices=next_visited_basis_indices,
        done=best_basis_still_optimal | top_basis_indices_visited,
    )


def _select_test_vectors_real(
    probe_iteration: _ResultProbeIteration,
    *,
    matvec_batch_adjoint: _BatchedMatvec,
) -> _Block1NormEstimatorState:
    """Select test vectors optionally with resampled probes.

    If all probes are repeated, terminate the estimation.
    """

    def choose_after_resampling(probe_iteration):
        probe_iteration = _resample_parallel_result_probes(probe_iteration)
        return _select_test_vectors(
            probe_iteration,
            matvec_batch_adjoint=matvec_batch_adjoint,
        )

    return jax.lax.cond(
        _all_result_probes_parallel_to_previous(probe_iteration),
        lambda iteration: _finish_norm_estimation(iteration.state),
        choose_after_resampling,
        probe_iteration,
    )


def _choose_next_test_vectors(
    forward_estimate: _ForwardNormEstimate,
    *,
    matvec_batch_adjoint: _BatchedMatvec,
) -> _Block1NormEstimatorState:
    """Derive result probes and use their pullbacks to choose test vectors."""
    # result probes are here the subgradients of the 1-norm of the result vectors
    result_probes = _sign_round_up(forward_estimate.result_vectors)
    best_test_vector = jnp.argmax(forward_estimate.result_onenorms)
    best_basis_index = forward_estimate.state.test_vector_indices[best_test_vector]
    probe_iteration = _ResultProbeIteration(
        step=forward_estimate.step,
        state=forward_estimate.state,
        result_probes=result_probes,
        best_basis_index=best_basis_index,
    )

    choose_from_adjoint = partial(
        _select_test_vectors,
        matvec_batch_adjoint=matvec_batch_adjoint,
    )

    if jnp.issubdtype(result_probes.dtype, jnp.complexfloating):
        return choose_from_adjoint(probe_iteration)
    return _select_test_vectors_real(
        probe_iteration,
        matvec_batch_adjoint=matvec_batch_adjoint,
    )


def _estimate_onenorm_from_test_vectors(
    state: _Block1NormEstimatorState,
    *,
    step: Int[Array, ""],
    matvec_batch: _BatchedMatvec,
    matvec_batch_adjoint: _BatchedMatvec,
    max_steps: int,
) -> _Block1NormEstimatorState:
    """Apply the operator, update the estimate, then stop or choose test vectors."""
    result_vectors = matvec_batch(state.test_vectors)
    result_onenorms = jnp.linalg.vector_norm(result_vectors, ord=1, axis=-1)
    estimate = jnp.max(result_onenorms)
    no_improvement = (step > 0) & (estimate <= state.estimate)
    forward_estimate = _ForwardNormEstimate(
        step=step,
        state=state._replace(estimate=jnp.maximum(state.estimate, estimate)),
        result_vectors=result_vectors,
        result_onenorms=result_onenorms,
    )
    choose_next_test_vectors = partial(
        _choose_next_test_vectors,
        matvec_batch_adjoint=matvec_batch_adjoint,
    )
    stop_after_forward = no_improvement | (step == max_steps - 1)
    return jax.lax.cond(
        stop_after_forward,
        lambda est: _finish_norm_estimation(est.state),
        choose_next_test_vectors,
        forward_estimate,
    )


def _block_onenorm_power_iteration_step(
    step: Int[Array, ""],
    state: _Block1NormEstimatorState,
    *,
    matvec_batch: _BatchedMatvec,
    matvec_batch_adjoint: _BatchedMatvec,
    max_steps: int,
) -> _Block1NormEstimatorState:
    """Run one guarded step of the block 1-norm power iteration."""
    estimate_from_test_vectors = partial(
        _estimate_onenorm_from_test_vectors,
        step=step,
        matvec_batch=matvec_batch,
        matvec_batch_adjoint=matvec_batch_adjoint,
        max_steps=max_steps,
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
    """Compute the exact operator 1-norm by materializing the operator."""
    mat = _materialize_operator(matvec_batch, v)
    return jnp.linalg.matrix_norm(mat, ord=1)


def _onenormest(
    block_size: int,
    max_steps: int,
    estimator_cost: int,
) -> Callable[..., _RealScalar]:
    """Return an estimator that switches between exact and block power iteration."""

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
            lambda result_probe: matvec_flat_adjoint(result_probe)[0]
        )

        real_dtype = jnp.real(flat_like).dtype

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
            previous_result_probes=jnp.zeros_like(test_vectors),
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
        )
        state = jax.lax.fori_loop(0, max_steps, power_iteration_step, state)
        return state.estimate

    return estimate_norm
