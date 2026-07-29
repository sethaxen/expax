import jax
import jax.numpy as jnp
from matfree import stochtrace

from expax._operator import _validate_vector_space


def _default_trace_estimator(matvec, v_like, key, *parameters):
    flat_like, _ = _validate_vector_space(v_like)
    dimension = flat_like.size
    if dimension < 3:
        raise ValueError("XTrace requires a vector-space dimension of at least three")

    num_samples = 2 if dimension >= 5 else 1
    sampler = stochtrace.sampler_normal(v_like, num=num_samples)
    integrand = stochtrace.leave_one_out_xtrace()
    estimate = stochtrace.estimator_leave_one_out(integrand, sampler)
    return estimate(matvec, key, *parameters)


def _build_operator_plan(
    times,
    matvec,
    parameters,
    *,
    v_like,
    key,
    trace_estimator,
    norm_estimator,
    theta,
    max_degree,
):
    flat_like, _ = _validate_vector_space(v_like)
    dimension = flat_like.size
    p_max = _compute_p_max(max_degree)
    theta = jnp.asarray(theta, dtype=jnp.real(flat_like).dtype)

    stopped_parameters = jax.tree.map(jax.lax.stop_gradient, parameters)
    stopped_v_like = jax.tree.map(
        lambda leaf: jnp.zeros_like(jax.lax.stop_gradient(leaf)), v_like
    )
    keys = jax.random.split(key, p_max + 2)
    trace = trace_estimator(matvec, stopped_v_like, keys[0], *stopped_parameters)
    candidate_mu = trace / dimension
    mu = jnp.where(jnp.isfinite(candidate_mu), candidate_mu, 0)

    def shifted_matvec(vector, *parameters):
        image = matvec(vector, *parameters)
        return jax.tree.map(
            lambda image_leaf, vector_leaf: image_leaf - mu * vector_leaf,
            image,
            vector,
        )

    estimate_norm, estimator_cost = norm_estimator
    norm = estimate_norm(shifted_matvec, stopped_v_like, keys[1], *stopped_parameters)

    def estimate_power_norms():
        root_norms = []
        for power in range(2, p_max + 2):

            def powered_matvec(vector, *parameters, _power=power):
                for _ in range(_power):
                    vector = shifted_matvec(vector, *parameters)
                return vector

            power_norm = estimate_norm(
                powered_matvec,
                stopped_v_like,
                keys[power],
                *stopped_parameters,
            )
            root_norms.append(power_norm ** (1 / power))

        root_norms = jnp.stack(root_norms)
        alpha = jnp.maximum(root_norms[:-1], root_norms[1:])
        matrix = alpha[:, None] / theta[None, :]
        powers = jnp.arange(2, p_max + 1)[:, None]
        degrees = jnp.arange(1, max_degree + 1)[None, :]
        valid = degrees >= powers * (powers - 1) - 1
        return jnp.where(valid, matrix, 0)

    if times is None:
        use_norm = jnp.array(False)
        matrix = estimate_power_norms()
    else:
        scale = jnp.max(jnp.abs(jax.lax.stop_gradient(jnp.asarray(times))))
        use_norm = _condition_3_13(scale, norm, estimator_cost, theta, max_degree)
        matrix = jax.lax.cond(
            use_norm,
            lambda: jnp.zeros((p_max - 1, max_degree), dtype=norm.dtype),
            estimate_power_norms,
        )

    return jax.tree.map(jax.lax.stop_gradient, (mu, norm, matrix, use_norm))


def _make_selector(norm, matrix, use_norm, *, theta, max_scaling):
    p_max = matrix.shape[-2] + 1
    max_degree = matrix.shape[-1]
    theta = jnp.asarray(theta, dtype=jnp.asarray(norm).dtype)
    powers = jnp.arange(2, p_max + 1)[:, None]
    degrees = jnp.arange(1, max_degree + 1)[None, :]
    valid = degrees >= powers * (powers - 1) - 1

    def select(times):
        stopped_times = jax.lax.stop_gradient(jnp.asarray(times))
        selected = jax.lax.cond(
            use_norm,
            lambda: _select_from_norm(stopped_times, norm, theta),
            lambda: _select_from_matrix(stopped_times, matrix, valid),
        )
        degree, scaling = selected
        selection_valid = jnp.isfinite(scaling)
        if max_scaling is not None:
            selection_valid &= scaling <= max_scaling
        return jax.tree.map(jax.lax.stop_gradient, (degree, scaling, selection_valid))

    return select


def _compute_p_max(max_degree):
    return max(p for p in range(1, max_degree + 2) if p * (p - 1) <= max_degree + 1)


def _condition_3_13(scale, norm, estimator_cost, theta, max_degree):
    p_max = _compute_p_max(max_degree)
    power_cost = estimator_cost * sum(range(2, p_max + 2))
    threshold = theta[max_degree - 1] * power_cost / max_degree
    return jnp.abs(scale) * norm <= threshold


def _take_last_axis(values, indices):
    return jnp.take_along_axis(values, indices[..., None], axis=-1)[..., 0]


def _select_from_norm(scales, norm, theta):
    scales = jnp.abs(jnp.asarray(scales))
    theta = jnp.asarray(theta)
    degrees = jnp.arange(1, theta.size + 1)
    scaled_norm = scales[..., None] * norm
    candidates = jnp.ceil(scaled_norm / theta)
    costs = candidates * degrees
    best = jnp.argmin(costs, axis=-1)
    selected_degrees = best + 1
    selected_scalings = _take_last_axis(candidates, best)
    is_zero = scales * norm == 0
    return (
        jnp.where(is_zero, 0, selected_degrees),
        jnp.where(is_zero, 1.0, selected_scalings),
    )


def _select_from_matrix(scales, matrix, valid):
    scales = jnp.abs(jnp.asarray(scales))
    matrix = jnp.asarray(matrix)
    valid = jnp.asarray(valid)
    degrees = jnp.arange(1, matrix.shape[-1] + 1)
    candidates = jnp.maximum(jnp.ceil(scales[..., None, None] * matrix), 1)
    candidates = jnp.where(valid, candidates, jnp.inf)
    candidates_by_degree = jnp.min(candidates, axis=-2)
    costs = candidates_by_degree * degrees
    best = jnp.argmin(costs, axis=-1)
    selected_degrees = best + 1
    selected_scalings = _take_last_axis(candidates_by_degree, best)
    is_zero = scales == 0
    return (
        jnp.where(is_zero, 0, selected_degrees),
        jnp.where(is_zero, 1.0, selected_scalings),
    )
