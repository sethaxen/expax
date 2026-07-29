import jax.numpy as jnp


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
