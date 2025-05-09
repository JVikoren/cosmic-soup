import jax
import jax.numpy as jnp


def initialize_particles(key, num_particles):
    """Initialize particle positions and velocities."""
    positions = jax.random.uniform(key, (num_particles, 2), minval=-1.0, maxval=1.0)
    velocities = jnp.zeros_like(positions)
    return positions, velocities


def update_particles(
    positions, velocities, interaction_strength=1.0, noise_scale=0.01, dt=0.01
):
    """Simple update: particles attract each other and have random noise."""
    diffs = positions[:, None, :] - positions[None, :, :]
    distances = jnp.linalg.norm(
        diffs + 1e-5, axis=-1
    )  # small epsilon to avoid divide-by-zero
    forces = -interaction_strength * diffs / (distances[..., None] ** 3)

    net_forces = jnp.sum(forces, axis=1)
    noise = noise_scale * jax.random.normal(jax.random.PRNGKey(0), positions.shape)

    new_velocities = velocities + (net_forces + noise) * dt
    new_positions = positions + new_velocities * dt

    return new_positions, new_velocities
