from collections import namedtuple
import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
# Removed matplotlib, PIL.Image, PIL.ImageDraw, PIL.ImageFont as direct imports here.
# np2pil from utils will handle PIL.Image if needed by render_lenia_state.

# Import from local utils
from .utils import np2pil, vmap2 # VideoWriter removed

# --- Core Data Structures and Parameters ---

Params = namedtuple('Params', 'mu_k sigma_k w_k mu_g sigma_g c_rep')
Fields = namedtuple('Fields', 'U G R E')

def get_default_lenia_params() -> Params:
    """Returns default parameters for the Lenia simulation."""
    return Params(mu_k=4.0, sigma_k=1.0, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1.0)

# --- Particle Initialization ---

def initialize_lenia_particles(
    key: jax.random.PRNGKey,
    num_particles: int,
    initial_spread: float = 12.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Initializes Lenia particle positions and velocities.

    Args:
        key: JAX random key.
        num_particles: Number of particles to initialize.
        initial_spread: The range over which particles are initially spread.
                        Positions will be in [-initial_spread/2, initial_spread/2].

    Returns:
        A tuple containing:
            - positions: An array of shape (num_particles, 2).
            - velocities: An array of shape (num_particles, 2), initialized to zeros.
    """
    positions = (jax.random.uniform(key, (num_particles, 2)) - 0.5) * initial_spread
    velocities = jnp.zeros_like(positions)
    return positions, velocities

# --- Simulation Physics and Update ---

def peak_f(x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
    """Gaussian-like peak function."""
    return jnp.exp(-((x - mu) / sigma)**2)

def fields_f(p: Params, points: jnp.ndarray, x: jnp.ndarray) -> Fields:
    """Calculates local fields at a point x due to all particles."""
    r = jnp.sqrt(jnp.square(x - points).sum(-1).clip(1e-10))
    U = peak_f(r, p.mu_k, p.sigma_k).sum() * p.w_k
    G = peak_f(U, p.mu_g, p.sigma_g)
    R = p.c_rep / 2 * ((1.0 - r).clip(0.0)**2).sum()
    return Fields(U, G, R, E=R - G)

@functools.partial(jax.jit, static_argnames=()) # Removed static_argnames related to motion_f itself
def motion_f(params: Params, points: jnp.ndarray) -> jnp.ndarray:
    """Calculates the force (negative gradient of energy) on each particle."""
    # `jax.grad` needs a scalar output function. fields_f(...).E is scalar for a single x.
    # We vmap grad_E over all points.
    grad_E_at_point = jax.grad(lambda x_point: fields_f(params, points, x_point).E)
    return -jax.vmap(grad_E_at_point)(points)

def update_lenia_particles(
    lenia_params: Params,
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    dt: float,
    extent: float  # New parameter for boundary wrapping
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Updates particle positions and velocities for a single time step,
    with periodic boundary conditions (wrapping).

    Args:
        lenia_params: The simulation parameters (Params namedtuple).
        positions: Current particle positions.
        velocities: Current particle velocities.
        dt: Time step size.
        extent: The spatial extent of the simulation area. Particles wrap around
                this extent (e.g., if extent is 20, space is [-10, 10]).

    Returns:
        A tuple containing:
            - new_positions: Updated particle positions, wrapped to the extent.
            - new_velocities: Updated particle velocities.
    """
    forces = motion_f(lenia_params, positions)
    
    # Euler integration
    new_velocities = velocities + forces * dt
    candidate_positions = positions + new_velocities * dt
    
    # Apply boundary wrapping
    half_extent = extent / 2.0
    # Current range is [-half_extent, half_extent]
    # Shift to [0, extent], apply modulo, then shift back to [-half_extent, half_extent]
    wrapped_x = (candidate_positions[:, 0] + half_extent) % extent - half_extent
    wrapped_y = (candidate_positions[:, 1] + half_extent) % extent - half_extent
    new_positions = jnp.stack([wrapped_x, wrapped_y], axis=-1)
    
    return new_positions, new_velocities

# --- Visualization ---

def lerp(x: jnp.ndarray, a: list[float], b: list[float]) -> jnp.ndarray:
    """Linear interpolation."""
    return jnp.array(a) * (1.0 - x) + jnp.array(b) * x

def cmap_e(e: jnp.ndarray) -> jnp.ndarray:
    """Colormap for energy field E."""
    # Adjusted for potential broadcasting issues if e is scalar
    e_stack = jnp.stack([e, -e], axis=-1) if e.ndim > 0 else jnp.array([e, -e])
    return 1.0 - e_stack.clip(0) @ jnp.array([[0.3, 1.0, 1.0], [1.0, 0.3, 1.0]])


def cmap_ug(u: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """Colormap for U and G fields."""
    vis_u_component = lerp(u[..., None], [0.1, 0.1, 0.3], [0.2, 0.7, 1.0])
    return lerp(g[..., None], vis_u_component, [1.17, 0.91, 0.13])

@functools.partial(jax.jit, static_argnames=['canvas_width', 'show_field_types', 'show_color_guides'])
def render_lenia_state(
    params: Params,
    points: jnp.ndarray,
    extent: float,
    canvas_width: int = 400,
    show_field_types: bool = False,
    show_color_guides: bool = True
) -> jnp.ndarray:
    """Renders the current Lenia particle state to a NumPy array (image).

    Args:
        params: Lenia simulation parameters (Params namedtuple).
        points: Current particle positions (num_particles, 2).
        extent: The spatial extent of the visualization area (e.g., if extent is 20,
                visualizes from -10 to 10).
        canvas_width: Width of the output image in pixels.
        show_field_types: If True, shows U and G fields alongside E field.
        show_color_guides: If True, adds color guide bars to the image.

    Returns:
        A JAX array representing the rendered image (height, width, 3), dtype=np.uint8.
    """
    # Create a grid of points to evaluate fields on
    # Extent defines the view: e.g. if extent is 1.0, view is [-0.5, 0.5]
    # For consistency with how extent might be used (e.g. max particle distance),
    # let's assume extent is the total width/height of the view.
    half_extent = extent / 2.0
    grid_coords = jnp.linspace(-half_extent, half_extent, canvas_width)
    xy = jnp.stack(jnp.meshgrid(grid_coords, grid_coords), axis=-1)

    # Calculate field values across the grid
    # Note: vmap2 from utils applies jax.vmap twice.
    # fields_f_partial = functools.partial(fields_f, params, points)
    # This makes fields_f_partial a function of a single 'x' (a 2D coordinate).
    # vmap(vmap(fields_f_partial)) will apply it to each point in the xy grid.
    
    # Redefine for clarity in JIT compilation context
    def single_point_field_calc(coord_x, coord_y):
        return fields_f(params, points, jnp.array([coord_x, coord_y]))

    # Apply vmap over the grid, vmap2 might be overly general if xy is already a meshgrid result
    # Let's use vmap directly for applying to the grid points
    # fields_on_grid = jax.vmap(jax.vmap(single_point_field_calc))(xy_for_vmap_y, xy_for_vmap_x) if we pass separate x,y grids
    # If xy is (width, height, 2) as from stack(meshgrid)
    
    # The original vmap2(f)(xy) where f = partial(fields_f, params, points)
    # and xy was mgrid.T (shape W,W,2)
    # fields_f takes (params, all_points, single_query_point_x)
    # So, functools.partial(fields_f, params, points) is a function of 'single_query_point_x'
    # vmap2 will apply this over a 2D grid of 'single_query_point_x'
    
    fields_calculator = functools.partial(fields_f, params, points)
    grid_fields = vmap2(fields_calculator)(xy) # xy should be (canvas_width, canvas_width, 2)

    # Masking for points (optional, from original code)
    # r2_to_points = jnp.square(xy[..., None, :] - points).sum(-1).min(-1) # Distance from grid cell to nearest particle
    # particle_render_mask = (r2_to_points / ( (extent*0.02)**2 ) ).clip(0, 1.0)[..., None] # Small radius around particles
    # For a field view, this particle mask might not be what we want, let's remove for now.

    # Energy field visualization (E)
    e_zero_offset = -peak_f(0.0, params.mu_g, params.sigma_g) # Potential energy offset
    vis_e_field = cmap_e(grid_fields.E - e_zero_offset)
    
    # Main visualization starts with E field
    final_visualization = vis_e_field

    guide_bar_width = 16 # pixels

    if show_color_guides:
        # Color guide for E field
        mean_particle_energy = jax.vmap(functools.partial(fields_f, params, points))(points).E.mean()
        guide_e_values = jnp.linspace(0.5, -0.5, canvas_width) # Values for the bar
        guide_e_bar_colors = cmap_e(guide_e_values)
        # Highlight mean particle energy on the bar
        guide_e_bar_colors = guide_e_bar_colors * (1.0 - peak_f(guide_e_values, mean_particle_energy - e_zero_offset, 0.005)[:, None])
        guide_e_bar_image = jnp.tile(guide_e_bar_colors[:, None, :], (1, guide_bar_width, 1))
        final_visualization = jnp.hstack([final_visualization, guide_e_bar_image])

    if show_field_types: # Corresponds to old show_UG
        vis_ug_fields = cmap_ug(grid_fields.U, grid_fields.G)
        
        if show_color_guides:
            # Color guide for U/G fields
            guide_u_values = jnp.linspace(1.0, 0.0, canvas_width)
            guide_g_for_u_bar = peak_f(guide_u_values, params.mu_g, params.sigma_g)
            guide_ug_bar_colors = cmap_ug(guide_u_values, guide_g_for_u_bar)
            guide_ug_bar_image = jnp.tile(guide_ug_bar_colors[:, None, :], (1, guide_bar_width, 1))
            # Prepend UG field and its guide
            final_visualization = jnp.hstack([guide_ug_bar_image, vis_ug_fields, final_visualization])
        else:
            final_visualization = jnp.hstack([vis_ug_fields, final_visualization])
            
    # Perform scaling and type conversion using jnp for JIT compatibility
    # Ensure the operations handle potential JAX arrays correctly.
    # final_visualization should already be a JAX array here.
    processed_visualization = jnp.clip(final_visualization, 0, 1) * 255.0
    uint8_visualization = processed_visualization.astype(jnp.uint8)
    
    return uint8_visualization # Return the JAX array

# Removed: fontpath, pil_font, text_overlay
# Removed: animate_lenia, _process_animation_frames
# Removed: odeint_euler (replaced by single-step update_lenia_particles)
# Removed: Example script execution (params = ..., key = ..., etc.)
