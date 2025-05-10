import streamlit as st
import jax
import numpy as np
import time # Added for controlling animation speed

# Updated imports for the Lenia simulation from particles_full.py
from cosmic_soup.particles import (
    get_default_lenia_params,
    initialize_lenia_particles,
    update_lenia_particles,
    render_lenia_state,
    Params as LeniaParams # Alias to avoid potential conflicts and for clarity
)

# -----------------------------------
# Streamlit App Config
# -----------------------------------
st.set_page_config(
    page_title="Cosmic Soup - Lenia (Live)",
    page_icon="🌌",
    layout="wide",
)

st.title("🌌 Cosmic Soup — Live Lenia Particle Playground")

# -----------------------------------
# Helper Function for Simulation Initialization/Reset
# -----------------------------------
def init_simulation_state(session_key_prefix="sim"):
    """Initializes or resets the simulation state in st.session_state."""
    # Use a unique key for the simulation state to allow multiple instances if ever needed
    # For now, just one main simulation.
    
    # Get current sidebar values
    current_seed = st.session_state.get(f'{session_key_prefix}_seed_slider', 42)
    current_num_particles = st.session_state.get(f'{session_key_prefix}_num_particles_slider', 200)
    current_simulation_extent = st.session_state.get(f'{session_key_prefix}_simulation_extent_slider', 20.0)

    new_key = jax.random.PRNGKey(current_seed)
    initial_positions, initial_velocities = initialize_lenia_particles(
        new_key,
        current_num_particles,
        initial_spread=current_simulation_extent
    )
    st.session_state[f'{session_key_prefix}_key'] = new_key
    st.session_state[f'{session_key_prefix}_positions'] = initial_positions
    st.session_state[f'{session_key_prefix}_velocities'] = initial_velocities
    st.session_state[f'{session_key_prefix}_is_running'] = False
    st.session_state[f'{session_key_prefix}_current_step'] = 0
    # To prevent re-initialization on every rerun unless explicitly reset
    st.session_state[f'{session_key_prefix}_initialized'] = True

# -----------------------------------
# Sidebar Controls
# -----------------------------------
st.sidebar.header("Simulation Setup")

# Store sidebar values also in session_state with unique keys to be accessible by init_simulation_state
num_particles = st.sidebar.slider("Number of Particles", 10, 1000, 200, step=10, key="sim_num_particles_slider")
simulation_extent = st.sidebar.slider("Simulation Extent", 10.0, 50.0, 20.0, step=1.0, key="sim_simulation_extent_slider")
dt = st.sidebar.slider("Time Step Size (dt)", 0.001, 0.2, 0.05, format="%.3f", key="sim_dt_slider")
seed = st.sidebar.number_input("Random Seed", value=42, step=1, key="sim_seed_slider")

st.sidebar.subheader("Lenia Parameters")
default_params = get_default_lenia_params()
mu_k = st.sidebar.slider("mu_k", 0.0, 10.0, default_params.mu_k, step=0.1, key="sim_mu_k_slider")
sigma_k = st.sidebar.slider("sigma_k", 0.1, 5.0, default_params.sigma_k, step=0.05, key="sim_sigma_k_slider")
w_k = st.sidebar.slider("w_k", 0.001, 0.1, default_params.w_k, step=0.001, format="%.3f", key="sim_w_k_slider")
mu_g = st.sidebar.slider("mu_g", 0.0, 1.0, default_params.mu_g, step=0.01, key="sim_mu_g_slider")
sigma_g = st.sidebar.slider("sigma_g", 0.01, 0.5, default_params.sigma_g, step=0.005, key="sim_sigma_g_slider")
c_rep = st.sidebar.slider("c_rep", 0.0, 2.0, default_params.c_rep, step=0.05, key="sim_c_rep_slider")

# Construct LeniaParams from current sidebar values
current_lenia_params = LeniaParams(mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep)

st.sidebar.subheader("Rendering Options")
canvas_width_pixels = st.sidebar.slider("Canvas Width (px)", 200, 800, 500, step=50, key="sim_canvas_width_slider")
show_field_types = st.sidebar.checkbox("Show U/G Fields", value=False, key="sim_show_field_types_cb")
show_color_guides = st.sidebar.checkbox("Show Color Guides", value=True, key="sim_show_color_guides_cb")
animation_delay = st.sidebar.slider("Animation Delay (s)", 0.0, 0.5, 0.01, step=0.01, format="%.2f", key="sim_animation_delay_slider")

# -----------------------------------
# Initialize simulation state if it's not already done
# -----------------------------------
if not st.session_state.get("sim_initialized", False):
    init_simulation_state()

# -----------------------------------
# Control Buttons
# -----------------------------------
st.sidebar.header("Animation Controls")
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("Start", key="sim_start_button"):
    st.session_state.sim_is_running = True
if col2.button("Stop", key="sim_stop_button"):
    st.session_state.sim_is_running = False
if col3.button("Reset", key="sim_reset_button"):
    init_simulation_state() # Re-initialize with current sidebar values
    st.session_state.sim_is_running = False # Ensure it stops on reset

# -----------------------------------
# Main Simulation Area
# -----------------------------------
image_placeholder = st.empty()
step_display_placeholder = st.empty()

# Retrieve current simulation state
current_positions = st.session_state.sim_positions
current_velocities = st.session_state.sim_velocities
current_step_count = st.session_state.sim_current_step

if st.session_state.get("sim_is_running", False):
    # Perform one simulation step
    current_positions, current_velocities = update_lenia_particles(
        current_lenia_params, # Use LeniaParams from current sidebar values
        current_positions,
        current_velocities,
        dt, # Use dt from current sidebar value
        extent=simulation_extent # Use simulation_extent from current sidebar value
    )
    current_step_count += 1
    
    # Update session state
    st.session_state.sim_positions = current_positions
    st.session_state.sim_velocities = current_velocities
    st.session_state.sim_current_step = current_step_count

# Always render the current state (whether running or paused/reset)
jax_image = render_lenia_state(
    params=current_lenia_params, # Use LeniaParams from current sidebar values
    points=current_positions,
    extent=simulation_extent, # Use simulation_extent from current sidebar value
    canvas_width=canvas_width_pixels,
    show_field_types=show_field_types,
    show_color_guides=show_color_guides
)
rendered_image_np = np.array(jax_image)
image_placeholder.image(rendered_image_np, 
                        caption=f"Lenia State (Step: {current_step_count})", 
                        width=canvas_width_pixels)
step_display_placeholder.caption(f"Simulation Step: {current_step_count}")

# If running, schedule a rerun
if st.session_state.get("sim_is_running", False):
    time.sleep(animation_delay) # Use delay from sidebar
    st.rerun()

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.caption("Crafted with 💫 using JAX + Streamlit (Live Lenia Simulation)")