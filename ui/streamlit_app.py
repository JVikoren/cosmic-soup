import streamlit as st
import jax
import numpy as np
import matplotlib.pyplot as plt

from particles import update_particles, initialize_particles

# -----------------------------------
# Streamlit App Config
# -----------------------------------
st.set_page_config(
    page_title="Cosmic Soup",
    page_icon="🌌",
    layout="wide",
)

st.title("🌌 Cosmic Soup — Particle Playground")

# -----------------------------------
# Sidebar Controls
# -----------------------------------
st.sidebar.header("Simulation Controls")

num_particles = st.sidebar.slider("Number of Particles", 10, 1000, 200, step=10)
steps = st.sidebar.slider("Simulation Steps", 1, 500, 100, step=10)
dt = st.sidebar.slider("Time Step Size (dt)", 0.001, 0.1, 0.01)

interaction_strength = st.sidebar.slider("Interaction Strength", 0.0, 5.0, 1.0)
noise_scale = st.sidebar.slider("Noise Scale", 0.0, 1.0, 0.01)

seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# -----------------------------------
# Initialize Particles
# -----------------------------------
key = jax.random.PRNGKey(seed)
positions, velocities = initialize_particles(key, num_particles)

# -----------------------------------
# Run Simulation
# -----------------------------------
for _ in range(steps):
    positions, velocities = update_particles(
        positions,
        velocities,
        interaction_strength=interaction_strength,
        noise_scale=noise_scale,
        dt=dt,
    )

# -----------------------------------
# Visualization
# -----------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
positions_np = np.array(positions)

ax.scatter(positions_np[:, 0], positions_np[:, 1], s=5, alpha=0.7)
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
ax.set_title(f"Particles after {steps} steps", fontsize=16)

st.pyplot(fig)

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.caption("Crafted with 💫 using JAX + Streamlit")
