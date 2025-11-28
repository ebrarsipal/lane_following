import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
from env.lane_env import LaneFollowingCircleEnv
from agent.dqn_agent import DQNAgent
import torch

# === PAGE CONFIG ===
st.set_page_config(page_title="RL Circle Track Visualizer", layout="wide")

# === LOAD ENV + AGENT ===
env = LaneFollowingCircleEnv()
agent = DQNAgent(state_dim=4, action_dim=3)
agent.model.load_state_dict(torch.load("dqn_model.pth", map_location="cpu"))
agent.model.eval()

# === UI TITLE ===
st.title("🚗 Reinforcement Learning – Circle Track Visualization")

# === SIDEBAR ===
st.sidebar.header("Simulation Controls")
episode_limit = st.sidebar.number_input("Episodes", min_value=1, max_value=50, value=5)
speed_multiplier = st.sidebar.slider("Visualization Speed", 0.1, 3.0, 1.0)
render_trail = st.sidebar.checkbox("Show Trail", value=True)

start_btn = st.sidebar.button("▶ Start Simulation")


# === STREAMLIT PLACEHOLDERS ===
col_left, col_right = st.columns([3, 1])

plot_area = col_left.empty()
info_box = col_right.empty()


def draw_environment(env, trail_points):
    fig, ax = plt.subplots(figsize=(6, 6))

    # draw circular track
    angles = np.linspace(0, 2 * np.pi, 400)
    outer = env.track_radius + env.lane_half_width
    inner = env.track_radius - env.lane_half_width

    ax.plot(outer * np.cos(angles), outer * np.sin(angles),
            '--', color='gray')
    ax.plot(inner * np.cos(angles), inner * np.sin(angles),
            '--', color='gray')

    # draw trail
    if render_trail and len(trail_points) > 1:
        xs = [p[0] for p in trail_points]
        ys = [p[1] for p in trail_points]
        ax.plot(xs, ys, color="orange", linewidth=2)

    # draw car
    ax.plot(env.x, env.y, 'ro', markersize=10)

    # heading arrow
    hx = env.x + 0.8 * np.cos(env.heading)
    hy = env.y + 0.8 * np.sin(env.heading)
    ax.arrow(env.x, env.y, hx - env.x, hy - env.y,
             head_width=0.2, color='red', length_includes_head=True)

    ax.set_aspect('equal')
    ax.set_xlim(-env.track_radius - 3, env.track_radius + 3)
    ax.set_ylim(-env.track_radius - 3, env.track_radius + 3)

    plt.tight_layout()
    return fig


# === SIMULATION LOOP ===
if start_btn:
    for ep in range(1, episode_limit + 1):
        state, _ = env.reset()
        trail = []
        total_reward = 0

        st.write(f"### === Episode {ep} ===")

        for step in range(env.max_steps):

            # agent action
            action = agent.act(state, eval_mode=True)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            trail.append((env.x, env.y))

            # draw frame
            fig = draw_environment(env, trail)
            plot_area.pyplot(fig)

            # update info panel
            info_box.markdown(
                f"""
                ### Episode Stats
                - **Step:** {step} / {env.max_steps}
                - **Reward:** {total_reward:.3f}
                - **Position:** ({env.x:.2f}, {env.y:.2f})
                - **Heading:** {env.heading:.2f} rad  
                """
            )

            # episode end?
            if done:
                st.warning(f"💥 Episode {ep} terminated (out of lane) at step {step}")
                break
            if truncated:
                st.info(f"⏹ Episode {ep} truncated at step {step}")
                break

            time.sleep(0.02 / speed_multiplier)

    st.success("🎉 Simulation finished.")
