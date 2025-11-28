import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
from env.lane_env import LaneFollowingCircleEnv
from agent.dqn_agent import DQNAgent
import torch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from PIL import Image 
from matplotlib.patches import Circle # Daire yamaları için eklendi

# === PAGE CONFIG ===
st.set_page_config(page_title="RL Circle Track Visualizer", layout="wide")

# === LOAD ENV + AGENT ===
env = None
agent = None
try:
    env = LaneFollowingCircleEnv()
    agent = DQNAgent(state_dim=4, action_dim=3)
    agent.model.load_state_dict(torch.load("dqn_model.pth", map_location="cpu"))
    agent.model.eval()
except Exception as e:
    st.error(f"Ortam/Model yüklenirken bir hata oluştu: {e}")

# === UI TITLE ===
st.title("🚗 Reinforcement Learning – Circle Track Visualization")

# === SIDEBAR ===
st.sidebar.header("Simulation Controls")
episode_limit = st.sidebar.number_input("Episodes", min_value=1, max_value=50, value=5)
render_trail = st.sidebar.checkbox("Show Trail", value=True)

start_btn = st.sidebar.button("▶ Start Simulation", disabled=(env is None or agent is None))

# === STREAMLIT PLACEHOLDERS ===
col_left, col_right = st.columns([3, 1])

plot_area = col_left.empty()
info_box = col_right.empty()

# === ARABA RESMİNİ YÜKLE, KÜÇÜLT VE SİMETRİSİNİ AL (Bir Kez) ===
CAR_ZOOM = 0.03 # İkonu daha da küçültmek için değeri düşürdük
car_img = None
try:
    car_img_path = "car.png" 
    pil_img = Image.open(car_img_path)
    flipped_pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT) 
    car_img = np.array(flipped_pil_img)
except FileNotFoundError:
    st.error(f"'{car_img_path}' dosyası bulunamadı.")
except Exception as e:
    st.error(f"Resim işlenirken bir hata oluştu: {e}")


def draw_environment(env, trail_points, raw_car_img=None, zoom=CAR_ZOOM):
    fig, ax = plt.subplots(figsize=(6, 6))

    # === YOL ALANINI AÇIK GRİ YAPMA ===
    
    # Dış ve iç yarıçapları hesapla
    outer = env.track_radius + env.lane_half_width
    inner = env.track_radius - env.lane_half_width
    
    # 1. Yol Alanını Açık Griyle Doldurmak için büyük daire
    # Yarıçapı dış sınıra eşit bir daire çiz
    outer_circle = Circle((0, 0), outer, 
                          color='#D3D3D3',  # Açık gri (Light Gray)
                          fill=True, 
                          linewidth=0)
    ax.add_patch(outer_circle)
    
    # 2. İç alanı boşaltmak için küçük daire
    # Yarıçapı iç sınıra eşit bir daire çiz ve arka plan rengiyle doldur.
    # Varsayılan arka plan rengi genellikle beyazdır, bu yüzden 'white' kullanmak güvenlidir.
    inner_circle = Circle((0, 0), inner, 
                          color='white', 
                          fill=True, 
                          linewidth=0)
    ax.add_patch(inner_circle)

    # === YOL ÇİZGİLERİNİ ÇİZME ===
    angles = np.linspace(0, 2 * np.pi, 400)
    
    # Dış şerit çizgisi (Koyu gri kesikli çizgi)
    ax.plot(outer * np.cos(angles), outer * np.sin(angles),
            '--', color='#555555', linewidth=2)
    # İç şerit çizgisi (Koyu gri kesikli çizgi)
    ax.plot(inner * np.cos(angles), inner * np.sin(angles),
            '--', color='#555555', linewidth=2)
    
    # Orta şerit çizgisi (İsteğe bağlı, beyaz kesikli çizgi)
    ax.plot(env.track_radius * np.cos(angles), env.track_radius * np.sin(angles),
            '--', color='#AAAAAA', linewidth=1.5, alpha=0.9)


    # draw trail
    if render_trail and len(trail_points) > 1:
        xs = [p[0] for p in trail_points]
        ys = [p[1] for p in trail_points]
        ax.plot(xs, ys, color="yellow", linewidth=2)

    # draw car as an image
    if raw_car_img is not None:
        car_offset_image = OffsetImage(raw_car_img, zoom=zoom)
        ab = AnnotationBbox(car_offset_image, (env.x, env.y), frameon=False, pad=0.0)
        ax.add_artist(ab)
    else: 
        ax.plot(env.x, env.y, 'r^', markersize=12)

    # heading arrow
    hx = env.x + 0.8 * np.cos(env.heading)
    hy = env.y + 0.8 * np.sin(env.heading)
    ax.arrow(env.x, env.y, hx - env.x, hy - env.y,
             head_width=0.2, color='red', length_includes_head=True)

    ax.set_aspect('equal')
    ax.set_xlim(-env.track_radius - 3, env.track_radius + 3)
    ax.set_ylim(-env.track_radius - 3, env.track_radius + 3)
    
    # Grafik kenarlarını ve eksenleri kaldırma
    ax.axis('off') 

    plt.tight_layout()
    return fig


# === SIMULATION LOOP ===
if start_btn:
    if env is None or agent is None:
        st.error("Simülasyon ortamı veya ajanı yüklenemedi. Lütfen hataları düzeltin.")
    else:
        for ep in range(1, episode_limit + 1):
            state, _ = env.reset()
            trail = []
            total_reward = 0

            for step in range(env.max_steps):

                # agent action
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    q_values = agent.model(state_tensor)
                    action = torch.argmax(q_values).item()

                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward

                trail.append((env.x, env.y))

                # draw frame
                fig = draw_environment(env, trail, raw_car_img=car_img)
                plot_area.pyplot(fig)

                # update info panel
                info_box.markdown(
                    f"""
                    ### ℹ️ Episode Stats
                    - **Step:** {step} / {env.max_steps}
                    - **Reward:** {total_reward:.3f}
                    - **Position:** ({env.x:.2f}, {env.y:.2f})
                    - **Heading:** {env.heading:.2f} rad  
                    """
                )

                # episode end?
                if done:
                    st.warning(f"💥 Episode {ep} terminated (**out of lane**) at step {step}")
                    break
                if truncated:
                    st.info(f"⏹ Episode {ep} truncated at step {step}")
                    break

                time.sleep(0.02) 
                
            # Bölüm bittiğinde nihai durumu çiz.
            fig = draw_environment(env, trail, raw_car_img=car_img)
            plot_area.pyplot(fig)

        st.success("🎉 Simulation finished.")