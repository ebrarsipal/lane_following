# test_circle_visual.py
import torch
from env.lane_env import LaneFollowingCircleEnv
from agent.dqn_agent import DQNAgent

import time

# === Environment ===
env = LaneFollowingCircleEnv()

# === Agent ===
agent = DQNAgent(state_dim=4, action_dim=3)

# Load trained weights
agent.model.load_state_dict(torch.load("dqn_model.pth", map_location="cpu"))
agent.model.eval()

print("🟢 Test başlıyor...")

episodes = 5  # 5 kez ölünce yeniden başla

for ep in range(episodes):

    state, _ = env.reset()
    print(f"\n=== Episode {ep+1} ===")

    for step in range(1500):  # uzun sürüş
        action = agent.act(state, eval_mode=True)

        state, reward, done, truncated, _ = env.step(action)

        env.render()

        if done:
            print(f"💀 Episode {ep+1} terminated at step {step}")
            break

        if truncated:
            print(f"⏹ Episode {ep+1} truncated at step {step}")
            break

        time.sleep(0.01)  # görsel hız ayarı (daha akıcı yapmak istersen küçült)

print("\n🎉 Test bitti!")
env.close()
