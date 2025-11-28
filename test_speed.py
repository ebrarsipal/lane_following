# test_speed.py
import torch
from env.lane_env_speed import LaneFollowingCircleSpeedEnv
from agent.dqn_agent import DQNAgent

env = LaneFollowingCircleSpeedEnv()
agent = DQNAgent(state_dim=5, action_dim=9)

agent.model.load_state_dict(torch.load("dqn_model_speed.pth", map_location="cpu"))
agent.model.eval()

num_episodes = 3

for ep in range(1, num_episodes + 1):
    print(f"\n=== Episode {ep} ===")
    state, _ = env.reset()

    for step in range(600):
        action = agent.act(state, eval_mode=True)  # epsilon yok
        state, reward, terminated, truncated, _ = env.step(action)
        env.render()

        if terminated:
            print(f"🚫 Out of lane at step {step}")
            break
        if truncated:
            print(f"⏹ Max steps reached at step {step}")
            break

env.close()
print("Test bitti.")
