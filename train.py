# train.py
from env.lane_env import LaneFollowingCircleEnv
from agent.dqn_agent import DQNAgent
import numpy as np
import torch

env = LaneFollowingCircleEnv()
agent = DQNAgent(state_dim=4, action_dim=3)

episodes = 800   # circle track için 500–1000 arası deneyebilirsin
max_steps = 400  # ep başına adım sınırı

best_mean_reward = -1e9

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0.0

    for t in range(max_steps):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        agent.replay_buffer.add((state, action, reward, next_state, float(done)))

        state = next_state
        total_reward += reward

        agent.train_step()

        if done:
            break

    # epsilon decay
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    print(f"Episode {ep+1}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Çok basit bir model kayıt mantığı
    if (ep + 1) % 20 == 0:
        torch.save(agent.model.state_dict(), "dqn_model.pth")
        print("Model kaydedildi → dqn_model.pth")
