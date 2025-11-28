# train_speed.py
import torch
from env.lane_env_speed import LaneFollowingCircleSpeedEnv
from agent.dqn_agent_speed import DQNAgent

env = LaneFollowingCircleSpeedEnv()
agent = DQNAgent(state_dim=5, action_dim=9)

num_episodes = 1000

for ep in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0.0

    for t in range(600):
        action = agent.act(state)   # epsilon-greedy
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # Replay buffer'a ekle
        agent.replay_buffer.add((state, action, reward, next_state, float(done)))

        state = next_state
        total_reward += reward

        # Bir adım öğren
        agent.train_step()

        if done:
            break

    # Target ağı güncelle
    agent.update_target()
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    print(f"Episode {ep+1}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # ara ara modeli kaydet
    if (ep + 1) % 50 == 0:
        torch.save(agent.model.state_dict(), "dqn_model_speed.pth")
        print("Model kaydedildi → dqn_model_speed.pth")

env.close()
