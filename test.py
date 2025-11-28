# test.py
from env.lane_env import LaneFollowingCircleEnv
from agent.dqn_agent import DQNAgent
import torch

env = LaneFollowingCircleEnv()
agent = DQNAgent(state_dim=4, action_dim=3)

agent.model.load_state_dict(torch.load("dqn_model.pth"))

state, _ = env.reset()

for _ in range(1000):
    action = agent.act(state, eval_mode=True)  # testte epsilonsuz
    state, reward, done, truncated, _ = env.step(action)
    env.render()
    if done:
        break
