# dqn_agent_speed.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .replay_buffer import ReplayBuffer


# --------------------
# Q-Network
# --------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# --------------------
# Agent
# --------------------
class DQNAgent:
    def __init__(
        self,
        state_dim=5,              # radial_err, heading_err, sinθ, cosθ, speed
        action_dim=4,             # left, right, speed_down, speed_up
        gamma=0.99,
        lr=1e-3,
        buffer_size=50_000,
        batch_size=64,
        tau=1.0
    ):
        # Q-Networks
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)

        # Epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    # ------------------------
    # Action selection
    # ------------------------
    def act(self, state, eval_mode=False):
        if (not eval_mode) and (np.random.rand() < self.epsilon):
            return np.random.randint(0, 4)

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.model(state_t)
        return torch.argmax(q_vals).item()

    # ------------------------
    # Training step
    # ------------------------
    def train_step(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Q(s, a)
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max Q(s', a') from target network
        next_q = self.target_model(next_states).max(1)[0]
        target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ------------------------
    # Target update
    # ------------------------
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
