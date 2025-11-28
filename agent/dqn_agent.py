# agent/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .replay_buffer import ReplayBuffer


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


class DQNAgent:
    def __init__(self, state_dim=4, action_dim=3,
                 gamma=0.98, lr=5e-4):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma

        self.replay_buffer = ReplayBuffer(max_size=50_000)
        self.batch_size = 64

        # Epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        # Training control
        self.train_start = 1_000          # bu kadar transition dolmadan train yok
        self.train_step_count = 0
        self.target_update_every = 500    # 500 gradient step'te bir target güncelle

        self.loss_fn = nn.MSELoss()

    def act(self, state, eval_mode=False):
        # Test sırasında tamamen greedy kullanmak istersen eval_mode=True verebilirsin
        if (not eval_mode) and (np.random.rand() < self.epsilon):
            return np.random.randint(3)

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def train_step(self):
        # Replay buffer yeterince dolmadan başlamayalım
        if len(self.replay_buffer.buffer) < self.train_start:
            return

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Q(s,a)
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]

        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping → patlamasın
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        # target network update
        self.train_step_count += 1
        if self.train_step_count % self.target_update_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
