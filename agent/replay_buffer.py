import random
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=50_000):
        self.max_size = max_size
        self.buffer = []

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            # ilk elemanı sil (FIFO)
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
