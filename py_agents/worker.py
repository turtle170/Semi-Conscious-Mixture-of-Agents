import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class WorkerAgent(nn.Module):
    def __init__(self, latent_dim=128, action_dim=64):
        super(WorkerAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, latent_state):
        logits = self.actor(latent_state)
        value = self.critic(latent_state)
        return logits, value

    def select_action(self, latent_state):
        logits, value = self.forward(latent_state)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.item(), probs.log_prob(action), value.item()

class RolloutBuffer:
    def __init__(self):
        self.latents = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []

    def clear(self):
        del self.latents[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.is_terminals[:]

# Dummy PPO logic for the Worker's agency
if __name__ == "__main__":
    print("Worker: RL Agent initialized.")
    # In the full SCMoA, the Worker would connect via its own pipe or be part of a multi-agent process.
    # For this demo, we've integrated Scientist and Worker roles or they are separate processes.
    # The directive asks for worker.py as a standalone component.
