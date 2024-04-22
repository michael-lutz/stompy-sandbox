import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from torch.distributions import Normal
from infrastructure.replay_buffer import Transition
from agents.basic_agent import RLAgent, ContinuousPolicyNetwork, ValueNetwork
torch.autograd.set_detect_anomaly(True)

class PPOAgent(RLAgent):
    """
    Implementation of Proximal Policy Optimization (PPO) agent for continuous action spaces using a general continuous policy network.
    """
    def __init__(self, policy_network: ContinuousPolicyNetwork, value_network: ValueNetwork,
                 learning_rate: float = 3e-4, gamma: float = 0.99, tau: float = 0.95,
                 clip_param: float = 0.2, ppo_epochs: int = 10, mini_batch_size: int = 64):
        super().__init__()
        self.policy_network = policy_network
        self.value_network = value_network
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass the input state through the policy network to get action distribution parameters.

        :param x: Input state tensor.
        :return: Tuple of tensors (mean, std) representing the parameters of the action distribution.
        """
        return self.policy_network(x)

    def get_action(self, observation: np.ndarray, detach=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine the action and its log probability given an observation.

        :param observation: Current state observation.
        :return: Tuple containing the action and log probability of that action.
        """
        x = torch.from_numpy(observation).float()
        mean, std = self.forward(x)
        

        if detach:
            mean = mean.detach()
            std = std.detach()

        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob

    def update(self, transitions: Transition) -> None:
        """
        Update the policy and value networks using the PPO algorithm.
        """
        observations = torch.from_numpy(transitions['observations']).float()
        actions = torch.from_numpy(transitions['actions']).float()
        rewards = torch.from_numpy(transitions['rewards']).unsqueeze(-1).float()
        next_observations = torch.from_numpy(transitions['next_observations']).float()
        dones = torch.from_numpy(transitions['dones']).unsqueeze(-1).float()

        # Compute advantages and discounted returns
        values = self.value_network(observations)
        next_values = self.value_network(next_observations).detach()
        deltas = rewards + self.gamma * (1 - dones) * next_values - values
        advantages = torch.zeros_like(deltas)
        advantage = torch.zeros_like(deltas[0])

        for i in reversed(range(len(deltas))):
            advantage = deltas[i] + self.gamma * self.tau * (1 - dones[i]) * advantage
            advantages[i] = advantage
        print(advantages.shape)
        returns = advantages + values
        print(returns.shape)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(observations))
            for start in range(0, len(observations), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_observations = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                mean, std = self.policy_network.forward(batch_observations)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(axis=-1)
                old_log_probs = torch.from_numpy(transitions['log_probs'][batch_indices]).float()

                # Ratio for clipping
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                self.optimizer_policy.zero_grad()
                if _ == self.ppo_epochs - 1:
                    policy_loss.backward()  # On last epoch, do not retain graph
                else:
                    policy_loss.backward(retain_graph=True)
                self.optimizer_policy.step()

                self.optimizer_value.zero_grad()
                current_values = self.value_network(batch_observations)
                value_loss = (batch_returns - current_values).pow(2).mean()
                if _ == self.ppo_epochs - 1:
                    value_loss.backward()  # On last epoch, do not retain graph
                else:
                    value_loss.backward(retain_graph=True)
                self.optimizer_value.step()

    def save(self, filepath: str) -> None:
        """
        Save the model parameters.

        :param filepath: Path to save the model parameters.
        """
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'optimizer_policy_state_dict': self.optimizer_policy.state_dict(),
            'optimizer_value_state_dict': self.optimizer_value.state_dict(),
        }, filepath)

    def load(self, filepath: str) -> None:
        """
        Load the model parameters from a saved file.

        :param filepath: Path to the file containing model parameters.
        """
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
        self.optimizer_value.load_state_dict(checkpoint['optimizer_value_state_dict'])
