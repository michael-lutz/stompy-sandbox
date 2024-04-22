import torch.nn as nn
import torch
from typing import Tuple, List

class RLAgent(nn.Module):
    """
    A agent interface for reinforcement learning.
    """
    
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_action(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError
    
class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
        print(self.net)
        self.std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(x)
        std = torch.exp(self.std)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.net(x)
        return value