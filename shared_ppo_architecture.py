from gymnasium import spaces
import torch
from torch import nn
from typing import Callable, Tuple

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback


## SHARED PPO
class CustomNetwork(nn.Module):
    """
    A custom network that shares parameters for both
    the policy and value functions.
    """
    def __init__(
        self,
        feature_dim: int,
        shared_dim: int = 64,
        last_layer_dim: int = 64,
    ):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, shared_dim),
            nn.ReLU(),
        )

        # A single shared head for both policy and value
        self.shared_head = nn.Sequential(
            nn.Linear(shared_dim, last_layer_dim),
            nn.ReLU()
        )
        # SB3 needs to know the dimension of the policy and value features
        self.latent_dim_pi = last_layer_dim
        self.latent_dim_vf = last_layer_dim

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared(features)
        latent = self.shared_head(shared_features)
        # Return the same representation for policy and value
        return latent, latent

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        # Same final layers for policy
        shared_features = self.shared(features)
        return self.shared_head(shared_features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        # Same final layers for value
        shared_features = self.shared(features)
        return self.shared_head(shared_features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCriticPolicy that uses the above CustomNetwork
    for the core (feature extractor).
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

class CustomCallback(BaseCallback):
    def __init__(self, eval_env, model_name):
        super().__init__()
        self.eval_env = eval_env
        self.episode_rewards = []
        self.current_episode_reward = 0 
        self.episode_count = 0
        self.model_name = model_name
    
    def _on_step(self) -> bool:
        if "rewards" in self.locals:
            # Increment the total reward for the current episode
            self.current_episode_reward += self.locals["rewards"].item()  # Convert numpy array to scalar
        
        done = self.locals["dones"]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  # Reset for the next episode
            self.episode_count += 1
        return True
    