import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn

from rlkit.dynamics.model import DynamicsModel
from rlkit.torch.networks import FlattenMlp

class MlpModel(nn.Module, DynamicsModel):
    def __init__(
            self,
            env,
            input_dim,
            action_dim,
            next_obs_dim,
            n_layers=3,
            hidden_layer_size=64,
            optimizer_class=optim.Adam
            learning_rate=1e-3,
            **kwargs
    ):
        super().__init__(
            env=env,
            input_dim=input_dim,
            action_dim=action_dim,
            next_obs_dim=next_obs_dim,
            **kwargs
        )
        self.env = env
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.next_obs_dim = next_obs_dim

        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate

        self.state = self.reset()

        self.reward_dim = 1
        #terminal_dim = 1

        self.net = FlattenMlp(
            hidden_sizes=[hidden_layer_size] * n_layers,
            input_size=input_dim + action_dim,
            output_size=next_obs_dim + reward_dim,
        )
        self.net_optimizer = optmizer_class(self.net.parameters(), lr=learning_rate)

    def _forward(self, state, action):
        output = self.net(self.state, action)
        next_state = output[:, :-self.reward_dim]
        reward = output[:, -self.reward_dim:]

        terminal = 0
        env_info = {}
        
        return next_state, reward, terminal, env_info

    def step(self, action):
        next_state, reward, terminal, env_info = self._forward(self.state, action)
        self.state = next_state

        return next_state, reward, terminal, env_info

    def train(self, paths):
        states = paths["observations"]
        actions = paths["actions"]
        rewards = paths["rewards"]
        next_states = paths["next_observations"]
        terminals = paths["terminals"]

        next_state_preds, reward_preds, terminal_preds, env_infos = self.forward(states, actions)
        self.net_optimizer.zero_grad()

        net_loss = torch.mean((next_state_preds - next_states) ** 2) + torch.mean((reward_preds - rewards) ** 2)
        net_loss.backward()
        self.net_optimizer.step()
