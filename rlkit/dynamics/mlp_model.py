import numpy as np

import torch
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.dynamics.model import DynamicsModel
from rlkit.torch.networks import FlattenMlp


class MlpModel(DynamicsModel):
    def __init__(
            self,
            env,
            n_layers=3,
            hidden_layer_size=64,
            optimizer_class=optim.Adam,
            learning_rate=1e-3,
            **kwargs
    ):
        super().__init__(env=env, **kwargs)
        self.env = env
        self.input_dim = len(env.observation_space.shape)
        self.action_dim = len(env.action_space.shape)
        self.next_obs_dim = len(env.observation_space.shape)

        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate

        self.state = self.reset()

        self.reward_dim = 1
        #terminal_dim = 1

        self.net = FlattenMlp(
            hidden_sizes=[hidden_layer_size] * n_layers,
            input_size=self.input_dim + self.action_dim,
            output_size=self.next_obs_dim + self.reward_dim,
        )
        self.net_optimizer = optimizer_class(self.net.parameters(), lr=learning_rate)

    def _forward(self, state, action):
        s = torch.from_numpy(state).float()
        a = torch.from_numpy(action).float()
        output = self.net(s, a)
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

        next_state_preds, reward_preds, terminal_preds, env_infos = self._forward(states, actions)
        self.net_optimizer.zero_grad()

        net_loss = torch.mean((next_state_preds - next_states) ** 2) + torch.mean((reward_preds - rewards) ** 2)
        net_loss.backward()
        self.net_optimizer.step()
