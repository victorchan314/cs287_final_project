import numpy as np

import rlkit.torch.pytorch_util as ptu


class DynamicsModel(object):
    def __init__(self, env, **kwargs):
        self.env = env

    def close(self):
        raise NotImplementedError

    def render(self):
        pass

    def reset(self):
        state = self.env.reset()
        self.state = ptu.from_numpy(state[np.newaxis, :])

        return state

    def seed(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
