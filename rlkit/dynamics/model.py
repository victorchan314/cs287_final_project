class DynamicsModel(object):
    def __init__(self, env, input_dim, action_dim, next_obs_dim, **kwargs):
        self.env = env
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.next_obs_dim = next_obs_dim

    def close(self):
        raise NotImplementedError

    def render(self):
        pass

    def reset(self):
        return env.reset()

    def seed(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
