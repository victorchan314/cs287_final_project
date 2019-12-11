class DynamicsModel(object):
    def __init__(self, env, **kwargs):
        self.env = env

    def close(self):
        raise NotImplementedError

    def render(self):
        pass

    def reset(self):
        return self.env.reset()

    def seed(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
