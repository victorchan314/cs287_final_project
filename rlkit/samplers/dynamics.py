import numpy as np

from rlkit.dynamics.mlp_model import MlpModel
from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic


class DynamicsSampler(object):
    """
    A sampler that uses samples to learn a dynamics model
    before using the dynamics model to generate samples.
    """
    def __init__(self, env, policy, max_path_length, num_train_itr=500, num_train_steps_per_itr=50, tandem_train=True, **kwargs):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length
        self.num_train_itr = num_train_itr
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.tandem_train = tandem_train

        self.model = MlpModel(env, **kwargs)
        self.itr = 0

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def to(self, device=None):
        self.model.to(device)

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1, testing=False):
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0

        if self.itr <= self.num_train_itr:
            if self.tandem_train:
                self._train(policy, accum_context)
                self.itr += 1
            else:
                for _ in range(self.num_train_itr):
                    self._train(policy, accum_context)
                    self.itr += 1

        while n_steps_total < max_samples and n_trajs < max_trajs:
            if testing:
                path = rollout(self.env, policy, max_path_length=self.max_path_length, accum_context=accum_context)
            else:
                path = rollout(self.model, policy, max_path_length=self.max_path_length, accum_context=accum_context)

            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            if n_trajs % resample == 0:
                policy.sample_z()

        return paths, n_steps_total

    def get_losses(self):
        return self.model.transition_model_loss, self.model.reward_model_loss, self.model.net_loss

    def _train(self, policy, accum_context):
        for i in range(self.num_train_steps_per_itr):
            path = rollout(self.env, policy, max_path_length=self.max_path_length, accum_context=accum_context)
            self.model.train(path)

