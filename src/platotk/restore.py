import pickle
from pathlib import Path

from gymnasium import Env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env


class DummyEnv(Env):
    def __init__(self, env_config, observation_space=None, action_space=None):
        if observation_space is None:
            raise TypeError("observation_space cannot be of type None.")
        self.observation_space = observation_space

        if action_space is None:
            raise TypeError("action_space cannot be of type None.")
        self.action_space = action_space

    def step(self, action):
        return self.observation_space.sample(), 0, False, False, {}

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}


def restore_agent_from_checkpoint(
    observation_space,
    action_space,
    checkpoint_path,
    name_env="sim_env",
):
    """Restore an RLlib agent using ``from_checkpoint``."""
    register_env(name_env, lambda conf: DummyEnv(conf, observation_space, action_space))
    return Algorithm.from_checkpoint(checkpoint_path)


def restore_agent_from_pickle(
    observation_space,
    action_space,
    checkpoint_path,
    name_env="sim_env",
):
    """Restore an RLlib agent by unpickling and modifying config."""
    with open(Path(checkpoint_path) / "algorithm_state.pkl", "rb") as fp:
        data = pickle.load(fp)
    agent_config = data["config"].copy(copy_frozen=False)
    agent = (
        agent_config.rollouts(
            num_rollout_workers=0,
        )
        .environment(
            env=None,
            observation_space=observation_space,
            action_space=action_space,
        )
        .build()
    )
    agent.restore(checkpoint_path)
    return agent
