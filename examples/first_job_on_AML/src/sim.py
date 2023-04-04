"""Implementation of a simple simulation/environment in AML."""
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete


class SimpleAdder(Env):
    """
    Implement a SimpleAdder as a gymnasium-like environment.

    Details on which attributes and methods are required for the integration
    can be found in the docs.

    The environment has a pretty simple state and action space. The state is
    composed of an integer numbers. The action is composed of an integer number
    between 0 and 20. At each episode, the state number is set to zero and at
    each iteration the agent chooses a number between 0 and 20. The chosen
    number is added to the state. The purpose of the simulation is to get the
    state equal to 50. The episode duration is limited to 10 iterations.
    """

    def __init__(self, env_config):
        self.action_space = Discrete(20)
        self.observation_space = Box(low=-float("inf"), high=float("inf"))
        self.number = 0

    def _get_obs(self):
        """Get the observable state."""
        return np.array([self.number], dtype=np.float32)

    def _get_info(self):
        """Get additional info not needed by the agent's decision."""
        return {}

    def reward(self, var):
        """
        Return the reward value.

        For this simple example this is just the distance to the number 50.
        """
        return -abs(var - 50)

    def reset(self, *, seed=None, options=None):
        """Start a new episode."""
        self.number = 0
        self.iter = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Advance one iteration by applying the given ``action``."""
        self.number += action - 10
        self.iter += 1
        reward = self.reward(self.number)
        terminated = self.iter >= 10
        return (
            self._get_obs(),
            reward,
            terminated,
            terminated,
            self._get_info(),
        )
