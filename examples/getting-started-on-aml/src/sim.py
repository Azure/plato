"""Implementation of a simple simulation/environment in AML."""
from random import choice

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict


class SimpleAdder(Env):
    """
    Implement a SimpleAdder as a custom Gymnasium environment.

    Details on which attributes and methods are required for the integration
    can be found in the docs.

    The environment has a pretty simple state and action space. The state is
    composed of an integer numbers. The action is composed of an integer number
    between -10 and 10.

    At each episode, the state number is initialized between 0 and 100. The
    range of possible initial values can be overwritten by passing an iterable
    to ``env_config``. This capability can be used to apply domain
    randomization to the training process.

    At each iteration, the agent chooses a number between -10 and 10. The
    chosen number is added to the state. The purpose of the simulation is to
    get the state equal to 50, at which point the episode terminates. The
    maximum episode duration can be configured via ``env_config``. The default
    duration is 10 steps.
    """

    def __init__(self, env_config):
        self.observation_space = Dict(
            {"value": Box(low=-float("-inf"), high=float("inf"), dtype=int)}
        )
        self.action_space = Dict({"addend": Box(low=-10, high=10, dtype=int)})
        self.initial_values = self.set_initial_values(env_config.get("values"))
        self.max_steps = env_config.get("max_steps", 10)
        self.state = {"value": self.get_initial_value()}

    def set_initial_values(self, values):
        """
        Set the allowed initial values for the sim environment.

        This function is used for domain randomization.
        If the user does not pass the set of allowed initial values, the sim
        environment takes the default range.

        It's important the function checks that the provided initial
        values are contained in observation_space.
        If they're not, it raises a ValueError.
        """
        if values is None:
            values = range(0, 101)
        for value in values:
            if not self.observation_space.get("value").contains([value]):
                raise ValueError(
                    "The initial value provided is not contained"
                    " in the allowed values."
                    f" Found {value}."
                )
        return values

    def get_initial_value(self):
        """Pick the initial value from the set of possible values."""
        return choice(self.initial_values)

    def _get_obs(self):
        """Get the observable state."""
        return {"value": np.array([self.state["value"]])}

    def _get_info(self):
        """Get additional info not needed by the agent's decision."""
        return {}

    def reward(self, state):
        """
        Return the reward value.

        For this simple example this is just the distance to the number 50.
        """
        return -abs(state["value"] - 50)

    def reset(self, *, seed=None, options=None):
        """Start a new episode."""
        self.state = {"value": self.get_initial_value()}
        self.iter = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Advance one iteration by applying the given ``action``."""
        self.state["value"] += action["addend"].item()
        self.iter += 1
        reward = self.reward(self.state)
        terminated = self.state["value"] == 50
        truncated = self.iter >= self.max_steps
        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )
