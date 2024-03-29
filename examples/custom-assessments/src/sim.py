"""Implementation of a simple simulation/environment in AML."""
from typing import Optional

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
    between -10 and 10. At each episode, the state number is initialized between
    0 and 100, and at each step the agent chooses a number between -10 and 10.
    The chosen number is added to the state. The purpose of the simulation is to
    get the state equal to 50, at which point the episode terminates. The episode
    duration is limited to 10 steps.
    """

    def __init__(self, env_config):
        self.observation_space = Dict(
            {"value": Box(low=-float("inf"), high=float("inf"))}
        )
        self.action_space = Dict({"addend": Box(low=-10, high=10, dtype=np.int32)})
        self.state = {"value": np.array([0])}

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

    def reset(
        self,
        *,
        seed: Optional[dict] = None,
        options: Optional[dict] = None,
    ):
        """Start a new episode."""
        self.iter = 0

        if options is None or options.get("value") is None:
            self.state = {"value": np.random.randint(0, 100)}
        else:
            self.state = {"value": options["value"]}

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Advance one step by applying the given ``action``."""
        self.state["value"] += action["addend"].item()
        self.iter += 1
        reward = self.reward(self.state)
        terminated = self.state["value"] == 50
        truncated = self.iter >= 10
        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )
