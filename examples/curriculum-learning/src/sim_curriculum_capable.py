"""Implementation of a simple simulation/environment in AML."""
import numpy as np

# from gymnasium import Env
from gymnasium.spaces import Box, Dict

# Import TaskSettableEnv from RLlib
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.utils.annotations import override


class SimpleAdder(TaskSettableEnv):
    """
    Implement a SimpleAdder as a custom Gymnasium environment.

    Details on which attributes and methods are required for the integration
    can be found in the docs.

    The environment has a pretty simple state and action space. The state is
    composed of an integer numbers. The action is composed of an integer number
    between -10 and 10. At each episode, the state number is initialized between
    0 and 100, and at each iteration the agent chooses a number between -10 and 10.
    The chosen number is added to the state. The purpose of the simulation is to
    get the state equal to 50, at which point the episode terminates. The episode
    duration is limited to 10 iterations.
    """

    def __init__(self, env_config):
        self.observation_space = Dict(
            {"value": Box(low=-float("inf"), high=float("inf"))}
        )
        self.action_space = Dict({"addend": Box(low=-10, high=10, dtype=np.int32)})

        # Initialize the task exponent attribute to 1
        self.exponent = 1

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
        We add 10 (maximum steps per episode) to the reward and subtract the
        current step to encourage to finish the episode as fast as possible.
        """
        return -abs(state["value"] - 50) + 10 - self.iter

    def reset(self, *, seed=None, options=None):
        """Start a new episode."""
        self.iter = 0
        # Get the current task (curriculum level)
        task = self.get_task()
        # Get the exponent of 2 for the task
        exponent = task["exponent"]
        # Initialize the state value randomly between +/- 2**exponent from target of 50
        self.state = {"value": 50 + np.random.randint(-(2**exponent), 2**exponent)}
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Advance one iteration by applying the given ``action``."""
        self.state["value"] += action["addend"]
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

    @override(TaskSettableEnv)
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        # Return the current exponent value as the task
        return {"exponent": self.exponent}

    @override(TaskSettableEnv)
    def set_task(self, task):
        """Set a new task for this sim env."""
        # Set the exponent value based on the task
        self.exponent = task["exponent"]
