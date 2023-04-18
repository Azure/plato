from pathlib import Path

import numpy as np
from ray import serve
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

# IMPORTANT: Check that the sim is available as sim.py in the src folder
from sim import SimpleAdder as SimEnv

# The name of the folder where the checkpoints are located in the checkpoints/ folder
# IMPORTANT: Remember to change it
CHECKPOINT_FOLDER = "checkpoint_000010"


def env_creator(env_config):
    return SimEnv(env_config)


# IMPORTANT: sim_env must be equal to the name used to register the environment when
# training the agent.
register_env("sim_env", env_creator)


# IMPORTANT: Modify the following function
def prepare_state(state):
    """Transform the state into the form expected by the agent."""
    return {"value": np.array([state["value"]])}


def prepare_path(checkpoint_folder):
    return str(
        Path(__file__).absolute().parent.parent / "checkpoints" / checkpoint_folder
    )


@serve.deployment
class ServeAgent:
    def __init__(self):
        check_path = prepare_path(CHECKPOINT_FOLDER)
        self.agent = Algorithm.from_checkpoint(check_path)

    async def __call__(self, request):
        json_input = await request.json()
        state = prepare_state(json_input["state"])
        # Set explore to false or the agent will not be deterministic
        action = self.agent.compute_single_action(state, explore=False)
        return action


agent = ServeAgent.bind()  # type: ignore
