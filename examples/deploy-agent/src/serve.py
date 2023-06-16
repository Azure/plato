from pathlib import Path

import gymnasium.spaces as spaces
import numpy as np
from ray import serve

from platotk.restore import restore_agent
from platotk.serialize import check_and_transform

# The name of the folder where the checkpoints are located in the checkpoints/ folder
# IMPORTANT: Remember to change it
CHECKPOINT_FOLDER = "checkpoint_000010"

# The observation and action spaces defined in your sim env
# IMPORTANT: Remember to change them
observation_space = spaces.Dict(
    {"value": spaces.Box(low=-float("inf"), high=float("inf"))}
)
action_space = spaces.Dict({"addend": spaces.Box(low=-10, high=10, dtype=np.int32)})

# Name used when registering sim environment at training
# IMPORTANT: Remember to change it
name_env = "sim_env"


def prepare_path(checkpoint_folder):
    return str(
        Path(__file__).absolute().parent.parent / "checkpoints" / checkpoint_folder
    )


@serve.deployment
class ServeAgent:
    def __init__(self):
        check_path = prepare_path(CHECKPOINT_FOLDER)
        self.agent = restore_agent(
            observation_space, action_space, check_path, name_env
        )

    async def __call__(self, request):
        json_input = await request.json()
        state = check_and_transform(observation_space, json_input["state"])
        # Set explore to false or the agent will not be deterministic
        action = self.agent.compute_single_action(state, explore=False)
        return action


agent = ServeAgent.bind()  # type: ignore
