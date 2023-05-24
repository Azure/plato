import pickle
from pathlib import Path

import gymnasium.spaces as spaces
import numpy as np
from ray import serve

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


def prepare_path(checkpoint_folder):
    return str(
        Path(__file__).absolute().parent.parent / "checkpoints" / checkpoint_folder
    )


def prepare_agent(checkpoint_folder):
    """Prepare an RLlib agent for inference from a checkpoint folder.

    This function loads the RLlib AlgorithmConfig from the checkpoint folder,
    unfreezes it to make changes to the environment, rebuilds the algorithm, and
    restores the algorithm state from the checkpoint to instantiate the agent.

    Args:
        checkpoint_folder (str): The path to the checkpoint folder.

    Returns:
        Algorithm: The prepared algorithm for inference.
    """
    # Load the algorithm information from the checkpoint
    with open(checkpoint_folder + "/algorithm_state.pkl", "rb") as f:
        data = pickle.load(f)

    # Unfreeze the AlgorithmConfig to make changes to env
    algo_config = data["config"].copy(copy_frozen=False)

    # Change the sim env to None (not needed for inference), and rebuild the algorithm
    algo = algo_config.environment(
        env=None,
        observation_space=observation_space,
        action_space=action_space,
    ).build()

    # Restore the algorithm state from the checkpoint
    algo.restore(checkpoint_folder)
    return algo


@serve.deployment
class ServeAgent:
    def __init__(self):
        check_path = prepare_path(CHECKPOINT_FOLDER)
        self.agent = prepare_agent(check_path)

    async def __call__(self, request):
        json_input = await request.json()
        state = check_and_transform(observation_space, json_input["state"])
        # Set explore to false or the agent will not be deterministic
        action = self.agent.compute_single_action(state, explore=False)
        return action


agent = ServeAgent.bind()  # type: ignore
