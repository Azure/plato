import pickle
from pathlib import Path

import gymnasium.spaces as spaces
import numpy as np
from ray import serve

# The name of the folder where the checkpoints are located in the checkpoints/ folder
# IMPORTANT: Remember to change it
CHECKPOINT_FOLDER = "checkpoint_000010"

# The observation and action spaces defined in your sim env
# IMPORTANT: Remember to change them
observation_space = spaces.Dict(
    {"value": spaces.Box(low=-float("inf"), high=float("inf"))}
)
action_space = spaces.Dict({"addend": spaces.Box(low=-10, high=10, dtype=np.int32)})


# Define a function that returns a valid state observation for the sim env space
def check_and_transform(observation_space, state):
    # Check if the instance is valid for the space
    if observation_space.contains(state):
        # Return the instance as it is
        return state
    else:
        # Try to transform the instance based on the space type
        if isinstance(observation_space, spaces.Box):
            # Convert the instance to a numpy array with the same
            # shape and dtype as the space
            return np.array(state, dtype=observation_space.dtype).reshape(
                observation_space.shape
            )
        elif isinstance(observation_space, spaces.Discrete):
            # Convert the instance to an integer within the range of the space
            return int(state) % observation_space.n
        elif isinstance(observation_space, spaces.Dict):
            # Convert the instance to a dictionary with valid values for each
            # key in the space
            return {
                key: check_and_transform(observation_space.spaces[key], state[key])
                for key in observation_space.spaces
            }
        # Add more cases for other types of spaces as needed
        else:
            # Raise an exception if the instance cannot be transformed
            raise ValueError(f"Cannot transform {state} for {observation_space}")


def prepare_path(checkpoint_folder):
    return str(
        Path(__file__).absolute().parent.parent / "checkpoints" / checkpoint_folder
    )


def prepare_algo(checkpoint_folder):
    # Load the algorithm information from the checkpoint
    with open(checkpoint_folder + "/algorithm_state.pkl", "rb") as f:
        data = pickle.load(f)

    # Unfreeze the AlgorithmConfig to make changes to env
    algo = data["config"].copy(copy_frozen=False)

    # Change the sim env to None (not needed for inference), and rebuild the algorithm
    algo = algo.environment(
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
        self.agent = prepare_algo(check_path)

    async def __call__(self, request):
        json_input = await request.json()
        state = check_and_transform(observation_space, json_input["state"])
        # Set explore to false or the agent will not be deterministic
        action = self.agent.compute_single_action(state, explore=False)
        return action


agent = ServeAgent.bind()  # type: ignore
