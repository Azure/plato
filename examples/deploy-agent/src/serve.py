from pathlib import Path

import gymnasium.spaces as spaces
import numpy as np
from ray import serve

from platotk.restore import restore_agent

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


def check_and_transform(observation_space, state):
    """Check and transform a observation state instance for a given observation space.

    This function takes a state instance and an observation space object that defines
    the valid shape and dtype of the state data for the simulation environment. The
    function then checks if the state instance is already valid for the observation
    space, and if not, it tries to transform it into a valid form.

    Args:
        observation_space (gym.Space): The observation space of the sim env.
        state (JSON): The state instance to check and transform.

    Returns:
        state: The transformed state instance that is valid for the observation space.

    Raises:
        ValueError: If the state instance cannot be transformed for the observation
        space.
    """
    # Check if the instance is valid for the space
    if isinstance(state, np.ndarray) and observation_space.contains(state):
        # Return the instance as it is
        return state
    # Try to transform the instance based on the space type
    elif isinstance(observation_space, spaces.Box):
        # Convert the instance to a numpy array with the same
        # shape and dtype as the space
        return np.array(state, dtype=observation_space.dtype).reshape(
            observation_space.shape
        )
    elif isinstance(observation_space, spaces.Discrete):
        # Convert the instance to an integer
        return int(state)
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
