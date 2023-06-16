import json

import gymnasium.spaces as spaces
import numpy as np


class GymEncoder(json.JSONEncoder):
    """
    Encode a Gymnasium-like state to JSON.

    A big caveat of the function is that it encodes an array of shape (1,)
    into a scalar. We have to do this because Gymnasium only supports scalars
    when using spaces.Discrete. Everything that is of type spaces.Box is
    treated as an array. So having a float scalar is not possible.
    """

    def default(self, obj):
        if isinstance(obj, np.number) or isinstance(obj, np.bool_):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            # Treat shape (1,) as scalars
            if obj.shape == (1,):
                return obj.item()
            else:
                return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def check_and_transform(observation_space, state):
    """
    Check and transform a observation state instance for a given observation space.

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
