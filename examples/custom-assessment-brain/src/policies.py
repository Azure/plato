import json
from typing import Dict

import numpy as np
import requests
from ray.rllib.algorithms import Algorithm


class NumpyEncoder(json.JSONEncoder):
    # Encode numpy arrays to json compatible format
    def default(self, obj):
        # check if obj is numpy array, if yes convert to list
        if isinstance(obj, np.ndarray):
            obj_list = obj.tolist()
            if len(obj_list) == 1:
                return obj_list[0]
        return json.JSONEncoder.default(self, obj)


def brain_policy(state: Dict[str, float], prediction_endpoint: str):
    """Get the action from the brain running in a docker container.

    Args:
        state (dict): The state of the simulation.
        prediction_endpoint (str): The URL of the brain's prediction endpoint.

    Returns:
        action: The action to take in the simulation.
    """
    headers = {"Content-Type": "application/json"}
    # Convert numpy arrays to lists
    rm_arrays = json.dumps(state, cls=NumpyEncoder)
    state = {"state": json.loads(rm_arrays)}
    # Send request to brain endpoint and return response
    response = requests.post(
        prediction_endpoint, data=json.dumps(state), headers=headers
    )

    return response.json()["concepts"]["Concept"]["action"]


def agent_policy(state: Dict[str, float], agent: Algorithm):
    """Get the action from the RLlib agent."""
    return agent.compute_single_action(state, explore=False)
