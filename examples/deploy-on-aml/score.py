import json
import logging
import os
from pathlib import Path

import gymnasium.spaces as spaces
import numpy as np

from platotk.restore import restore_agent_from_pickle
from platotk.serialize import GymEncoder, check_and_transform

# IMPORTANT: Change the following parameters to fit your problem
CHECKPOINT_FOLDER = "checkpoint_000010"
observation_space = spaces.Dict(
    {"value": spaces.Box(low=-float("inf"), high=float("inf"))}
)
action_space = spaces.Dict({"addend": spaces.Box(low=-10, high=10, dtype=np.int32)})


def init():
    """Initialize the agent from the checkpoints."""
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    checkpoint_folder = Path(os.getenv("AZUREML_MODEL_DIR")) / CHECKPOINT_FOLDER
    model = restore_agent_from_pickle(
        observation_space, action_space, checkpoint_folder
    )
    logging.info("Init complete")


def run(raw_state):
    """Invoke the agent and return the action for the given state."""
    logging.info("Request received.")
    try:
        logging.info(str(raw_state))
    finally:
        pass
    state = json.loads(raw_state)["state"]
    state = check_and_transform(observation_space, state)
    action = model.compute_single_action(state, explore=False)
    logging.info("Request processed")
    norm_action = json.loads(json.dumps(action, cls=GymEncoder))
    return norm_action
