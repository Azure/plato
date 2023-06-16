import json

import requests
from gymnasium import Env, spaces

from platotk.logger import log
from platotk.serialize import GymEncoder, check_and_transform

BASE_URL = "http://localhost:8000"


class SimWrapper(Env):
    def __init__(self, env_config):
        self.base_url = BASE_URL

        self.action_space = spaces.Dict(
            {
                "numResourceA": spaces.Discrete(20, start=1),
                "numResourceB": spaces.Discrete(20, start=1),
                "processTime": spaces.Box(1.0, 12.0),
                "conveyorSpeed": spaces.Box(0.01, 1.0),
            }
        )
        self.observation_space = spaces.Dict({"arrivalRate": spaces.Box(0.5, 2.0)})
        self.config = {
            "arrivalRate": 0.5,
            "sizeBufferQueues": 45,
        }

    def reset(self, *, seed=None, options=None):
        log.debug("Reset send.")
        resp = requests.post(self.base_url + "/reset", json={"config": self.config})
        state = resp.json()
        log.debug("Reset response.")
        return check_and_transform(self.observation_space, state), {}

    def step(self, action):
        log.debug("Send step.")
        json_action = json.dumps({"action": action}, cls=GymEncoder)
        resp = requests.post(
            self.base_url + "/step",
            data=json_action,
            headers={"Content-Type": "application/json"},
        )
        state = resp.json()
        log.debug("Step response.")

        return (
            check_and_transform(self.observation_space, state),
            self.reward(state),
            self.terminal(state),
            self.truncate(state),
            {},
        )

    def reward(self, state):
        return -state.get("costPerProduct") - 1000 * state.get("exceededCapacityFlag")

    def terminal(self, state):
        return state.get("exceededCapacityFlag") == 1 or state.get("simTimeMonths") >= 6

    def truncate(self, state):
        return self.terminal(state)
