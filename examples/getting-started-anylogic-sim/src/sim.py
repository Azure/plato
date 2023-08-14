import json
import os
import subprocess
import time
from pathlib import Path

import requests
from gymnasium import Env, spaces

from platotk.logger import log
from platotk.serialize import GymEncoder, check_and_transform


class SimWrapper(Env):
    def __init__(self, env_config):
        self.base_host = "localhost"
        self.env_id = int(f"{env_config.worker_index}{env_config.vector_index}")
        self.base_port = 8000 + self.env_id
        self.base_url = f"http://{self.base_host}:{self.base_port}"

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
        self.start_sim_framework()

    @staticmethod
    def find_unique_port(worker, vector):
        return 8000 + int(f"{worker}{vector}")

    def start_sim_framework(self):
        """Start Baobab API and external sim."""

        # Find the sim executable
        scripts = [script for script in Path(__file__).parent.rglob("*_linux.sh")]
        if len(scripts) > 1:
            raise RuntimeError(f"Too many Anylogic sims found: {scripts}")
        elif len(scripts) < 1:
            raise RuntimeError("No Anylogic sim found.")
        sim_exec = scripts.pop()

        os.environ["BAOBAB_NAMESPACE"] = str(self.env_id)

        # Launch Baobab
        subprocess.Popen(
            [
                "gunicorn",
                "--worker-class",
                "uvicorn.workers.UvicornWorker",
                "--bind",
                f"{self.base_host}:{self.base_port}",
                "platotk.baobab:app",
            ]
        )
        time.sleep(2)

        # Launch the sim that will connect to Baobab
        penv = {
            "SIM_API_HOST": self.base_url,
            "SIM_CONTEXT": "{}",
            "SIM_WORKSPACE": "dummy",
            "SIM_ACCESS_KEY": "dummy",
        }
        subprocess.Popen([sim_exec], env=penv)
        time.sleep(5)

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
