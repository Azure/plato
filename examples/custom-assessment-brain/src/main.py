"""
Main script for assessing an agent with custom initial conditions on Azure ML.

You can test this code locally by launching the script with the flag
``--test-local``. This will run the script locally using Ray.

Note that RLlib Algorithm instances are not serializable by Ray. Therefore,
we cannot use ray.put() to pass them as arguments to remote functions.
Instead, we will use RLlib’s checkpointing mechanism to save and load the
Algorithm state for each worker.
"""
import argparse
import copy
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import ray
from policies import brain_policy
from ray.tune.registry import register_env
from ray_on_aml.core import Ray_On_AML
from sim import (
    SimpleAdder as SimEnv,  # IMPORTANT: Change to import your own environment
)

# IMPORTANT: sim_env must be equal to the name used to register the environment when
# training the agent.
register_env("sim_env", lambda conf: SimEnv(conf))


@ray.remote
class EpisodeRunner:
    """Class to run an episode of the simulation using a trained agent's policy."""

    def __init__(
        self,
        brain_url,
    ):
        """Initialize the episode runner by instantiating the agent.

        Args:
            brain_url (str): The URL of the docker container running the brain.
        """
        # Create a unique client id for the brain
        myClientId = str(uuid.uuid4())
        # Create the brain endpoint
        self.brain_endpoint = f"{brain_url}/v2/clients/{myClientId}/predict"

        # Initialize the simulation environment
        # IMPORTANT: Change to import your own environment
        self.sim = SimEnv(env_config={})

    def episode_to_df(
        self, states, actions, rewards, terminated, truncated, n_steps
    ) -> pd.DataFrame:
        """Create Pandas dataframe from episode steps.

        Args:
            states (dict): The states of the simulation at each step.
            actions (dict): The actions of the agent at each step.
            rewards (list): The rewards of the agent at each step.
            terminated (list): Whether the episode terminated at each step.
            truncated (list): Whether the episode was truncated at each step.
            n_steps (int): The final number of steps in the episode.

        Returns:
            pd.DataFrame: The dataframe containing the states, actions, episode id,
            step number, and initial condition.
        """
        # Create a dictionary for the results. Prepend "state_" to state columns.
        df = pd.DataFrame(states).add_prefix("state_")

        # Add action columns
        df = pd.concat([df, pd.DataFrame(actions).add_prefix("action_")], axis=1)

        # Add initial condition columns
        # Check the type of the initial_condition argument
        if isinstance(self.initial_condition, dict):  # If it is a dictionary
            # Use the dictionary keys and values to create new columns
            df[[f"config_{k}" for k in self.initial_condition.keys()]] = list(
                self.initial_condition.values()
            )
        elif isinstance(self.initial_condition, list):  # If it is a list
            # Use the list index and values to create new columns
            df[
                [f"config_{i}" for i in range(len(self.initial_condition))]
            ] = self.initial_condition
        else:  # If it is neither a list nor a dictionary
            # Raise an exception
            raise TypeError("The config argument must be a list or a dictionary")

        # Add reward, terminated, truncated, episode id, and step id columns
        df["reward"] = pd.Series(rewards)
        df["reward_cumsum"] = df["reward"].cumsum()
        df["terminated"] = pd.Series(terminated)
        df["truncated"] = pd.Series(truncated)
        df["episode_id"] = self.episode_id
        df["step_id"] = pd.Series(np.arange(n_steps + 2, dtype=int))

        return df

    def run_episode(
        self,
        initial_condition: Dict[str, int],
        episode_id: int = 0,
        max_steps: int = 100,
    ) -> pd.DataFrame:
        """Run an episode for the initial conditions using the agent policy and the
        simulator.

        Args:
            initial_condition (dict): The initial condition for the simulation.
            episode_id (int, optional): The id of the episode. Defaults to 0.
            max_steps (int, optional): The maximum number of steps for the
                episode. Defaults to 100.

        Returns:
            pd.DataFrame: The dataframe containing the results of the episode run.
        """
        # assign values to object properties
        self.initial_condition = initial_condition
        self.episode_id = episode_id
        self.max_steps = max_steps

        # Reset state with initial condition and return observable state
        observable_state, _ = self.sim.reset(options=self.initial_condition)

        # Initialize lists for episode logging
        states = []
        actions = []
        rewards = []
        terminated = []
        truncated = []
        # Test for max_steps
        for step in range(self.max_steps):
            # Get full current state and save it
            full_state = copy.deepcopy(self.sim.state)
            states.append(full_state)
            # Compute action based on current state
            action = brain_policy(observable_state, self.brain_endpoint)
            # Compute and update all states based on agent action
            observable_state, reward, terminate, truncate, _ = self.sim.step(action)
            # Save the step results
            rewards.append(reward)
            actions.append(action)
            terminated.append(terminate)
            truncated.append(truncate)
            if terminate or truncate:
                # If the episode is terminated or truncated, break the loop
                break
        # Save the final state
        full_state = (
            self.sim.state.copy() if type(self.sim.state) == dict else self.sim.state
        )
        states.append(full_state)
        return self.episode_to_df(states, actions, rewards, terminated, truncated, step)


def run_episodes(brain_url, initial_conditions_json):
    # Load episode initial conditions from json file
    json_path = Path(__file__).absolute().parent.parent / initial_conditions_json
    initial_conditions = json.load(open(json_path, "r"))

    # Create an actor instance for each worker. Initialize the brain endpoint and
    # simulation environment
    num_workers = 4
    actors = [EpisodeRunner.remote(brain_url=brain_url) for _ in range(num_workers)]

    # Assign episodes to actors in a round-robin fashion.
    results = []
    for id, init_cond in enumerate(initial_conditions):
        actor = actors[id % num_workers]
        result_id = actor.run_episode.remote(init_cond, id)
        results.append(result_id)

    # Get the results from the actors.
    results_df = pd.DataFrame()
    for result_id in results:
        episode_df = ray.get(result_id)
        results_df = pd.concat([results_df, episode_df], axis=0)

    # outputs can be found in AML Studio under the "Outputs + Logs" tab of your job
    outdir = "./outputs"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    results_df.to_csv(
        path_or_buf=outdir + "/assessment_logs.csv", mode="w", header=True, index=False
    )
    print("Results saved in directory ./outputs/assessment_logs.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-local", action="store_true", default=True)
    parser.add_argument("--brain-url", action="store", default="http://localhost:5000")
    parser.add_argument("--input-json", action="store", default="init_conditions.json")
    args = parser.parse_args()

    # Test locally
    if args.test_local:
        run_episodes(brain_url=args.brain_url, initial_conditions_json=args.input_json)
        sys.exit()

    # Test on AML compute
    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()

    if ray:
        print("head node detected")
        ray.init(address="auto")
        print(ray.cluster_resources())
        run_episodes(args.checkpoint_folder, args.input_json)
    else:
        print("in worker node")
