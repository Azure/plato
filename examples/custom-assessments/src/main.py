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
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import ray
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
        checkpoint_folder: str,
    ):
        """Initialize the episode runner by instantiating the agent.

        Args:
            checkpoint_folder (str): The name of the folder where the checkpoint is
            located.
        """
        # Instantiate the trained agent from the checkpoint folder
        checkpoint_path = Path(__file__).absolute().parent.parent / checkpoint_folder
        # self.agent = Algorithm.from_checkpoint(checkpoint_path)

        # Restore the RLlib agent by unpickling and modifying config to reset
        # num_workers. Workers for the Ray Remote API are set in run_episodes() below.
        with open(Path(checkpoint_path) / "algorithm_state.pkl", "rb") as fp:
            data = pickle.load(fp)
        agent_config = data["config"].copy(copy_frozen=False)
        self.agent = agent_config.rollouts(
            num_rollout_workers=0,
        ).build()
        self.agent.restore(checkpoint_path)

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
        df = pd.concat([pd.DataFrame(dict) for dict in [states, actions]], axis=1)
        # Prepend "state_" to state columns and "action_" to action columns for clarity
        df.rename(
            columns=lambda x: "action_" + x if x in actions else "state_" + x,
            inplace=True,
        )
        for key, value in self.initial_condition.items():
            df["config_" + key] = value
        df["reward"] = pd.Series(rewards)
        df["reward_cumsum"] = df["reward"].cumsum()
        df["terminated"] = pd.Series(terminated)
        df["truncated"] = pd.Series(truncated)
        df["episode_id"] = self.episode_id
        df["step_id"] = pd.Series(np.arange(n_steps, dtype=int))

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
            full_state = self.sim.state.copy()
            states.append(full_state)
            # Compute action based on current state
            action = self.agent.compute_single_action(observable_state, explore=False)
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
        states.append(self.sim.state.copy())
        return self.episode_to_df(states, actions, rewards, terminated, truncated, step)


def run_episodes(checkpoint_folder, initial_conditions_json):
    # Load episode initial conditions from json file
    json_path = Path(__file__).absolute().parent.parent / initial_conditions_json
    initial_conditions = json.load(open(json_path, "r"))

    # Create an actor instance for each worker. Initialize the agent and simulation
    # environment
    num_workers = 4
    actors = [EpisodeRunner.remote(checkpoint_folder) for _ in range(num_workers)]

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
    parser.add_argument("--test-local", action="store_true", default=False)
    parser.add_argument(
        "--checkpoint-folder", action="store", default="checkpoints/checkpoint_000010"
    )
    parser.add_argument("--input-json", action="store", default="init_conditions.json")
    args = parser.parse_args()

    # Test locally
    if args.test_local:
        run_episodes(args.checkpoint_folder, args.input_json)
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
