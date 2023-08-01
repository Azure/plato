"""
Adapted from
https://github.com/ray-project/ray/blob/master/rllib/examples/curriculum_learning.py
to use the simple adder simulation environment.

Example of a curriculum learning setup using the `TaskSettableEnv` API
and the env_task_fn config.

This example shows:
  - Writing your own curriculum-capable environment using gym.Env.
  - Defining an env_task_fn that determines whether and which new task
    the env(s) should be set to (using the TaskSettableEnv API).
  - Using Tune and RLlib to curriculum-learn this env.

You can visualize experiment results in ~/ray_results using TensorBoard locally,
or via AML performance metrics if you run this script on AML.
"""
import argparse
import os
import sys

from azureml.core import Run
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env
from ray_on_aml.core import Ray_On_AML

# IMPORTANT: Remember to change it for your own simulation environment
from sim_curriculum_capable import SimpleAdder as CurriculumCapableEnv

register_env("curriculum_env", lambda config: CurriculumCapableEnv(config))


# Define an env_task_fn that returns a new task based on some criteria
def curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results (dict): The train results returned by Algorithm.train().
        task_settable_env (TaskSettableEnv): A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx (EnvContext): The env context object (i.e. env's config dict plus
            properties worker_index, vector_index and num_workers) used to setup the
            `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the current one.
    """
    # With each task, the initial state value will be between (50-2**exponent) &
    # (50+2**exponent)
    # Task 1: Randomly sample a number between 48 and 52
    # Task 2: Randomly sample a number between 46 and 54
    # We will thus increase the task number each time we hit the reward threshold
    # Define a reward threshold for each task
    reward_threshold = 0
    # Get the current task level
    task_exponent = task_settable_env.get_task()["exponent"]
    # Get the average episode reward over the last training iteration
    avg_reward = train_results["episode_reward_mean"]
    # If the average reward is above or equal to the threshold, increase the task's
    # exponent
    if avg_reward >= reward_threshold:
        # Increase the task level by 1
        return {"exponent": task_exponent + 1}
    else:
        # Keep the same task level
        return task_settable_env.get_task()


class CurriculumCallback(DefaultCallbacks):
    """A custom callback class that logs the current task of the environment to
    tensorboard and Azure ML.

    This class inherits from the DefaultCallbacks class provided by RLlib and overrides
    the on_episode_start and on_epoch_end methods to access the curriculum "task"
    information from the base environment and the episode object, and log it to both
    tensorboard and Azure ML.
    """

    def __init__(self):
        self.run = Run.get_context()

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        # Get the current task of the sim
        task = base_env.get_sub_environments()[env_index].get_task()
        # Log the task to tensorboard
        episode.custom_metrics["task"] = task["exponent"]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        print(
            "Algorithm.train() result: {} -> {} episodes".format(
                algorithm, result["episodes_this_iter"]
            )
        )
        # Log metrics to TensorBoard
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

        # Filter the results dictionary to only log metrics with the substring "episode"
        to_log = {
            k: v for k, v in result.items() if "episode" in k and "media" not in k
        }
        # Add the curriculum task to the dictionary
        to_log["task"] = result["custom_metrics"]["task_mean"]
        # Log metrics to Azure ML
        for k, v in to_log.items():
            self.run.log(name=k, value=v)


def train():
    # Define a config object with the desired parameters
    param_space = {
        "env": "curriculum_env",
        "env_task_fn": curriculum_fn,
        "framework": "torch",
        # IMPORTANT: Change num_workers to scale training
        "num_workers": 1,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "callbacks": CurriculumCallback,
    }

    stopping_criteria = {
        "training_iteration": 300,
        "timesteps_total": 100000,
        # "episode_reward_mean": 0,
    }

    # Build the algorithm from the config and pass it to the tune.Tuner constructor
    tuner = tune.Tuner(
        "PPO",
        param_space=param_space,
        run_config=air.RunConfig(
            stop=stopping_criteria,
            verbose=2,
        ),
    )

    results = tuner.fit()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-local", action="store_true", default=True)
    args = parser.parse_args()

    if args.test_local:
        train()
        sys.exit()

    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()

    if ray:
        print("head node detected")
        ray.init(address="auto")
        print(ray.cluster_resources())
        train()
        ray.shutdown()
    else:
        print("in worker node")
