"""
Main script for training an agent on Azure ML.

You can test this code locally by launching the script with the flag
``--test-local``
"""
import argparse
import sys

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from sim import SimpleAdder as SimEnv

# Register the simulation as an RLlib environment.
register_env("sim_env", lambda config: SimEnv(config))


def train(local=False):
    # Define the algo for training the agent
    algo = (
        PPOConfig()
        # Modify also instance_count in job definition to use more than one worker
        # Setting workers to zero allows using breakpoints in sim for debugging
        .rollouts(num_rollout_workers=1 if not local else 0)
        .resources(num_gpus=0)
        # Set the training batch size to the appropriate number of steps
        .training(train_batch_size=4_000)
        .environment(env="sim_env")
        .build()
    )
    # Train for 10 iterations
    for i in range(10):
        result = algo.train()
        print(pretty_print(result))

    # outputs can be found in AML Studio under the "Outputs + Logs" tab of your job
    checkpoint_dir = algo.save(checkpoint_dir="./outputs")
    print(f"Checkpoint saved in directory {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-local", action="store_true", default=False)
    args = parser.parse_args()

    train(args.test_local)
