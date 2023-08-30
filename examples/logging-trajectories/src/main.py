import argparse
import csv
import datetime as dt
from pathlib import Path

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray_on_aml.core import Ray_On_AML

parser = argparse.ArgumentParser()
parser.add_argument("--storage-path", type=Path)
args = parser.parse_args()


class TrajectoryCallback(DefaultCallbacks):
    fname = None

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
    ):
        obs = postprocessed_batch["obs"]
        actions = postprocessed_batch["actions"]
        episode_id = postprocessed_batch["eps_id"]
        rewards = postprocessed_batch["rewards"]
        terminated = postprocessed_batch["terminateds"]
        truncated = postprocessed_batch["truncateds"]
        step_id = postprocessed_batch["t"]

        if self.fname is None:
            output_name = f"{dt.datetime.now().isoformat(timespec='milliseconds')}.csv"
            self.fname = args.storage_path / output_name
            print(f"Saving trajectories is {self.fname}...")

        first = False if Path(self.fname).exists() else True

        # Ordering of the tuple should be consistent with the zipped variables below
        header = (
            "episode_id",
            "step_id",
            "state",
            "action",
            "reward",
            "terminated",
            "truncated",
        )

        # Check that headers match with file
        if not first:
            with open(self.fname, "r") as fp:
                reader = csv.reader(fp)
                file_head = tuple(next(reader))
                if file_head != header:
                    raise ValueError(f"Unexpected header in file {self.fname}")

        with open(self.fname, "a") as fp:
            writer = csv.writer(fp)
            if first:
                writer.writerow(header)
            for row in zip(
                episode_id, step_id, obs, actions, rewards, terminated, truncated
            ):
                writer.writerow(row)


def train():
    # Define the algo for training the agent
    algo = (
        PPOConfig()
        .callbacks(TrajectoryCallback)
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=0)
        # Set the training batch size to the appropriate number of steps
        .training(train_batch_size=4_000)
        .environment(env="CartPole-v1")
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
    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()

    if ray:
        print("head node detected")
        ray.init(address="auto")
        print(ray.cluster_resources())
        train()
    else:
        print("in worker node")
