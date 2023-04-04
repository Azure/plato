from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray_on_aml.core import Ray_On_AML
from sim import SimpleAdder as SimEnv


# Register the simulation as an RLlib environment.
def env_creator(env_config):
    return SimEnv(env_config)


register_env("sim_env", env_creator)

if __name__ == "__main__":
    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()

    if ray:
        print("head node detected")
        ray.init(address="auto")
        print(ray.cluster_resources())

        # Define the algo for training the agent
        algo = (
            PPOConfig()
            # Modify also instance_count in job definition to use more than one worker
            .rollouts(num_rollout_workers=1)
            .resources(num_gpus=0)
            # Set the training batch size to 4,000 steps
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

    else:
        print("in worker node")
