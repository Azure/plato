# Module to test the sim locally
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from sim import SimpleAdder as SimEnv


def env_creator(env_config):
    return SimEnv(env_config)


register_env("sim_env", env_creator)
algo = (
    PPOConfig()
    # Setting workers to zero allows using breakpoints in sim for debugging
    .rollouts(num_rollout_workers=0)
    .resources(num_gpus=0)
    .environment(env="sim_env")
    .build()
)

# Train for 10 iterations
for i in range(10):
    result = algo.train()
    print(pretty_print(result))
