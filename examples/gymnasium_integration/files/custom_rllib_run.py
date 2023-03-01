
## Custom code to install or update libraries without having to build the image from scratch
#import subprocess
#import sys
#subprocess.check_call([sys.executable, "-m", "pip", "install", "XX"])

from ray_on_aml.core import Ray_On_AML

import ray.tune as tune
from ray.rllib import train

from utils.callbacks import MyCallback


# IMPORT OUR CUSTOM ENVIRONMENT
from custom_gym_wrapper import Gym_Wrapper
# TODO: Review parsing parameters to the sim through the wrapper.
# azaidi: Do all the scenario parameters go into the options arg? And should that be of type Dict?
#         We shall consider validating the config before it is passed
#         i.e., obj is dataclass, or a Dict[str, type] which specifies the type of each scenario parameter)?
#         It should help prevent errors when users provide invalid scenario parameter values during domain randomization.
env = Gym_Wrapper
tune.register_env('custom_gym_env', lambda config: env(config))
#tune.register_env('custom_gym_env', lambda: config, env(config))


# Run the job using ray.tune.run
if __name__ == "__main__":

    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()
    if ray:  # in the headnode
        # Parse arguments
        #train_parser = train.create_parser()

        # CREATE A CUSTOM PARSER
        # TODO: Review this
        import argparse
        train_parser = argparse.ArgumentParser(description = 'Training parser for ray job.')
        train_parser.add_argument('--run')
        train_parser.add_argument('--env')
        train_parser.add_argument('--config')
        train_parser.add_argument('--stop')

        args = train_parser.parse_args()
        
        # Load dictionaries from strings
        import json
        args.config = json.loads(args.config)
        args.stop = json.loads(args.stop)

        print("Algorithm config:", args.config)

        tune.run(
            run_or_experiment=args.run,
            config={
                "env": args.env,  #"WrappedCustomEnv-v0"
                "env_config": {"reset_config": {"noise_percentage": 10}},
                "num_gpus": args.config["num_gpus"],
                "num_workers": args.config["num_workers"],
                #"callbacks": {"on_train_result": callbacks.on_train_result},
                "callbacks": MyCallback,
                "sample_batch_size": 50,
                "train_batch_size": 1000,
                "num_sgd_iter": 2,
                "num_multi_gpu_tower_stacks": 2,
                "model": {"dim": 42},
            },
            stop=args.stop,
            local_dir='./logs')
    else:
        print("in worker node")
