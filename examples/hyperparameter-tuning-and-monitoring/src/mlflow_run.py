import argparse
import random

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from ray import air, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import PopulationBasedTraining
from ray_on_aml.core import Ray_On_AML


def run(num_tune_samples: int = 10, env_name: str = "CartPole-v1") -> tune.ResultGrid:
    """Run Ray Tune with MLflow on AzureML or locally.
    This is an example of using Ray Tune and MLflow with Ray on AzureML.
    It is based on the PBT example from the Ray Tune documentation:
    https://docs.ray.io/en/releases-2.3.1/tune/examples/pbt_ppo_example.html

    Parameters
    ----------
    num_tune_samples : int, optional
        Number of samples to run with Ray Tune, by default 10
    env_name : str, optional
        Name of the environment to use, by default "CartPole-v1"

    This example uses the `MLflowLoggerCallback` from Ray Tune to log the results
    to MLflow. The MLflow integration allows you to log all the
    artifacts produced by Ray Tune, such as the model checkpoints, to MLflow. For
    more information, see the documentation:
    https://docs.ray.io/en/releases-2.3.1/tune/api/doc/ray.air.integrations.mlflow.MLflowLoggerCallback.html#ray.air.integrations.mlflow.MLflowLoggerCallback

    This example uses a fixed starting value for the hyperparameters. The
    hyperparameters are then tuned using the Population Based Training (PBT)
    algorithm. The hyperparameters used for the tuning algorithm are specified
    in the `tune_config` dictionary, and the algorithm hyperparameters
    to be tuned are specified in the `param_space` dictionary with a
    function of the search space for each hyperparameter.

    You can control the population size using `num_tune_samples`.

    This script can be run on AML or locally. If running locally, pass the parameter
    `--test-local` to the script. If running on AML, omit the parameter as is done
    in the provided `job.yml` file.

    # quick local test
    python mlflow_run.py --test-local --num-tune-samples 10

    Returns
    -------
    tune.ResultGrid
        A set of Result objects for interacting with Ray Tune results.
    """

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough environment iteration to do a sgd step
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    # Define the resample distributions for the hyperparameters to mutate during PBT
    # these parameters should be from the
    # there are model-specific hyperparameters:
    # https://docs.ray.io/en/releases-2.3.0/rllib/rllib-algorithms.html#ppo
    # and general training hyperaparameters:
    # https://docs.ray.io/en/releases-2.3.0/rllib/rllib-training.html#specifying-training-options
    # which you can define here
    hyperparam_mutations = {
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_sgd_iter": lambda: random.randint(1, 30),
        "sgd_minibatch_size": lambda: random.randint(128, 16384),
        "train_batch_size": lambda: random.randint(2000, 160000),
    }

    # the scheduler we use for tuning is population based training
    # other schedulers:
    # https://docs.ray.io/en/releases-2.3.0/tune/api/schedulers.html
    # see
    # https://docs.ray.io/en/releases-2.3.0/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )

    # define the stopping criteria for training
    # here we stop after 100 training gradient steps or when the average
    # episode reward during a training batch reaches 300
    # Stop when we've either reached 100 training iterations or reward=300
    stopping_criteria = {"training_iteration": 100, "episode_reward_mean": 300}

    # Get current run ID from MLFlow and pass it the callback
    current_run = mlflow.active_run()
    if current_run is None:
        current_run = mlflow.start_run()

    tuner = tune.Tuner(
        # the algorithm/trainable to be tuned
        "PPO",
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            # the number of hyperparameters to sample
            num_samples=num_tune_samples,
        ),
        # specify the initial config input into the trainer
        # these are the initial samples used, which are then mutated by
        # the population based training algorithm if they are specified in
        # `hyperparam_mutations`.
        # the `num_workers` specifies the number of sample collection workers
        # that are used for gathering samples
        # the `num_cpus` specifies the number of CPUs for each training trial
        # here `num_workers=4` and `num_cpus=1` means we will use 5 cpus
        # if you want to run these trials concurrently, then you will need
        # CLUSTER_CPUS >= 5 x num_tune_samples
        # otherwise the PBT scheduler will round-robin between training each trial
        param_space={
            "env": env_name,
            "kl_coeff": 1.0,
            "num_workers": 4,
            "num_cpus": 1,  # number of CPUs to use per trial
            "num_gpus": 0,  # number of GPUs to use per trial
            # For DiagGaussian action distributions, make the second half of the model
            # outputs floating bias variables instead of state-dependent. This only
            # has an effect is using the default fully connected net.
            # does not work for non-continuous action spaces
            # "model": {"free_log_std": True},
            # These params are tuned from a fixed starting value.
            "lambda": 0.95,
            "clip_param": 0.2,
            "lr": 1e-4,
            # These params start off randomly drawn from a set.
            "num_sgd_iter": tune.choice([10, 20, 30]),
            "sgd_minibatch_size": tune.choice([128, 512, 2048]),
            "train_batch_size": tune.choice([10000, 20000, 40000]),
        },
        # MLFlow callback uses parent_run_id and tracks all hyperparameter
        # runs as child jobs
        run_config=air.RunConfig(
            stop=stopping_criteria,
            callbacks=[
                MLflowLoggerCallback(
                    tags={MLFLOW_PARENT_RUN_ID: current_run.info.run_id},
                    experiment_name="pbt_ppo",
                    save_artifact=True,
                )
            ],
        ),
    )
    results = tuner.fit()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-local", action="store_true", help="Test locally. If false runs on AML"
    )
    parser.add_argument(
        "--num-tune-samples",
        type=int,
        default=10,
        help="Number of times to sample from hyperparameter space",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="CartPole-v1",
        help="Registered gym environment to use for training",
    )
    args, _ = parser.parse_known_args()

    if not args.test_local:
        ray_on_aml = Ray_On_AML()
        ray = ray_on_aml.getRay()

        if ray:
            print("head node detected")
            ray.init(address="auto")
            print(ray.cluster_resources())

            run(args.num_tune_samples)
        else:
            print("in worker node")

    else:
        run(args.num_tune_samples)
