import argparse
import random

from ray import air, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import PopulationBasedTraining
from ray_on_aml.core import Ray_On_AML


def run() -> tune.ResultGrid:
    """Run Ray Tune with MLflow on AzureML or locally.
    This is an example of using Ray Tune and MLflow with Ray on AzureML.
    It is based on the PBT example from the Ray Tune documentation:
    https://docs.ray.io/en/releases-2.3.1/tune/examples/pbt_ppo_example.html

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

    This example uses a population size of 10. If the parameter `smoke-test` is
    provided, then the example finishes in a single iteration for testing purposes.

    This script can be run on AML or locally. If running locally, pass the parameter
    `--test-local` to the script. If running on AML, omit the parameter as is done
    in the provided `job.yml` file.

    # quick local test
    python mlflow_run.py --test-local --smoke-test

    Returns
    -------
    tune.ResultGrid
        A set of Result objects for interacting with Ray Tune results.
    """

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    hyperparam_mutations = {
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_sgd_iter": lambda: random.randint(1, 30),
        "sgd_minibatch_size": lambda: random.randint(128, 16384),
        "train_batch_size": lambda: random.randint(2000, 160000),
    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )

    # Stop when we've either reached 100 training iterations or reward=300
    stopping_criteria = {"training_iteration": 100, "episode_reward_mean": 300}

    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples=1 if args.smoke_test else 10,
        ),
        param_space={
            "env": "CartPole-v1",
            "kl_coeff": 1.0,
            "num_workers": 4,
            "num_cpus": 1,  # number of CPUs to use per trial
            "num_gpus": 0,  # number of GPUs to use per trial
            "model": {"free_log_std": True},
            # These params are tuned from a fixed starting value.
            "lambda": 0.95,
            "clip_param": 0.2,
            "lr": 1e-4,
            # These params start off randomly drawn from a set.
            "num_sgd_iter": tune.choice([10, 20, 30]),
            "sgd_minibatch_size": tune.choice([128, 512, 2048]),
            "train_batch_size": tune.choice([10000, 20000, 40000]),
        },
        run_config=air.RunConfig(
            stop=stopping_criteria,
            callbacks=[
                MLflowLoggerCallback(
                    experiment_name="pbt_ppo", tags=None, save_artifact=True
                )
            ],
        ),
    )
    results = tuner.fit()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--test-local", action="store_true", help="Test locally. If false runs on AML"
    )
    args, _ = parser.parse_known_args()

    if not args.test_local:
        ray_on_aml = Ray_On_AML()
        ray = ray_on_aml.getRay()

        if ray:
            print("head node detected")
            ray.init(address="auto")
            print(ray.cluster_resources())

            run()
        else:
            print("in worker node")

    else:
        run()
