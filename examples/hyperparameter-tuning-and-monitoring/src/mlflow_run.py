import argparse

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from ray import air, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler, MedianStoppingRule
from ray.tune.stopper import TrialPlateauStopper
from ray_on_aml.core import Ray_On_AML


def run(
    num_tune_samples: int = 10, env_name: str = "CartPole-v1", scheduler: str = "median"
) -> tune.ResultGrid:
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
    scheduler: str, optional
        Name of the scheduler to use, option of median or asha, by default "median"

    This example uses the `MLflowLoggerCallback` from Ray Tune to log the results
    to MLflow. The MLflow integration allows you to log all the
    artifacts produced by Ray Tune, such as the model checkpoints, to MLflow. For
    more information, see the documentation:
    https://docs.ray.io/en/releases-2.3.1/tune/api/doc/
    ray.air.integrations.mlflow.MLflowLoggerCallback.html

    This example uses a fixed starting value for the hyperparameters. The
    hyperparameters are then tuned using the scheduler specified by `scheduler`.
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

    if scheduler == "median":
        tune_sched = MedianStoppingRule(
            time_attr="training_iteration",
            metric="episode_reward_mean",
            mode="max",
            grace_period=10,
            min_samples_required=10,
        )
    elif scheduler == "asha":
        tune_sched = ASHAScheduler(
            time_attr="training_iteration",
            metric="episode_reward_mean",
            mode="max",
            max_t=1000,
            grace_period=10,
            reduction_factor=3,
            brackets=1,
        )
    else:
        raise ValueError(
            f"Unknown scheduler provided: {scheduler}. Valid options are median or asha"
        )

    # define the stopping criteria for training
    # see available stopping criteria in the ray docs:
    # https://docs.ray.io/en/latest/tune/api/stoppers.html
    # we stop when a trial plateaus
    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.stopper.TrialPlateauStopper.html
    stopping_criteria = TrialPlateauStopper(metric="episode_reward_mean", mode="max")
    # another example: 100 training gradient steps or when the average
    # episode reward during a training batch reaches 300
    # stopping_criteria = {"training_iteration": 100, "episode_reward_mean": 300}

    # Get current run ID from MLFlow and pass it the callback
    current_run = mlflow.active_run()
    if current_run is None:
        current_run = mlflow.start_run()

    tuner = tune.Tuner(
        # the algorithm/trainable to be tuned
        "PPO",
        tune_config=tune.TuneConfig(
            # metric="episode_reward_mean",
            # mode="max",
            scheduler=tune_sched,
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
            "scaling_config": air.ScalingConfig(
                num_workers=10, resources_per_worker={"cpu": 2}
            ),
            # "num_cpus": 1,  # number of CPUs to use per trial
            # "num_gpus": 0,  # number of GPUs to use per trial
            # For DiagGaussian action distributions, make the second half of the model
            # outputs floating bias variables instead of state-dependent. This only
            # has an effect is using the default fully connected net.
            # does not work for non-continuous action spaces
            # "model": {"free_log_std": True},
            # These params are tuned from a fixed starting value.
            "lambda": tune.choice([0.9, 0.95, 1.0]),
            "clip_param": tune.choice([0.1, 0.2, 0.3]),
            "lr": tune.choice([1e-3, 5e-4, 1e-4]),
            # These params start off randomly drawn from a set.
            "num_sgd_iter": tune.choice([10, 20, 30]),
            # "sgd_minibatch_size": tune.choice([128, 512, 2048]),
            "sgd_minibatch_size": 2048,
            # "train_batch_size": tune.choice([10000, 20000, 40000]),
            "train_batch_size": 40000,
        },
        # MLFlow callback uses parent_run_id and tracks all hyperparameter
        # runs as child jobs
        run_config=air.RunConfig(
            stop=stopping_criteria,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=20,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max",
            ),
            callbacks=[
                MLflowLoggerCallback(
                    tags={MLFLOW_PARENT_RUN_ID: current_run.info.run_id},
                    experiment_name="hpt_ppo",
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

    parser.add_argument(
        "--scheduler",
        type=str,
        default="median",
        help="Scheduler to use for training. Options are median or asha",
    )
    args, _ = parser.parse_known_args()

    if not args.test_local:
        ray_on_aml = Ray_On_AML()
        ray = ray_on_aml.getRay()

        if ray:
            print("head node detected")
            ray.init(address="auto")
            print(ray.cluster_resources())
            run(args.num_tune_samples, args.env_name, args.scheduler)
        else:
            print("in worker node")

    else:
        run(args.num_tune_samples)
