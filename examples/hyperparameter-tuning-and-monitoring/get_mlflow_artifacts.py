"""
View and download artifacts from AzureML run with MLFlow
"""

from typing import List, Optional

import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def list_experiments(path: Optional[str] = None) -> List:
    """List experiments in AzureML MLFlow

    Parameters
    ----------
    path : Optional[str], optional
        Path to AML workspace configuration, by default None

    Returns
    -------
    List
        A list of :py:class:`Experiment <mlflow.entities.Experiment>` objects.
    """
    ml_client = MLClient.from_config(path=path, credential=DefaultAzureCredential())
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiments = mlflow.search_experiments()

    return experiments


def get_top_run(
    experiment_name: str,
    save_path: str,
    path: Optional[str] = None,
    rank_metric: str = "metrics.episode_reward_mean",
):
    """Get top run from AzureML MLFlow experiment and download
    model artifacts locally.

    Parameters
    ----------
    experiment_name : str
        name of experiment to query from
    save_path : str
        local path to save model artifacts
    path : Optional[str], optional
        Path to AML workspace configuration, by default None

    Returns
    -------
    _type_
        _description_
    """
    ml_client = MLClient.from_config(path=path, credential=DefaultAzureCredential())
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    runs = mlflow.search_runs(experiment_names=[experiment_name])
    top_run_id = runs.sort_values(by=rank_metric, ascending=False)[
        ["run_id", rank_metric]
    ]["run_id"].values[0]
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(top_run_id)

    model_local_path = mlflow.artifacts.download_artifacts(
        run_id=top_run_id, artifact_path=artifacts[0].path, dst_path=save_path
    )

    return model_local_path


if __name__ == "__main__":
    experiments = list_experiments()
    # pick the experiment you ran
    experiment = experiments[-1].name
    local_path = "model_checkpoints"
    get_top_run(experiment, local_path)

    # If your environment is already registered, then you
    # can use the following to load the model
    # however, this will fail if you have an environment
    # that is not registered
    # this might take a while, ~= 10 - 60 seconds
    # from ray.rllib.algorithms.algorithm import Algorithm
    # model = Algorithm.from_checkpoint(local_path)

    # import ray
    # ray.shutdown()
