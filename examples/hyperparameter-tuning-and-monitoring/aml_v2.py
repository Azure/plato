from azure.ai.ml import MLClient, MpiDistribution, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# connect to the workspace
ml_client = MLClient.from_config(DefaultAzureCredential())

# set up pytorch environment
compute_env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="conda.yml",
    name="ray-mlflow",
)

for env in [
    "CartPole-v1",
    "MountainCarContinuous-v0"
    "Pendulum-v1"
    "Acrobot-v1"
    "LunarLander-v2"
    "BipedalWalker-v3",
]:
    # define the command
    command_job = command(
        name=f"mlflow-run-{env}",
        experiment_name="mlflow-experiments-{env}",
        code="./src",
        command="python mlflow_run.py --env-name ${{inputs.env_name}}",
        environment=compute_env,
        inputs={"env_name": env},
        compute="cpu",
        distribution=MpiDistribution(process_count_per_instance=2),
    )

    returned_job = ml_client.jobs.create_or_update(command_job)
