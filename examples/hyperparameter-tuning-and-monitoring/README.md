# Hyperparameter Tuning and Monitoring Training Jobs

This folder demonstrates an example of using MLFlow to log models and data with AzureML, and how to use the Ray Tune library to sweep over models for hyperparameter tuning.

### What this sample covers

- How to sweep over hyperparameters to optimize the training algorithm for best performance
- How to visualize the performance of the agent

### What this sample does not cover

- How to evaluate the agent
- How to deploy the agent


## Prerequisites

- Install the Azure CLI on your machine:
```
pip install azure-cli
```
- Add the ML extension:
```
az extension add -n ml
```
- [Create an AML workspace and compute cluster](https://azure.github.io/plato/#create-azure-resources)
- Create an AML environment using the conda file provided: ``conda.yml`` by running the following command:

```bash
az ml environment create --name ray-mlflow-env --conda-file conda.yml --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 --resource-group $YOUR_RESOURCE_GROUP --workspace-name $YOUR_WORKSPACE
```

Note that the environment in this sample is slightly different than the _Getting Started on AML_ sample, so you will need to create a new environment.

## What is being "simulated"?
This example uses the classic ["CartPole-v1"](https://gymnasium.farama.org/environments/classic_control/cart_pole/) simulation environment. The example runs Ray Tune's [population based training (PBT)](https://docs.ray.io/en/latest/_modules/ray/tune/schedulers/pbt.html) to simultaneously train and optimize a group of agents - regularly testing the agents, replicating the top performers, and perturbing their hyperparameters. The MLflow integration allows logging of all artifacts produced by Ray Tune, such as the model checkpoints, to MLflow.

A user guide providing more details on hyperparameter tuning with PBT is available in [user_guides/hyperparameter-tuning.md](../../docs/user_guides/hyperparameter-tuning.md).

## Run Locally
The source for running the experiment is under `src/mlflow_run.py`. The script defines a single function: `run()` which can be run locally or on AzureML.

You can run the script locally for testing and experimentation. You can select the  number of samples to run with the `--num-tune-samples` flag. For example, to run a single sample locally, run the following command:

```bash
python mlflow_run.py --test-local --num-tune-samples 1
```

You can also test out different environments, provided they work with the algorithm you have selected and are registered `gymnasium` environments by specifying the environment name with `env-name`:


```bash
python mlflow_run.py --test-local --num-tune-samples 1 --env-name CartPole-v1
```

## Tutorial: Run on AML

1. Modify the ``job.yml`` file by changing the name of the AML ``environment``
   and ``compute`` to be the same as those you created in the prerequisites
   section.
    - Note that the `environment` parameter has a prefix `azureml` and a suffix with the version, in this case `@latest`. Refer to the [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2) docs for more information.

2. Launch the experiment using the Azure CLI `ml` extension:

    ```bash
    az ml job create -f job.yml --workspace-name $YOUR_WORKSPACE --resource-group $YOUR_RESOURCE_GROUP
    ```

    - After submitting your experiment using the Azure CLI and your yaml file, AML will create an experiment and spin up multiple jobs, one for each sweep over your hyperparameter space.

3. Monitor the jobs on the [AML studio website](https://ml.azure.com/) by locating your experiment, which is listed under the display name of your `job.yml` file above.
    - If you have hyperparameter tuning enabled, you will see separate model files for each run under *Outputs + logs*.
    - You can also view the metrics for each job in the experiment under the *Metrics* tab.

4. Upon completion, download the model files to your local machine using the script `get_mlflow_artifacts.py`. This script allows you to query your MLFlow registry on AzureML and download the top performing policy.
    - In order to run this script, first create a [workspace configuration file](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment?view=azureml-api-2#local-and-dsvm-only-create-a-workspace-configuration-file) in your local directory.

## Next Steps
Now that you understand how to use MLflow to log models, you can:
- Optimize the hyperparameter tuning process with different search algorithms and spaces.
- Evaluate your best performing model on new episodes or metrics.
- Deploy your model to production or serve them on Azure.
