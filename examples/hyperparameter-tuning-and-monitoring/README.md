# Logging Models with AzureML and MLFlow

This folder demonstrates an example of using MLFlow to log models and data with AzureML.

## How to Run

The source for running the experiment is under `src/mlflow_run.py`. The script defines a single function: `run()` which can be run locally or on AzureML using the `Ray-on-AML` library.

### Run Locally

You can run the script locally for testing and experimentation. The `smoke-test` option ensures each run is only for a single training iteration to save time.

```bash
python mlflow_run.py --test-local --smoke-test
```

## Prerequisites

- Install the Azure CLI and the ML extension
- Create an AML workspace
- Create a compute cluster in your AML workspace
- Create an AML environment using the conda file provided: ``conda.yml`` by running the following command:

```bash
az ml environment create --name ray-mlflow-env --conda-file conda.yml --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 --resource-group $YOUR_RESOURCE_GROUP --workspace-name $YOUR_WORKSPACE
```

Note that the environment in this sample is slightly different than the previous sample, so you will need to create a new environment.

## Run the Job on AML

To run the experiment, you can use the included ``job.yml``. The file
defines all parameters needed to successfully launch the job. Please
remember to change the name of the ``environment`` and ``compute`` to be the
same as those you created in the prerequisites section. Please see the provided ``job.yml`` for an example.
Note that the `environment` parameter has a prefix `azureml` and a suffix with the version, in this case `@latest`. Please refer to the [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2) docs for more information.

Launch the experiment using the Azure CLI and the `ml` extension:

```bash
az ml job create -f job.yml --workspace-name $YOUR_WORKSPACE --resource-group $YOUR_RESOURCE_GROUP
```

## Monitor Your Job

After submitting your experiment using the Azure CLI and your yaml file, AML will create an experiment and spin up multiple jobs, one for each sweep over your hyperparameter space. You can monitor the jobs on the AML studio website by locating your experiment. The experiment should be listed under the display name of your `yaml` file above. If you have hyperparameter tuning enabled, you will see separate model files for each run under the *Outputs + logs*. You can also view the metrics for each job in the experiment under the *Metrics* tab.

## Download Model Files for Local Testing

Upon completion of your experiment, you can download the model files to your local machine for testing and assessment. The script `get_mlflow_artifacts.py` provides function for querying your MLFlow registry on AzureML and downloads the top performing policy to your local machine. In order to run this script, first create a [workspace configuration file](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment?view=azureml-api-2#local-and-dsvm-only-create-a-workspace-configuration-file) in your local directory to run this script.
