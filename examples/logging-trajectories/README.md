# Logging Episode Trajectories

In this folder we will show how to log episode trajectories during training.


## Prerequisites

-   Install the Azure CLI on your machine:
```
pip install azure-cli
```
-   Add the ML extension:
```
az extension add -n ml
```
- [Create an AML workspace and compute cluster](https://azure.github.io/plato/#create-azure-resources)
-   Create an AML environment using the conda file provided: ``conda.yml`` by running the following command:
    ```bash
    az ml environment create --name aml-environment --conda-file conda.yml --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 --resource-group $YOUR_RESOURCE_GROUP --workspace-name $YOUR_WORKSPACE
    ```
-   This sample uses a builtin Gymnasium environment. To use your own
    simulation environment please read one of our getting started samples
    first

## How the sample is structured

This sample follows the usual structure. There is a ``job.yml`` that you can
use for sending a job to Azure ML, and a Python script in ``src`` that is
launched by the job.

The important steps for logging trajectories are the following.

### 1. Modify the job definition

In ``job.yml`` we have the following block:

```yaml
outputs:
  output_data:
    mode: rw_mount
    path: azureml://datastores/workspaceblobstore/paths/trajectories
    type: uri_folder
```

This block defines mount instructions for our Azure ML Datastore. Once the
job starts, the compute environment will have the datastore folder specified
in ``path`` available as a writing destination. The customizable parts of
``path`` are: ``workspaceblobstore`` (the name of an Azure ML datastore), and
``trajectories`` (the name of the folder where we will save the trajectories
in the datastore). The datastore *workspaceblobstore* should be available by
default in your Azure ML workspace. Should you want to, you can create a
custom [Azure ML
datastore](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-datastore?view=azureml-api-2&tabs=sdk-identity-based-access%2Csdk-adls-identity-access%2Csdk-azfiles-accountkey%2Csdk-adlsgen1-identity-access)
and replace ``workspaceblobstore``.

### 2. Modify the python files

In ``src/main.py``, two things are relevant for us. The first one is that the
script must accept a folder path as a parameter (in our case this is
``--storage-path``). This is passed in the ``job.yml`` and will be the folder
where we will save the trajectories for the training run. The second
important thing is the class named ``TrajectoryCallback``. In this class, we
define a function ``on_postprocess_trajectory`` that runs during training. In
this function we provide instructions on how to save the trajectories in the
datastore. Note, we added the ``.callback`` method in the algorithm
configuration (in our case ``PPOConfig``) to load the ``TrajectoryCallback``
functionality.

### 3. Run the job on AML

Launch the job using the Azure CLI:
```
az ml job create -f job.yml --workspace-name $YOUR_WORKSPACE --resource-group $YOUR_RESOURCE_GROUP
```

### 4. Get the final output

Once the job completes, a CSV file containing your training agent trajectories
will be stored in your datastore. To access and download it, open your Azure
Machine Learning (AML) workspace. In your workspace go to *Data*, *Datastores*,
*workspaceblobstore* and then *Browse*. In the trajectories folder you will
find the CSV. The name of the CSV is the time when the job started.
