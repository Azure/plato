# Plato Toolkit documentation

Learn how to train and deploy RL agents at scale with Ray RLlib and Azure Machine Learning (AML).

## Overview

* Prerequisites
    * [Create Azure Resources](#create-azure-resources)
    * [AML Environment Setup](#aml-environment-setup)
    * [Custom Simulation Environment](#custom-simulation-environment-with-gymnasium)
* Samples
    * [Simple Adder](https://github.com/Azure/plato/tree/main/examples/first_job_on_AML):  A minimal working example of a Python simulator that can be connected to RLlib and used to train an agent on AML. You can think of it as a "Hello World" sample.
* User Guides



## Create Azure Resources
To use this toolkit, you'll need the following:

* [An Azure subscription and a resource group](https://portal.azure.com/)
* [An AML workspace](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#create-the-workspace)
* [An AML compute cluster](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#create-compute-clusters)

You can also create these using the [AML Python SDK](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=python).

### Selecting a Compute Cluster Size

There is no definitive answer to how to select a compute cluster size for RL, as it depends on many factors (e.g., your project budget, simulation environment, and model architecture). However, some general guidelines are:

* Unless you have a compute intensive RL model (e.g., a large deep residual network), we recommend selecting a general purpose CPU VM.
* Choose a _minimum_ number of nodes that defines how many nodes are always running and ready for your jobs.
> We recommend selecting 0 as your minimum  to de-allocate the nodes when they aren't in use. Any value larger than 0 will keep that number of nodes running and incur cost.
* Choose a _maximum_ number of nodes that defines how many nodes can be added to scale up your training when needed.
* Avoid large unexpected Azure costs by familiarizing yourself with the [size and cost of Azure VMs](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/series/).
* If you are still unsure which VM to select, a cluster with 6 CPU cores and 64GB RAM should be a good starting point for most RL workloads using a Python simulation environment. You can also [monitor your job's resource utilization in AML studio](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-track-monitor-analyze-runs?view=azureml-api-2#monitor-your-job-resources-preview) during experiment runs and adjust your VM size accordingly.

Once you have an AML workspace that contains a compute cluster ready to go, the next step is to [set up an AML environment](#environment-setup) to add your Python package dependencies.


## AML Environment Setup

A user-managed AML environment specifies the Python packages required to run your simulation and Ray RLlib code. You can follow the [how-to guide on configuring AML environments](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment) or try our preferred method below using a conda file.

We've provided a `conda.yml` file and Azure CLI command that you can use to create an environment for the [Simple Adder sample](https://github.com/Azure/plato/tree/main/examples/first_job_on_AML) within this toolkit. Simply save the file and run the CLI command from the same location. For more detailed instructions, you can follow the guide to [create an environment from a conda file in AML studio](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-in-studio) or with the [AML Python SDK](https://learn.microsoft.com/en-us/cli/azure/ml/environment?view=azure-cli-latest#az-ml-environment-create).

```yaml
# conda.yml
channels:
- anaconda
- conda-forge
dependencies:
- python=3.8.5
- pip=22.3.1
- pip:
   # Dependencies for Ray on AML
   - azureml-mlflow
   - azureml-defaults
   - ray-on-aml
   - ray[data]==2.3.0
   - ray[rllib]==2.3.0
   # Dependencies for RLlib
   - tensorflow==2.11.1
   # Dependencies for the Simulator
   - gymnasium
   - numpy==1.24.2
```

Azure CLI command:
``` Azure CLI
az ml environment create --name aml-environment --conda-file conda.yml --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 --resource-group $YOUR_RESOURCE_GROUP --workspace-name $YOUR_WORKSPACE
```

## Custom Simulation Environment with Gymnasium
Before you can train an RL agent on AML, your simulation environment needs to be compatible with Ray RLlib. For Python simulation environments, we recommend modifying your code to create a custom Gymnasium environment by following [this tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) and using the samples in this repository for reference. The basic steps are:

* Implement the [`gymnasium.Env`](https://gymnasium.farama.org/api/env) interface and define methods for _reset()_ and _step()_.
* Specify the _action_space_ and _observation_space_ attributes during initialization using [`gymnasium.spaces`](https://gymnasium.farama.org/api/spaces/fundamental/#fundamental-spaces).
* Ensure that the actions and observations returned by _reset()_ and _step()_ have the same shape and dtype as the _action_space_ and _observation_space_ defined.
    * For example, if your observation_space is a `gymnasium.spaces.Box` space with shape=(1,) and dtype=np.float32, you should make sure that your observation is a numpy array of shape (1,) and dtype np.float32.


After you complete the integration, we suggest that you confirm it can run on your local machine before scaling on AML. Our [Simple Adder sample](https://github.com/Azure/plato/tree/main/examples/first_job_on_AML) provides you with the code to run it both locally and on AML.



<!-- ## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files. -->
