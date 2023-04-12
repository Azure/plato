# Simple Adder Sample

In this folder we show how to get started training an RL agent on Azure ML
with a custom Gymnasium environment.

### What this sample covers

- How to integrate a custom Gymnasium environment with RLlib
- How to train an agent using that environment on AML

### What this sample does not cover

- How to optimize the training algorithm for best performance
- How to visualize the performance of the agent
- How to evaluate the agent
- How to deploy the trained agent

## Prerequisites

- Install the Azure CLI and the ML extension
- Create an AML workspace
- Create a compute cluster in your AML workspace
- Create an AML environment using the conda file provided: ``conda.yml`` by
  running the following command:
```bash
az ml environment create --name aml-environment --conda-file conda.yml --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 --resource-group $YOUR_RESOURCE_GROUP --workspace-name $YOUR_WORKSPACE
```

## Run the experiment

To run the experiment, you can use the included ``job.yml``. The file
defines all parameters needed to successfully launch the job. Please
remember to change the name of the ``environment`` and ``compute`` to be the
same as those you created in the prerequisites section.

Launch the job using the Azure CLI, for instance with
```bash
az ml job create -f job.yml --workspace-name $YOUR_WORKSPACE --resource-group $YOUR_RESOURCE_GROUP
```

## Check that it runs

Find the job you launched in the Azure UI. You should see that ``ray`` is
writing logs in the *Outputs + logs* tab in the ``user_logs`` folder.

## Trained agent

Once the training is completed, the model checkpoints can be found in AML
Studio under the *Outputs + Logs* tab of your job in the ``outputs`` folder.
