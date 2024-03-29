# Getting Started on AML

In this folder we show how to get started training an RL agent on Azure ML
with a custom Gymnasium environment ("Simple Adder").

### What this sample covers

- Integration of a custom Gymnasium simulation environment with RLlib
- How to train an agent using the simulation environment on AML

### What this sample does not cover

- How to optimize the training algorithm for best performance
- How to visualize the performance of the agent
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
az ml environment create --name aml-environment --conda-file conda.yml --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 --resource-group $YOUR_RESOURCE_GROUP --workspace-name $YOUR_WORKSPACE
```

## What is being "simulated"?
The simulation in this sample (`./scr/sim.py`) is intentionally very simple.
During an episode, at each step, the simulation adds the action ("addend") to
its state "value". Note that addend can be negative, which will cause value
to decrease.

During training, the agent learns to adjust the addend action to achieve a
state value of 50.

## Run Locally

As a preliminary step, you should check that the simulation works on your
local machine to save precious development time.
The ``main.py`` script in the ``src`` folder allows you to test locally with
the following command:

```bash
python main.py --test-local
```

## Tutorial: Run on AML
After you checked that the simulation works properly, follow these steps to
train an RL agent on AML:

1. Modify the ``job.yml`` file by changing the name of the AML ``environment``
   and ``compute`` to be the same as those you created in the prerequisites
   section.

2. Launch the job using the Azure CLI:
```
az ml job create -f job.yml --workspace-name $YOUR_WORKSPACE --resource-group $YOUR_RESOURCE_GROUP
```

3. Check that it is running by finding the job you launched in [AML
   studio](https://ml.azure.com/). You should see that ``ray`` is writing
   logs in the *Outputs + logs* tab in the ``user_logs`` folder.

4. Once the job is completed, the model checkpoints can also be found in AML
   studio under the *Outputs + Logs* tab of your job
   in the ``outputs`` folder.

## Next Steps
Congratulations! You've trained your first agent at scale on Azure. Now that
you understand the basics of running an experiment, try our more advanced
examples that show you how to modify your algorithm and evaluate/deploy a
trained agent.
