# Run an Anylogic sim in AML

In this folder we show how to get started training an RL agent on Azure ML
with an Anylogic sim.

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
- Create an AML environment using the docker instructions provided in the
  ``docker`` folder. Go in that folder and run the following command:
```bash
az ml environment create --name anylogic-env --build-context ./ --dockerfile-path Dockerfile --resource-group $YOUR_RESOURCE_GROUP --workspace-name $YOUR_WORKSPACE
```
- Have an Anylogic sim that supports Bonsai integration. See
  [this](https://github.com/microsoft/bonsai-anylogic) for details.

## Prepare the Anylogic sim

This sample was built around the sim that you can find
[here](https://github.com/microsoft/bonsai-anylogic/tree/master/samples/abca).
Unzip the content of ``export.zip`` to this location: ``src/export``. Make
sure that you have the script for launching the sim in a Linux environment
and that the name of the script ends with ``_linux.sh``.
Note that only one sim should be present in the ``src`` directory, otherwise
the job will fail.

## Test Locally

If you have Docker available and working on your machine, you could try to
test if everything starts up correctly on your machine before sending the job
to AML.
To do this:

1. Go into the ``docker`` folder and build the docker image:

```bash
docker build -t anylogic-sim .
```

2. Go back to the ``getting-started-anylogic-sim`` folder and run:
  ```bash
  docker run --rm -it -e LOG_LEVEL=debug -v $(pwd)/src:/opt/src anylogic-sim bash /opt/src/start.sh
  ```
  or if you are on Windows:
  ```powershell
  docker run --rm -it -e LOG_LEVEL=debug -v "%cd%"/src:/opt/src anylogic-sim bash /opt/src/start.sh
  ```


After some time you should see that training starts as the terminal will show
the communication between the sim, Baobab and RLlib. When that happens, you
can proceed to the next section. If you see errors, please try to fix them
before proceeding further.

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

## Modify the sample to use your own sim

If you have your own Anylogic sim, there are a couple of changes you have to
make.

In ``src/sim.py``:

- In ``SimWrapper.__init__``: modify ``self.action_space`` to be the possible
  actions, modify ``self.observation_space`` to contain all states that the
  agent has access to, and modify ``self.config`` in order to pass the config
  needed by the sim.
- Modify ``SimWrapper.reward`` in ``src/sim.py`` by adapting it to the
  problem you want to solve.
- Modify ``SimWrapper.terminal`` in ``src/sim.py`` to return ``True`` when
  conditions are met to terminate the episode
- Modify ``SimWrapper.truncate`` in ``src/sim.py``

To understand the meaning of ``reward``, ``terminal``, and ``truncate`` we
suggest checking Gymnasium's
[documentation](https://gymnasium.farama.org/api/env/#gymnasium.Env.step).
