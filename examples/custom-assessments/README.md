# Custom Assessments on AML

In this folder, we show how to run custom assessment episodes with a trained RL agent on Azure ML
with a custom Gymnasium environment ("Simple Adder").

### What this sample covers

- How to evaluate the agent with custom episode configurations

### What this sample does not cover

- How to train an agent using the simulation environment on AML
- How to optimize the training algorithm for best performance
- How to visualize the performance of the agent
- How to deploy the agent

## Prerequisites

- You have a trained agent available (for example, you followed our tutorial `getting-started-on-aml`)
- You have the simulation environment used for training the agent available.
- (Optional) To run assessments on AML instead of locally, you will need to install the [Azure CLI (with the ML extension)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public) and an [AML environment](https://azure.github.io/plato/#aml-environment-setup) with the same Python and package versions you used for training the agent.

## Example Overview

This example shows how to run custom assessments on a trained agent using Ray and AML. The example uses the "Simple Adder" simulation environment and a trained agent from the `getting-started-on-aml` example. The simulation environment is modified to enable custom episode configurations via the _options_ parameter in [gymnasium.Env.reset()](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset). Ray is then used to run parallel assessments on different episode configurations, either locally or on AML. The assessment results are saved to a CSV file for further analysis and visualization.

## Tutorial

1. Add the simulation environment to `./scr/sim.py` and modify the `reset()` method to pass episode configurations via the _options_ input parameter.

2. Add the episode configurations to `./init_conditions.json`.

3. Download the agent checkpoints locally to the ``./checkpoints`` folder.

4. To test locally, activate the Python environment you used to train the agent, and run the following command in the `./src` folder:

```bash
python main.py --test-local
```
WARNING: Running custom assessments locally may overload your machine depending on various factors, such as the specifications of your machine, the simulation environment complexity, the assessment size and duration, the data and model size, etc. You can adjust the number of Ray workers or run the code on AML to avoid this.

5. (Optional) To run on AML, follow these steps:

- Modify the ``job.yml`` file by changing the name of the AML ``environment`` and ``compute`` to be the same as those you created in the prerequisites section. Also, please modify the paths for ``checkpoint_folder`` and ``input_json`` if they are different.

- Launch the job using the Azure CLI:
```
az ml job create -f job.yml --workspace-name $YOUR_WORKSPACE --resource-group $YOUR_RESOURCE_GROUP
```

- Check that it is running by finding the job you launched in [AML
   studio](https://ml.azure.com/). You should see that ``ray`` is writing
   logs in the *Outputs + logs* tab in the ``user_logs`` folder.

- Once the job is completed, the assessment logs can also be found in AML
   studio under the *Outputs + Logs* tab of your job
   in the ``outputs`` folder.

## Next Steps
Now that you've run assessment episodes for your trained agent, you can calculate custom metrics that are relevant to your problem domain and measure key performance indicators (KPIs) that reflect your agent’s goals and objectives. You can also visualize the data to gain insights into your agent’s behavior and identify areas for improvement.
