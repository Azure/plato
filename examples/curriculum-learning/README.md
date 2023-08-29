# Curriculum Learning

In this example, we show how to use curriculum learning to train an RL agent on Azure ML with a custom Gymnasium environment (“Simple Adder”). Curriculum learning is a technique that orders the training data according to some measure of difficulty, and gradually exposes the model to harder episodes as it learns.

### What this sample covers
- How to modify a custom Gymnasium simulation environment to use curriculum learning with RLlib
- How to implement curriculum learning on your local machine and on Azure ML

### What this sample does not cover
- How to create an optimized curriculum for best performance
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
- Create an AML environment using the conda file provided (``conda.yml``) by running the following command:
```bash
az ml environment create --name curriculum-learning-env --conda-file conda.yml --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 --resource-group $YOUR_RESOURCE_GROUP --workspace-name $YOUR_WORKSPACE
```

## Example Overview
The simulation environment in this sample is "Simple Adder" (`scr/sim_curriculum_capable.py`), where the agent has to choose a number between -10 and 10 to add to a state value. The goal is to get the state value equal to 50 in 10 time steps or less. The difficulty level depends on how far the state value is from 50 at the start, as larger distances make the task harder. The curriculum learning strategy is to gradually expand the range of possible state values around 50, increasing the level of difficulty. The agent starts training on the easiest range (smallest distances) and progresses to the next range, which includes the previous one, after reaching a certain average reward threshold. The agent repeats this process until it reaches the hardest range (largest distances) or the maximum number of iterations.

## Run Locally
As a preliminary step, you should check that the simulation works on your local machine to save precious development time. The main.py script in the src folder allows you to test locally with the following command:

```
python main.py --test-local
```

## Tutorial: Run on AML
After you checked that the simulation works properly, follow these steps to train an RL agent on AML using curriculum learning:

1. Open the `conda.yaml` file and fill in the values for the AML `workspace_name`, `resource_group`, `subscription_id`, and `compute_target_name` with the ones you created in the prerequisites section. These values are used to connect to your AML workspace and compute cluster.

2. Open the `src/main.py` file and do the following:
    - Modify the curriculum learning function (`curriculum_fn()`) to return a new task (or difficulty level) for your environment based on some criteria. For example, you can set a threshold on the average episode reward as a measure of difficulty.
    - Adjust the `train()` function parameters, such as `trainable`, `rollouts`, and `stopping_criteria`, according to your desired strategy.
    - Modify the `CurriculumCallback()` class to log the current task of the environment to TensorBoard. This class can also implement other methods to customize the training behavior, such as `on_train_result`, `on_episode_end`, etc. For example, you can log other metrics, save checkpoints, or update hyperparameters based on the curriculum learning progress.

3. Open your custom simulation environment file (`src/sim_curriculum_capable.py`) and make sure it inherits from the `TaskSettableEnv` class from Ray RLLib and implements its methods, such as `get_task()` and `set_task()`. These methods are used by the curriculum learning function and callback to get and set the current task of the environment. The task should be a dictionary that contains any information that defines the difficulty of the environment, such as the number of obstacles, the size of the grid, the speed of the agent, etc.

4. Launch the job using the Azure CLI:
```
az ml job create -f job.yml --workspace-name $YOUR_WORKSPACE --resource-group $YOUR_RESOURCE_GROUP
```

5. Check that it is running by finding the job you launched in AML studio. You should see that ray is writing logs in the Outputs + logs tab in the user_logs folder.

6. Monitor the curriculum learning progress and results on the AML studio or using TensorBoard on your local machine. You should see a custom metric called “task” that shows the current difficulty level of the environment for each episode.

7. Once the job is completed, download the model checkpoints from AML studio under the Outputs + Logs tab of your job in the outputs folder.

## Next Steps
Now that you've successfully trained an agent using curriculum learning, you can experiment with different ways to design and evaluate curriculum learning. For example, you can use reward, entropy, uncertainty, or diversity as measures of difficulty.

To learn more about how to use your trained agent, check out our [deploy-agent sample](https://github.com/Azure/plato/tree/main/examples/deploy-agent), which shows you how to deploy a trained agent and interact with it.
