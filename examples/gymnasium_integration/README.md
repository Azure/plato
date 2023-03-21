# Gymnasium Integration

> This example shows an integration between a custom Python simulation and AML (Azure
> Machine Learning). We use Gymnasium as a wrapper for simpler integration with RLLIB
> (using IMPALAs algorithm for RL training). This ReadMe describes how to replicate
> our training with any Python simulation on your own AML workspace.


## Objective of this Example

This example will enable you to train your first AML-trained agent. This example sets up:
- Compute Clusters: Clusters are the hardware needed to run your job (gpu/cpu, # of cores, memory...)
- Environment: Environments are defined through Docker images, and identify the software of your job (linux/win, Python version, libraries...)
- Experiment: Experiments are used to collect similar jobs for easier agent comparison throughout the training session.
- Job: Each job is an individual run for each experiment.

Prior to getting started, please go through the `Prerequisites and Setup` section below.

There are two workflows described in this README:
1. The first section explains how to run the current embedded simulation as given.
   - Go to the  `AML Setup & Experiment Run` section below.
   - Following this codeflow you will understand how to set up your AML workspace, and get your first experiment job run.
   - We will be using the CSTR (Continuous-Stirred Tank Reactor) Python sim for integration.
   - Note, the prerriquisites to get started are included as part of the workflow below.
2. The second section helps you embed your own simulation for training.
   - Go to the `Job Setup & Sim Integration` section below.
   - Following this codeflow you will be able to embed your own Python simulation with Gymnasium/Rllib.
   - Then, go to the first section to get an experiment job running with your embedded simulation.


## Prerequisites and Setup

To run the script, you will need to setup a conda environment. Before setting it up, please install
[miniconda](https://docs.conda.io/en/latest/miniconda.html). Once installed, you will be ready to set up your
conda environment. First, open the CLI window. Then, traverse to the gymnasium_integration folder. Lastly,
run the "conda env create" command:

```
cd examples
cd gymnasium_integration
conda env create --file core_aml_rllib.yml
```

*We designed this workflow to run locally in your computer. With some minor changes we expect you will be able*
*to run this one directly on AML 'notebooks' so you decide to do so.*


## AML Setup & Experiment Run

The file `core_aml_pipeline_class.py` is the one you want to run to setup your AML workspace and run experiments.

At the very top of the file, we have defined the params needed for workspace setup.
Modify them as needed:

- Param related to compute to be used to run the job:
   - COMPUTE_NAME: desired compute name. Rename after making changes to compute spec in `_compute_target_setup` method.
- Params related to the environment image to load into the compute:
   - RAY_ENVIRONMENT_NAME: environment name.
   - RAY_ENVIRONMENT_RECREATE: force recreation. Use this when making changes to environment Dockerfile at `docker/Dockerfile-gpu`.
- Params related to the job to be run (aggregated as experiments): 
   - EXPERIMENT_NAME: experiment name to group all training jobs.
   - MAX_EXPERIMENT_TIME: max time to run the job for before halting execution.

If not already, traverse to the `gymnasium_integration` folder:

```
cd examples
cd gymnasium_integration
```

With your conda environment ready (see `Prerequisites and Setup` above), you can run the following command to activate it:

```
conda activate aml_rllib_test
```

Once your conda environment has been activated, run the following command to setup your AML workspace, initialize
your experiment, and run your first RL-based AML job.

```
python core_aml_pipeline_class.py
```

Once your experiment job is submitted to the cloud, you can track the progress directly in Azure.

- First, go to (Azure Portal)[https://ms.portal.azure.com/#home].
- Traverse to your Azure Machine Learning (AML) workspace.
- Click over the "Launch Studio" platform, on the `Overview` tab.
- In the Studio, on the left pane, click over `Jobs`.
- In the `Jobs` tab, select the experiment that you run (by default "gym-gpu-rllib-multi-node").
- You should see the job "Running" for at least half an hour.


### AML Setup: Next Steps

You can visualize training metrics, by following these steps:

1. With your experiment open, select "Add chart" on the top menu.
2. Open the dropdowen menu for X-axis, and select "Step".
3. Open the dropdowen menu for Y-axis, and select "episode_reward_mean".
4. Repeat 1-3, this time selecting "episode_len_mean" for Y-axis.
5. Repeat 1-3, this time selecting "episodes_total" for Y-axis.

If your training failed, you can look at logs here:

1. With your experiment open, select the job that "Failed".
2. Traverse to "Outputs + logs" on the top pane.
3. Open the `user_logs` folder, and open `std_log_process_0.txt` file.
4. If these logs aren't sufficient to determine the problem, then continue.
5. In the same  `user_logs` folder, open the `mpi_log.txt` file.
6. If these logs aren't sufficient to determine the problem, then continue.
7. Open the `logs` folder, `IMPALA` subfolder, and `IMAPALA_(your_env_name)` subfolder.
8. Then, open the `error.txt` file.

Note, this notebook does NOT yet offer inference capabilities. If you want to inference your
saved checkpoints, you will have to follow up AML documentation to transform your checkpoints
into a Model. More documentation on that here:
(Creating a Model with AML)[https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models?tabs=cli%2Cuse-local].


## Job Setup & Sim Integration

The codeflow described above runs the code as is. Yet, you certainly want to go beyond the current layout. No matter
if you want to integrate your own RL simulation or if you want to tweak the reward, you seek experimenting in between
runs. This section describes how the code is structured, so you can ue this integration for your own purposes.

All files submitted to the job can be found in the `files` folder. This will be the set of files you want to change
whenever you want to make a change to your training session. Once your new setup is ready, you can run the job again as
usual using the "AML Setup & Experiment Run" workflow described above.

Most of the codeflow is automated, and thus, we will not explain the purpose of each line of code. Instead, let us
provide you with a summary of the key sections that are relevant for training with your own sim, and param tuning.

### Sim Integration: (1) Wrapping your simulation

The first thing you want to consider for sim integration is setting up the interactions between the training codeflow
and your **Python** simulation.

1. Traverse to the simulation folder (`files/sim`) and remove all files except `simulator_model.py`.
   - *Note, we are removing all files related to the already integrated example. You will not be needing these after embedding your own simulation.*
2. In the same sim folder, drop all your simulation files.
3. Then, you must setup `simulator_model.py` to wrap your simulation into the RL training framework as follows:
   - `__init__` method should initialize your simulation object.
     -  *Please, remove reference to 'CSTRSimulation' from this method AND the header.*
   - `reset` method should call your simulation method to restart your simulation.
     - *Please, remove reference to `self.sim.reset()` if your sim method's name is not RESET.*
     - Additionally, you will have to modify the `self.sim.get_states()` method to point to whichever method retrieves all your sim states (for RL training or logging). Sim states returned by your method should be in the form of a dictionary (state_name: state_value). 
   - `step` method should call your simulation method to run a step of the simulation given the provided action by RL agent.
     - *Please, remove reference to `self.sim.step()` if your sim method's name is not STEP.*
     - Note, your simulation step function must take a dictionary with the list of actions (action_name: action_value) for integration with the section below.
     - Aditionally, as done above for RESET method, you will also need to also update the reference to `self.sim.get_states()`.
   - `termination` method should call your sim method (or attribute) in charge of raising a flag (True) whenever the sim must be halted and restarted.
   - `truncation` method should call your sim method (or attribute) in charge of raising a flag (True) whenever the sim has reached the max amount of alloted time per episode.
   - `sim_name` property is used to define the name of your simulation. Note, it is currently unused.

Following these steps, you have correctly wrapped your simulation to be compatible with our gymnasium wrapper. Yet,
this is not the end of the journey. Please, keep reading below.

### Sim Integration: (2) Set up RL & gymnasium properties

As part of the integration with gymnasium, you also need to define the specs of your simulation. We decided to split
the integration part with the RL-specific parts, to simplify the workflow. The RL-related training configuration
can be accessed at `files/training_setup`.

The first thing you want to modify now are the RL properties for the training session. You can modify them at
`files/training_setup/rl_sim_spec` as follows:

1. Set up your state and action spaces:
   - `obs_state_len` property (int): number of states that will be provided to the RL agent during training and inference (to get actions).
   - `sim_states_to_log` property (list[str]): list of state names from the variables you want to log throughout your training session.
     - *These state names should exist in the dictionary returned by STEP and RESET methods in `simulator_model.py`.
   - `action_len` property (int): number of actions that will be recommended by the RL agent.
   - ANY OTHER PROPERTIES: under "RL PROPERTIES" feel free to add, or remove, any desired properties relevant to your use case.
2. Set up your reward in `compute_reward_term_and_trun` method:
   - Based on your simulation termination and truncation properties, you want to define a shaped reward that guides your brain to better performing policies.
   - *Note, positive should always mean better performance.*
3. Set up your normalized observable states in `sim_state_to_gym` method:
   - Filter your simulation dictionary of states, and normalize them to the range [-1, 1].
   - *Note, this should include state values up to 99 percentile at the very least. Your states will be clipped when outside your defined normalization ranges.*
4. Set up your transformation from state to action in `gym_action_to_sim` method:
   - Note, your simulation must take a dictionary as input at each step: (action_name: action_value).

After setting up the main RL properties, you are ready to define your training scenario at
`files/training_setup/rl_lesson.yml`. This yaml file should contain the config parameters you want to set up during
your RL agent training session. Note, your RESET method should be the one applying the received dictionary from
`simulator_model.py` to the simulation. Otherwise, these parameters will not take affect.

- The first level should contain the name of your reset config parameters (such as "noise_percentage" and "Cref_signal" in our case).
- For each reset config parameter, you can choose to define the scenario with either:
  - `min`/`max` values: this will be used to collect a float number uniformly.
  - `values` list: this array should contain all the values among each the agent should choose randomly during each episode start.

### Sim Integration: (3) Set up Python libraries

There are two files that require your attention when updating this workflow with your own custom simulation. The first
step is identifying the Python libraries your simulation relies on. Then, go and make changes in the following files:

1. Update the docker file at `docker/Dockerfile-gpu`, to include your desired libraries.
   - Feel free to modify the libraries under `# REQUIREMENT: Install CSTR-specific packages` to the ones applicable to your simulation.
2. Add the libraries to `core_aml_rllib.yml`.
   - You will have to update your conda environment: `conda env update -n aml_rllib_test --file core_aml_rllib.yml`.

### Sim Integration: (4) Test sim integration locally

Prior to submitting your new experiment/job to Azure, test your local integration running the following command(s):

```
cd files
python custom_gym_wrapper.py
```

*Note, you will have to first update your conda environment with the necessary Python libraries, as described in the previous step.*

## Final Thoughts

As mentioned at the very beginning, the above sections describe the relevant sections so that you can integrate your own
simulation and play around with different setups across runs. The remaining files that are not mentioned on this overview
should not be changed, since they depend on one another to run the workflow.

Please, let us know if any sections are unclear or unnecessary. We are happy to make changes to maximize both usability
and readibility.
