# Gymnasium Integration

> This example shows an integration between a custom python simulation and AML (Azure
> Machine Learning). We use Gymnasium as a wrapper for simpler integration with RLLIB
> (using IMPALAs algorithm for RL training). This ReadMe describes how to replicate
> our training with any Python simmulation on your own AML workspace.

## Example Overview

This example has two different sections:

1. A PY file to setup your AML workspace (workspace accesss, compute instances, environment, and experiment). Which then is ready to run RL-based AML jobs.
2. A set of files and scripts that define your job properties, your integration with AML, and your simulation.


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

To run the script, you will need to setup a conda environment. Before setting it up, please install
[miniconda](https://docs.conda.io/en/latest/miniconda.html). Once installed, run the following command in your CLI
to get your conda environment set up:

```
conda env create --file core_aml_rllib.yml
```

Once your conda environment is ready, you can run the following command to activate it:

```
conda activate aml_rllib_test
```

Once your conda environment has been activated, run the following command to setup your AML workspace, initialize
your experiment, and run your first RL-based AML job.

```
python core_aml_pipeline_class.py
```

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

## Final Thoughts

As mentioned at the very beginning, the above sections describe the relevant sections so that you can integrate your own
simulation and play around with different setups across runs. The remaining files that are not mentioned on this overview
should not be changed, since they depend on one another to run the workflow.

Please, let us know if any sections are unclear or unnecessary. We are happy to make changes to maximize both usability
and readibility.
