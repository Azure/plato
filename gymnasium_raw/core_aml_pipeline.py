import azureml.core
from azureml.core import Workspace

import os

# Check core SDK version number
#print("Azure Machine Learning SDK version: ", azureml.core.VERSION)

###############################################################################################################
####################### PARAM SETUP ###########################################################################
###############################################################################################################
compute_name = 'compute-gym-gpu'
# Base image for the training environment (no files pushed).
ray_environment_name = 'cstr-gpu'
ray_environment_recreate = False
# Experiment to run (sim files pushed).
experiment_name = 'rllib-gym-multi-node'
experiment_run = True
max_experiment_time = 1800 # seconds


###############################################################################################################
####################### WORKSPACE ACCESS ######################################################################
###############################################################################################################

print("\n##### WORKSPACE ACCESS SETUP #####")

# IDENTIFY CONFIG FOLDER WITH ACCESS TO AZURE ML RESOURCES.
# If not: (1) get the workspace data, (2) create folder, (3) dump data.
if not os.path.exists(".azureml") or not os.path.exists(".azureml/azureml-config.json"):

    # GET WORKSPACE DATA.
    print("You don't have a config file. Please, provide the following information.")
    workspace_name = input("workspace_name (AML): ")
    subscription_id = input("subscription_id: ")
    resource_group = input("resource_group: ")

    # CREATE FOLDER.
    try:
        os.makedirs(".azureml")
    except:
        pass

    # DUMP CONFIG TO FILE.
    ws = Workspace(workspace_name=workspace_name,
                   subscription_id=subscription_id,
                   resource_group=resource_group)
    ws.write_config(path=".azureml", file_name="azureml-config.json")


# Attempt to load workspace from the config file.
try:
    ws = Workspace.from_config(path=".azureml/azureml-config.json")
    print(ws.name, ws.location, ws.resource_group, sep = ' | ')

except:
    
    from azureml.core.authentication import InteractiveLoginAuthentication
    
    print("\nWe were unable to load the workspace.")

    if not os.path.exists(".azureml/user-access.json"):
        # Request TENANT ID.
        print("Please, provide the following information for interactive login (it will be saved for future runs).")
        tenant_id = input("tenant_id: ")

        # Store TENANT ID in config file.
        import json
        with open('.azureml/user-access.json', 'w+') as json_file:
            d = {'tenant_id': tenant_id}
            json.dump(d, json_file)
    
    else:
        # Load TENANT ID from config file.
        import json
        with open('.azureml/user-access.json', 'r') as json_file:
            d = json.load(json_file)
            tenant_id = d['tenant_id']

    # Attempt to load workspace from the config file with interactive login.
    interactive_auth = InteractiveLoginAuthentication(tenant_id=tenant_id)
    ws = Workspace.from_config(path=".azureml/azureml-config.json", auth=interactive_auth)

print("We were able to access the following resources:", ws.name, ws.location, ws.resource_group, sep = ' | ')


###############################################################################################################
####################### EXPERIMENT SETUP ######################################################################
###############################################################################################################
from azureml.core.experiment import Experiment

print("\n##### EXPERIMENT SETUP #####")

# Experiment name
experiment_name = experiment_name #'rllib-cstr-multi-node'
exp = Experiment(workspace=ws, name=experiment_name)


###############################################################################################################
####################### COMPUTE TARGET ########################################################################
###############################################################################################################
from azureml.core.compute import AmlCompute, ComputeTarget

print("\n##### COMPUTE TARGET SETUP #####")

# Choose a name for the Ray cluster
compute_name = compute_name #'compute-gpu'
compute_min_nodes = 0
compute_max_nodes = 2

# This example uses GPU VM. For using CPU VM, set SKU to STANDARD_D2_V2
vm_size = 'STANDARD_NC6'

if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print(f'Found compute target with name {compute_name}.')
        if compute_target.provisioning_state == 'Succeeded':
             print(f'Compute target is ready to use.')
        else: 
            raise Exception(
                f'Compute target cannot be used. Found it in state: {compute_target.provisioning_state}.')
else:
    print(f'Did not find compute target with name: {compute_name}. Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size=vm_size,
        min_nodes=compute_min_nodes, 
        max_nodes=compute_max_nodes,
    )

    # Create the cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
    
    # Can poll for a minimum number of nodes and for a specific timeout. 
    # If no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
    # For a more detailed view of current AmlCompute status, use get_status()
    print(compute_target.get_status().serialize())


###############################################################################################################
####################### CREATE ENVIRONMENT ####################################################################
###############################################################################################################
import os
from azureml.core import Environment

print("\n##### ENVIRONMENT SETUP #####")

# Check if environment exists
if not ray_environment_recreate:
    try:
        ray_gpu_env = Environment.get(workspace=ws, name=ray_environment_name)
        print(f"Found environment with name {ray_environment_name}.")
    except:
        ray_environment_recreate = True
        print(f"Did not find environment with name {ray_environment_name}. Creating a new environment...")

# Create environment if it doesn't exist or whenever it is requested
if ray_environment_recreate:
    print(f"Updating environment with name {ray_environment_name}...")
    ray_environment_name = ray_environment_name #'cstr-gpu'
    ray_environment_dockerfile_path = os.path.join(os.getcwd(), 'docker', 'Dockerfile-gpu')

    # Build GPU image
    ray_gpu_env = Environment. \
        from_dockerfile(name=ray_environment_name, dockerfile=ray_environment_dockerfile_path). \
        register(workspace=ws)
    ray_gpu_build_details = ray_gpu_env.build(workspace=ws)

    ray_gpu_build_details.wait_for_completion(show_output=True)


###############################################################################################################
####################### RUN EXPERIMENT ########################################################################
###############################################################################################################
from azureml.core import RunConfiguration, ScriptRunConfig

print("\n##### RUN EXPERIMENT #####")

if experiment_run:
    print(f"Running experiment with name {experiment_name}...")
    experiment_name = experiment_name #'rllib-multi-node'

    experiment = Experiment(workspace=ws, name=experiment_name)
    ray_environment = Environment.get(workspace=ws, name=ray_environment_name)

    aml_run_config_ml = RunConfiguration(communicator='OpenMpi')
    aml_run_config_ml.target = compute_target
    aml_run_config_ml.node_count = 2
    aml_run_config_ml.environment = ray_environment

    training_algorithm = "IMPALA"
    rl_environment = "custom_gym_env"
    script_name='custom_rllib_run.py'

    command=[
        'python', script_name,
        '--run', training_algorithm,
        '--env', rl_environment,
        '--config', '\'{"num_gpus": 1, "num_workers": 11}\'',
        '--stop', '\'{"episode_reward_mean": 100, "time_total_s": ' + str(max_experiment_time) + '}\''
    ]

    config = ScriptRunConfig(source_directory='./files',
                             command=command,
                             run_config = aml_run_config_ml
                             )
    training_run = experiment.submit(config)
