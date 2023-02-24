import os
import azureml.core
from azureml.core import Workspace, Environment, RunConfiguration, ScriptRunConfig
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.experiment import Experiment

from aml_utils.aml_codeflow import aml_environment_create, aml_compute_create

############ PARAM SETUP ############
compute_name = 'gym-gpu-compute'
# Base image for the training environment (no files pushed).
ray_environment_name = 'gym-gpu-image'
ray_environment_recreate = False
# Experiment to run (sim files pushed).
experiment_name = 'gym-gpu-rllib-multi-node'
experiment_run = True
max_experiment_time = 1800 # seconds


class AML_Pipeline:

    def __init__(self,
                 compute_name,
                 ray_environment_name,
                 ray_environment_recreate,
                 experiment_name,
                 experiment_run,
                 max_experiment_time):


        self.compute_name = compute_name
        self.ray_environment_name = ray_environment_name
        self.ray_environment_recreate = ray_environment_recreate
        self.experiment_name = experiment_name
        self.experiment_run = experiment_run
        self.max_experiment_time = max_experiment_time

        # WORKSPACE ACCESS SETUP.
        self._workspace_setup()

        # COMPUTE TARGET SETUP.
        self._compute_target_setup()

        # ENVIRONMENT SETUP.
        self._environment_setup()

        # RUN EXPERIMENT.
        if self.experiment_run:
            self._experiment_run()
    
    def _workspace_setup(self,
                         config_folder=".azureml",
                         aml_config_file="azureml-config.json",
                         user_access_config_file="user-access.json"):

        print("\n##### WORKSPACE ACCESS SETUP #####")
        aml_config_filepath = os.path.join(config_folder, aml_config_file)
        user_access_config_filepath = os.path.join(config_folder, user_access_config_file)

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
            ws.write_config(path=config_folder, file_name=aml_config_file)


        # Attempt to load workspace from the config file.
        try:
            ws = Workspace.from_config(path=aml_config_filepath)
            print(ws.name, ws.location, ws.resource_group, sep = ' | ')

        except:
            
            from azureml.core.authentication import InteractiveLoginAuthentication
            
            print("\nWe were unable to load the workspace.")

            if not os.path.exists(user_access_config_filepath):
                # Request TENANT ID.
                print("Please, provide the following information for interactive login (it will be saved for future runs).")
                tenant_id = input("tenant_id: ")

                # Store TENANT ID in config file.
                import json
                with open(user_access_config_filepath, 'w+') as json_file:
                    d = {'tenant_id': tenant_id}
                    json.dump(d, json_file)
            
            else:
                # Load TENANT ID from config file.
                import json
                with open(user_access_config_filepath, 'r') as json_file:
                    d = json.load(json_file)
                    tenant_id = d['tenant_id']

            # Attempt to load workspace from the config file with interactive login.
            interactive_auth = InteractiveLoginAuthentication(tenant_id=tenant_id)
            ws = Workspace.from_config(path=aml_config_filepath, auth=interactive_auth)

        print("We were able to access the following resources:", ws.name, ws.location, ws.resource_group, sep = ' | ')

        self.ws = ws
        return self.ws
    

    def _compute_target_setup(self):

        print("\n##### COMPUTE TARGET SETUP #####")

        # HARD CODED VM SPECS.
        # This example uses GPU VM. For using CPU VM, set SKU to STANDARD_D2_V2
        compute_min_nodes = 0
        compute_max_nodes = 2
        vm_size = 'STANDARD_NC6'

        if self.compute_name in self.ws.compute_targets:
            compute_target = self.ws.compute_targets[self.compute_name]
            if compute_target and type(compute_target) is AmlCompute:
                print(f'Found compute target with name {self.compute_name}.')
                if compute_target.provisioning_state == 'Succeeded':
                    print(f'Compute target is ready to use.')
                else: 
                    raise Exception(
                        f'Compute target cannot be used. Found it in state: {compute_target.provisioning_state}.')
        else:
            print(f'Did not find compute target with name.')
            compute_target = aml_compute_create(ws=self.ws,
                                                compute_name=self.compute_name,
                                                compute_min_nodes=compute_min_nodes,
                                                compute_max_nodes=compute_max_nodes,
                                                vm_size=vm_size)
        
        self.compute_target = compute_target
        return self.compute_target


    def _environment_setup(self, dockerfile_path="docker/Dockerfile-gpu"):

        print("\n##### ENVIRONMENT SETUP #####")

        # Check if environment exists.
        if not self.ray_environment_recreate:
            try:
                ray_gpu_env = Environment.get(workspace=self.ws, name=self.ray_environment_name)
                print(f"Found environment with name {self.ray_environment_name}.")
            except:
                self.ray_environment_recreate = True
                print(f"Did not find environment with name {self.ray_environment_name}. Creating a new environment...")

        # Create environment if it doesn't exist or whenever an update is requested.
        if self.ray_environment_recreate:
            ray_gpu_env = aml_environment_create(self.ws, self.ray_environment_name, dockerfile_path)
    
        self.ray_gpu_env = ray_gpu_env
        return self.ray_gpu_env


    def _experiment_run(self):

        print("\n##### RUN EXPERIMENT #####")

        training_algorithm = "IMPALA"
        rl_environment = "custom_gym_env"
        script_name='custom_rllib_run.py'

        print(f"Running experiment with name {self.experiment_name}...")
        self.experiment_name = self.experiment_name #'rllib-multi-node'

        experiment = Experiment(workspace=self.ws, name=self.experiment_name)
        self.experiment = experiment
        ray_environment = Environment.get(workspace=self.ws, name=self.ray_environment_name)
        self.ray_environment = ray_environment

        aml_run_config_ml = RunConfiguration(communicator='OpenMpi')
        aml_run_config_ml.target = self.compute_target
        aml_run_config_ml.node_count = 2
        aml_run_config_ml.environment = ray_environment

        command=[
            'python', script_name,
            '--run', training_algorithm,
            '--env', rl_environment,
            '--config', '\'{"num_gpus": 1, "num_workers": 11}\'',
            '--stop', '\'{"episode_reward_mean": 100, "time_total_s": ' + str(self.max_experiment_time) + '}\''
        ]

        config = ScriptRunConfig(source_directory='./files',
                                command=command,
                                run_config = aml_run_config_ml
                                )
        training_run = self.experiment.submit(config)

        #training_run.wait_for_completion(show_output=True)
    



if __name__ == "__main__":

    job_config = {
        "compute_name": compute_name,
        "ray_environment_name": ray_environment_name,
        "ray_environment_recreate": ray_environment_recreate,
        "experiment_name": experiment_name,
        "experiment_run": experiment_run,
        "max_experiment_time": max_experiment_time
    }

    # Create an AML_Pipeline object.
    aml_pipeline = AML_Pipeline(**job_config)

    # Run the pipeline.
    aml_pipeline.run()
