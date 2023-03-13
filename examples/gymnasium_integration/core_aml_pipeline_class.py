import os
from azure.ai.ml import MLClient, command, MpiDistribution
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute, Workspace

from aml_utils.aml_codeflow import aml_environment_create, aml_compute_create

############ PARAM SETUP ############
# Compute target to run the experiment on (note: no files pushed to compute).
COMPUTE_NAME = 'gym-gpu-compute'
# Base image for the training environment (note: no files pushed to environment).
RAY_ENVIRONMENT_NAME = 'gym-gpu-image'
RAY_ENVIRONMENT_RECREATE = False
# Experiment to run (note: includes sim files to be used for training).
EXPERIMENT_NAME = 'gym-gpu-rllib-multi-node'
MAX_EXPERIMENT_TIME = 1800 # seconds


class AML_Pipeline:

    def __init__(self,
                 compute_name,
                 ray_environment_name,
                 ray_environment_recreate,
                 experiment_name,
                 max_experiment_time):

        self.compute_name = compute_name
        self.ray_environment_name = ray_environment_name
        self.ray_environment_recreate = ray_environment_recreate
        self.experiment_name = experiment_name
        self.max_experiment_time = max_experiment_time

        # WORKSPACE ACCESS SETUP.
        self._workspace_setup()

        # COMPUTE TARGET SETUP.
        self._compute_target_setup()

        # ENVIRONMENT SETUP.
        self._environment_setup()


    def experiment_run(self,
                       experiment_name = None,
                       max_experiment_time = None,
                       training_algorithm = "IMPALA",
                       rl_environment = "custom_gym_env",
                       script_name='custom_rllib_run.py'):

        print("\n##### RUN EXPERIMENT #####")

        if experiment_name is not None:
            self.experiment_name = experiment_name
        if max_experiment_time is not None:
            self.max_experiment_time = max_experiment_time

        print(f"Running experiment with name {self.experiment_name}, with a max time limit of {self.max_experiment_time} secs...")
        
        exp_command=[
            'python', script_name,
            '--run', training_algorithm,
            '--env', rl_environment,
            '--config', '\'{"num_gpus": 1, "num_workers": 11}\'',
            '--stop', '\'{"episode_reward_mean": 100, "time_total_s": ' + str(self.max_experiment_time) + '}\''
        ]
        exp_command_str = ' '.join(exp_command)

        job_config_command = command(
            experiment_name=self.experiment_name,
            code='./files',
            command=exp_command_str,
            environment=self.ray_gpu_env,
            compute=self.compute_name,
            distribution=MpiDistribution(process_count_per_instance=2),
            #instance_count=2,
        )

        returned_job = self.ml_client.jobs.create_or_update(job_config_command)
        return returned_job


    ### PRIVATE SETUP METHODS ###
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
            import json
            with open(aml_config_filepath, 'w+') as f:
                aml_config = {
                    "Id": None,
                    "Scope": "/subscriptions/" + subscription_id + \
                            "/resourceGroups/" + resource_group + \
                            "/providers/Microsoft.MachineLearningServices" + \
                            "/workspaces/" + workspace_name,
                }
                print("aml_config:", aml_config)
                json.dump(aml_config, f)


        # GET CREDENTIAL
        try:
            credential = DefaultAzureCredential()
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            print("\nWe were unable to load your credentials. Using Browser Interactive Credentials instead.")
            # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
            credential = InteractiveBrowserCredential()
        
        # Get a handle to the workspace
        ml_client = MLClient.from_config(credential=credential, path=aml_config_filepath)
        #ml_client = aux_ml_client.from_config(path=aml_config_filepath)

        print("We were able to access the following resources:", ml_client.workspace_name, ml_client.resource_group_name, sep = ' | ')

        self.ml_client = ml_client
        return self.ml_client
    

    def _compute_target_setup(self):

        print("\n##### COMPUTE TARGET SETUP #####")

        # HARD CODED VM SPECS.
        # This example uses GPU VM. For using CPU VM, set SKU to STANDARD_D2_V2
        compute_min_nodes = 0
        compute_max_nodes = 2
        vm_size = 'STANDARD_NC6'

        
        # CHECK IF COMPUTE TARGET EXISTS.
        try:
            compute_target = self.ml_client.compute.get(self.compute_name)
        except:
            print(f'Did not find compute target with name: {self.compute_name}.')
            compute_target = None
        
        # CHECK IF COMPUTE TARGET IS READY.
        if compute_target:
            if compute_target and type(compute_target) is AmlCompute:
                print(f'Found compute target with name {self.compute_name}.')
                if compute_target.provisioning_state == 'Succeeded':
                    print(f'Compute target is ready to use.')
                else: 
                    raise Exception(
                        f'Compute target cannot be used. Found it in state: {compute_target.provisioning_state}.')
            else:
                raise Exception(
                    f'Found compute target with name: {self.compute_name}, but it is not of type AmlCompute.')
        
        # CREATE COMPUTE TARGET IF IT DOES NOT EXIST.
        else:
            compute_target = aml_compute_create(ml_client=self.ml_client,
                                                compute_name=self.compute_name,
                                                compute_min_nodes=compute_min_nodes,
                                                compute_max_nodes=compute_max_nodes,
                                                vm_size=vm_size)
        
        self.compute_target = compute_target
        return self.compute_target


    def _environment_setup(self, dockerfile_path="docker\\Dockerfile-gpu"):

        print("\n##### ENVIRONMENT SETUP #####")

        # Check if environment exists.
        if not self.ray_environment_recreate:
            try:
                #ray_gpu_env = self.ml_client.environments.get(self.ray_environment_name, version="1")
                ray_gpu_env = self.ml_client.environments.get(self.ray_environment_name, label="latest")
                print(f"Found environment with name {self.ray_environment_name}.")
            except:
                self.ray_environment_recreate = True
                print(f"Did not find environment with name {self.ray_environment_name}. Creating a new environment...")

        # Create environment if it doesn't exist or whenever an update is requested.
        if self.ray_environment_recreate:
            ray_gpu_env = aml_environment_create(self.ml_client, self.ray_environment_name, dockerfile_path)
    
        self.ray_gpu_env = ray_gpu_env
        return self.ray_gpu_env
    



if __name__ == "__main__":

    job_config = {
        "compute_name": COMPUTE_NAME,
        "ray_environment_name": RAY_ENVIRONMENT_NAME,
        "ray_environment_recreate": RAY_ENVIRONMENT_RECREATE,
        "experiment_name": EXPERIMENT_NAME,
        "max_experiment_time": MAX_EXPERIMENT_TIME
    }

    # Create an AML_Pipeline object.
    aml_pipeline = AML_Pipeline(**job_config)

    # Run the pipeline.
    aml_pipeline.experiment_run()
