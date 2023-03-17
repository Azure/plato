
import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Environment, BuildContext

### AUXILIARY STAND-ALONE METHODS ###

def aml_environment_create(ml_client: MLClient,
                           ray_environment_name,
                           dockerfolder,
                           dockerfile):
    '''Create environment from Dockerfile.'''

    print(f"Updating environment with name {ray_environment_name}...")
    ray_environment_dockerfolder_path = os.path.join(os.getcwd(), dockerfolder)

    ray_gpu_env = Environment(
        #image="XX", #pytorch/pytorch:latest",
        name=ray_environment_name,
        build=BuildContext(dockerfile_path=dockerfile,
                           path=ray_environment_dockerfolder_path)
    )
    
    ray_gpu_env = ml_client.environments.create_or_update(ray_gpu_env)
    
    # Wait for the environment to be ready. 
    # TODO: Add loop to check if environment is ready. (see compute creation below)
    
    return ray_gpu_env


def aml_compute_create(ml_client: MLClient,
                       compute_name: str,
                       compute_min_nodes: int,
                       compute_max_nodes: int,
                       vm_size: str):
    '''
    Create compute target.
    '''

    print(f'Creating a new compute target with name: {compute_name}. Please, wait...')
    compute = AmlCompute(
        # Name assigned to the compute cluster
        name=compute_name,
        # Azure ML Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size=vm_size,
        # Minimum running nodes when there is no job running
        min_instances=compute_min_nodes,
        # Nodes in cluster
        max_instances=compute_max_nodes,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )

    # Create the cluster
    compute_target = ml_client.compute.begin_create_or_update(compute)

    # Wait for the cluster to be ready. 
    status = compute_target.status()
    while status == "InProgress":
        print(f"Current status is: {status}. Waiting for cluster to be ready...")
        compute_target.wait(15)
        status = compute_target.status()
    
    print(f"Current status is: {status}. Cluster is ready.")

    return compute_target