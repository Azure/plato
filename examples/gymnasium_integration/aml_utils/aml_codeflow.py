
import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Environment, BuildContext

### AUXILIARY STAND-ALONE METHODS ###

def aml_environment_create(ml_client: MLClient,
                           ray_environment_name,
                           dockerfile_path):
    '''Create environment from Dockerfile.'''

    print(f"Updating environment with name {ray_environment_name}...")
    ray_environment_dockerfile_path = os.path.join(os.getcwd(), dockerfile_path)

    # # TODO: UNCOMMENT SDK v2 INTEGRATION ONCE AML BUG IS RESOLVED, as follows:
    # #    - follow bug resolution here: https://github.com/Azure/azure-sdk-for-python/issues/29279 -
    # # 1. uncomment this code.
    # # 2. make any changes to workflow to match resolution. 
    # # 3. remove SDK v1 Environment creation right after this comment.
    # # 4. remove SDK v1 library references in yml (i.e.: azureml-core==1.49.0).
    # 
    # ray_gpu_env = Environment(
    #     #image="XX", #pytorch/pytorch:latest",
    #     name=ray_environment_name,
    #     build=BuildContext(dockerfile_path=ray_environment_dockerfile_path)
    # )
    # 
    # ray_gpu_env = ml_client.environments.create_or_update(ray_gpu_env)
    # 
    # # Wait for the environment to be ready. 
    # # TODO: Add loop to check if environment is ready. (see compute creation below)
    # 
    # return ray_gpu_env

    ## SDK V1 INTEGRATION
    from azureml.core import Workspace as workspace_sdk_v1
    from azureml.core import Environment as environment_sdk_v1

    # Create workspace object
    workspace_name = ml_client.workspace_name
    rg_name = ml_client.resource_group_name
    subs_id_name = ml_client.subscription_id
    ws = workspace_sdk_v1(subs_id_name, rg_name, workspace_name)

    # Build GPU image
    ray_gpu_env = environment_sdk_v1. \
        from_dockerfile(name=ray_environment_name, dockerfile=ray_environment_dockerfile_path). \
        register(workspace=ws)
    ray_gpu_build_details = ray_gpu_env.build(workspace=ws)

    ray_gpu_build_details.wait_for_completion(show_output=True)

    # ENVIRONMENT OBJECT IN SDK V2
    ray_gpu_env = ml_client.environments.get(ray_environment_name, label="latest")

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