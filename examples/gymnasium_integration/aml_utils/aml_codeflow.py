
import os
from azureml.core import Environment
from azureml.core.compute import AmlCompute, ComputeTarget

### AUXILIARY STAND-ALONE METHODS ###

def aml_environment_create(self, ws, ray_environment_name, dockerfile_path):
    '''Create environment from Dockerfile.'''

    print(f"Updating environment with name {ray_environment_name}...")
    ray_environment_dockerfile_path = os.path.join(os.getcwd(), dockerfile_path)

    # Build GPU image
    ray_gpu_env = Environment. \
        from_dockerfile(name=ray_environment_name, dockerfile=ray_environment_dockerfile_path). \
        register(workspace=ws)
    ray_gpu_build_details = ray_gpu_env.build(workspace=ws)

    ray_gpu_build_details.wait_for_completion(show_output=True)

    return ray_gpu_env


def aml_compute_create(self,
                        ws,
                        compute_name,
                        compute_min_nodes,
                        compute_max_nodes,
                        vm_size):
    '''
    Create compute target.
    '''

    print(f'Creating a new compute target with name: {compute_name}. Please, wait...')
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

    return compute_target