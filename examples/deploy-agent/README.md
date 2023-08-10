# Deploy a Trained Agent

This sample shows how to deploy a trained agent using ``ray``'s module
``serve``.

### What this sample covers

- How to serve the agent locally
- How to package and deploy the agent with Docker

### What this sample does not cover

- How to train the agent
- How to visualize the performance of the agent
- How to evaluate the agent

## Prerequisites

- You have a trained agent available (for example, you followed our tutorial
    ``getting-started-on-aml``)
- You have the `observation_space` and `action_space` available from the simulation environment used to train the agent.
- (Optional) If you would like to package and run the agent locally, you need [Docker](https://docs.docker.com/get-docker/) installed on your local machine.

## Tutorial

Follow these steps to build and test a Docker image with your trained agent.
Please note that if you run Docker on Windows you should first switch to
Linux containers. Please find instructions
[here](https://docs.docker.com/desktop/faqs/windowsfaqs/#how-do-i-switch-between-windows-and-linux-containers)

1. Download the checkpoints locally to the ``checkpoints`` folder.

2. Modify the deployment script at ``src/serve.py``. Please adapt the script to your setup as follows:
    - Change the variable ``CHECKPOINT_FOLDER`` to the name of the folder containing your RLlib checkpoints.
    - Please note that ``CHECKPOINT_FOLDER`` must be the name of the folder inside ``checkpoints`` that contains the trained agent.
    - Modify the `observation_space` and `action_space` variables to be the same as in your simulation environment used for training the agent.
    - Change value of the ``name_env`` and make it equal to the one used for
      registering the simulation environment at training.

3. Change package versions in `requirements.txt` to match your training
   environment. Pay particular attention to the ``ray``, ``tensorflow``,
   ``torch``, ``numpy`` versions you have in the training environment and
   specify the same versions in ``requirements.txt``.

4. Package and deploy the agent locally:
    - Use the included ``Dockerfile`` for packaging the agent. If you have Docker installed locally, build the agent by running the following command in this folder:

    ```bash
    docker build -t rl-agent .
    ```

    - You can then use the image ``rl-agent`` and deploy the brain locally via docker:

    ```bash
    docker run -p 8000:8000 rl-agent
    ```

    - You can now reach the agent at ``http://localhost:8000``. For example, use the
    following snippet to get the action from the agent (the `state` is for
    the Simple Adder environment. If you use a different one, please modify
    accordingly):

    ```python
    import requests
    resp = requests.get('http://localhost:8000', json={'state':{'value': 5}})
    print(resp.json)
    ```

    - Or, try the following cURL command in your terminal:

    ```
    curl --request POST --url "http://localhost:8000/" --data '{"state":{"value": 5}}'
    ```
## Next Steps
After you have containerized your RL agent, you can:

- __Share your agent with others__. You can easily share containerized agents with others by uploading them to a container registry, such as Azure Container Registry. This makes it easy for others to use your agent, even if they don't have the same software or hardware as you do.
- __Deploy your agent to a variety of environments__. Containerized RL agents are portable, so you can deploy them to a variety of environments, including on-premises servers, cloud computing platforms, and edge devices.
- __Scale your agents to meet your needs__. Containerized RL agents are scalable, so you can easily increase the number of running instances of the agent to meet your needs. This is useful for applications that require a large number of actions to be taken.
