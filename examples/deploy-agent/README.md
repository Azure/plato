# Deploy a trained agent

This sample shows how to deploy a trained agent using ``ray``'s module
``serve``.

## Prerequisites

- You have a trained agent available (for example, you followed our tutorial
    ``getting-started-on-aml``)
- You have the `observation_space` and `action_space` available from the simulation environment used to train the agent.
- (Optional) If you would like to package and run the agent locally, you need [Docker](https://docs.docker.com/get-docker/) installed on your local machine.

## Tutorial

### Step 1: Download the checkpoints locally

Download the model checkpoints to the ``checkpoints`` folder.

### Step 2: Modify the deployment script

You can find the deployment script at ``src/serve.py``.
Please adapt the script to your setup as follows:

- Change the variable ``CHECKPOINT_FOLDER`` to the name of the folder containing your RLlib checkpoints.
  - Please note that
``CHECKPOINT_FOLDER`` must be the name of the folder inside ``checkpoints``
that contains the trained agent.
- Modify the `observation_space` and `action_space` variables to be the same as in your simulation environment used for training the agent.

### Step 3: Change package versions to align with your training environment

The `requirements.txt` file lists the Python packages needed to deploy your agent. You need to edit the versions in this file to match your agent training environment. For example, if your training environment has ray==2.3.0 and this file has ray==2.4.0, you need to change it to ray==2.3.0.

### Step 4: Test locally

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then, test the agent locally by running the following command in the
``src`` folder:

```bash
serve run serve:agent
```

You can reach the agent at ``http://localhost:8000``. For example, use the
following snippet to get the action from the agent:

```python
import requests
resp = requests.get('http://localhost:8000', json={'state':{'value': 5}})
print(resp.json)
```

Or, try the following cURL command in your terminal:

```
curl --request POST --url "http://localhost:8000/" --data '{"state":{"value": 5}}'
```

### (Optional) Step 5: Package and deploy the agent locally

Use the included ``Dockerfile`` for packaging the agent. If you have Docker installed locally, you can build the agent and test that
it works.
To build the agent run the following command in this folder

```bash
docker build -t rl-agent .
```

You can then use the image ``rl-agent`` locally and deploy the brain locally
via docker:

```bash
docker run -p 8000:8000 rl-agent
```

To ensure that the agent responds correctly, use the same requests or cURL commands shown in Step 4 (*Test locally*).

## Next Steps
After you have containerized your RL agent, you can:

- __Share your agent with others__. You can easily share containerized agents with others by uploading them to a container registry, such as Azure Container Registry. This makes it easy for others to use your agent, even if they don't have the same software or hardware as you do.
- __Deploy your agent to a variety of environments__. Containerized RL agents are portable, so you can deploy them to a variety of environments, including on-premises servers, cloud computing platforms, and edge devices.
- __Scale your agents to meet your needs__. Containerized RL agents are scalable, so you can easily increase the number of running instances of the agent to meet your needs. This is useful for applications that require a large number of actions to be taken.
