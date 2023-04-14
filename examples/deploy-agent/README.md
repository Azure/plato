# Deploy a trained agent

This sample shows how to deploy a trained agent using ``ray``'s module
``serve``.

## Prerequisites

- You have a trained agent available (for example you followed our tutorial
    ``getting-started-on-aml``)
- You have the simulation environment used to train the agent available on
  your local machine

## Step by step tutorial

### Download the checkpoints locally

Download the model checkpoints to the ``checkpoints`` folder

### Modify the deployment script

You can find the deployment script at ``src/serve.py``.
Please adapt the script to your setup.
For instance, ensure the variable ``CHECKPOINT_FOLDER`` contains the correct
name to the folder containing your checkpoints. Please note that
``CHECKPOINT_FOLDER`` must be the name of the folder inside ``checkpoints``
that contains the trained agent.
In addition, ensure the simulation environment is registered with the same
name used for training the agent, and that the function ``prepare_state``
contains the logic that modifies a ``JSON`` object into the
``observation_space`` expected by the agent.

### Deploy locally

First of all, install the needed dependencies:

```bash
pip install -r requirements.txt
```

Then, you can test the agent locally by running the following command in the
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

### Package the agent

Use the included ``Dockerfile`` for packaging the agent.

#### Build the image locally

If you have Docker installed locally, you can build the agent and test that
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

To test that the agent responds correctly, use the same commands shown in the
section *Deploy locally*.
