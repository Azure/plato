# Deploy a Trained Agent on AML

This sample shows how to deploy a trained agent using AML.

### What this sample covers

- How to serve the agent via an AML managed endpoint

### What this sample does not cover

- How to train the agent
- How to visualize the performance of the agent
- How to evaluate the agent

## Prerequisites

- You have a trained agent available (for example, you followed our tutorial
 ``getting-started-on-aml``)
- You have the `observation_space` and `action_space` available from the
  simulation environment used to train the agent
- You have the same AML environment used to train the agent available in your
  AML Workspace and Plato (git+https://github.com/Azure/plato.git@main) is
  installed there


## Tutorial

### Deployment steps

There is more than one way to deploy the agent on AML. For instance, you
could follow the official AML tutorial
[here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints)
or you can do all steps on the AML portal.
If you want to deploy using the portal:

1. Go to the *Outputs + logs* tab of the job you used to train the agent
2. Click on *Register model* and select *Model type: unspecified* and as *Job
   output* the folder containing the checkpoints (for example
   *checkpoint_000010*)
3. Add model settings as you see fit and register the model
4. Go into the *Models* section of your AML Workspace
5. Select the model registered at step 3 and click on *Deploy* and *Real-time
   endpoint*
6. Leave the default values in *Endpoint*, *Model* and *Deployment* section
   (Or apply changes as needed)
7. Make the relevant changes to ``score.py`` to adapt it to your model.
   The things to change are located after the imports and include: the
   checkpoint location, the observation space, and the action space
8. In the *Environment* tab upload the scoring script ``score.py``
9. Select the environment you used to train the agent and go to the next page
10. Select the size of the machine depending on your needs (1 CPU with RAM
    ~4GB and 1 instance should be enough for most workloads. Please note that
    the costs are incurred until the endpoint is manually deleted.)
11. In the *Traffic* tab leave everything unchanged or apply changes as
    needed
12. Finalize the deployment

### Test the deployment

Once you have deployed the agent, it will take some time before the
deployment is successful. When that happens, you can test that it replies to
your requests with the next action to take.
For instance, the following cURL command can be used:
```bash
curl --request POST --header "Authorization: Bearer $TOKEN" --header 'Content-Type: application/json' --url "$AGENT_URL" --data '{"state":{"value": 5}}'
```
You can find ``$TOKEN`` and ``$AGENT_URL`` in the *Consume* tab of the agent
*Endpoint*. The URL is the *REST endpoint* and the token is the *Primary
key*.
