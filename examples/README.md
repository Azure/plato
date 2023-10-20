# Examples

In this folder you can find different samples showing you how to do RL on
AML.

We suggest going through the samples in the following order:

1. ``getting-started-on-aml``: learn how train your first RL agent on AML
   with a Python sim, or ``getting-started-anylogic-sim`` if you have an
   Anylogic sim.
2. ``hyperparameter-tuning-and-monitoring``: learn how to leverage `mlflow` and `ray.tune` to save model checkpoints and tune hyperparameters.
3. ``curriculum-learning``: learn how to gradually increase the difficulty or complexity of the task(s) that the agent has to solve.
4. ``logging-trajectories``: learn how to use callbacks to log the agent’s trajectories during training and evaluation, and access the logged data in your Azure Machine Learning workspace.
5. ``custom-assessments``: learn how to log assessment episodes from a trained agent using custom episode configurations.
6. ``deploy-agent``: learn how to deploy an agent locally or with docker.
