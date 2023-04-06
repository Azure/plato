# Glossary

## RLlib Terms

* __Action__: A decision made by the agent to change the state of the environment.
* __Algorithm__: A set of instructions that an agent follows to learn how to behave in an environment by performing actions and receiving feedback (reward/penalty) based on those actions.
* __Agent__: The learner and decision maker that interacts with an environment and receives a reward signal based on its actions.
* __Batch__: A collection of steps that are used to update a policy.
* __Environment (Simulation)__: A simulation of a real-world scenario that an agent interacts with.
* __Episode__: A sequence of actions taken by an agent from an initial state to either a “success” or “failure” causing the environment to reach its “terminal” state. At each step, the agent receives an observation (i.e., the observable states of the environment), takes an action, and receives a reward.
* __Iteration__: A single training call for an RLlib Trainer (calling Trainer.train() once). An iteration may contain one or more episodes (collecting data for the train batch or for a replay buffer), and one or more SGD update steps, depending on the particular Trainer being used. NOTE: In RLlib, _iteration_ should not be confused with the term _step_.
* __Gymnasium__: An open source Python library for developing and comparing reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of example environments compliant with the API.
* __Observation__: The part of a state that the agent can observe.
* __Policy__: A function mapping of the environment’s observational states to an action to take, usually written π (s(t)) -> a(t).
* __Ray__: A distributed computing framework that makes it easy to scale your applications and to leverage state-of-the-art machine learning libraries such as RLlib.
* __Reward__: A scalar value that indicates how well the agent is doing at a given step. For each good action, the agent gets positive feedback/reward, and for each bad action, the agent gets negative feedback/reward or penalty.
* __RLlib__: An open source Python library that provides scalable and easy-to-use reinforcement learning solutions.
* __Rollout worker (Ray Actor)__:  A process that interacts with an environment and collects trajectories for training.
* __State__: A set of information that an agent has about the environment at a given time. States should have the Markov property, which means that knowing the state means you know everything that could determine the response of the environment to a specific action.
* __Step__: A single interaction between an agent and an environment, which consists of an observation (i.e., the state of the environment), an action, a reward, and a new observation.

## Azure Machine Learning Terms

* __Compute cluster__: A managed-compute infrastructure that allows you to easily create a single or multi-node resource for training or inference.
* __Environment (AML)__:  A collection of software dependencies and configurations that are needed to run your reinforcement learning code on AML.
* __Workspace__: A top-level resource for your machine learning activities, providing a centralized place to view and manage the artifacts you create when you use AML. A workspace contains your experiments, models, datastores, compute targets, environments, and other resources.
