# Hyperparameter Tuning with Ray Tune

## Overview


Reinforcement learning algorithms are notoriously tricky to optimize and train efficiently. Hyperparameter tuning with an efficient scheduler can often help you find a better performing agent in less time than a manual search.

The example  in `examples/hyperparameter-tuning-and-monitoring` demonstrates an example of how to use [Population Based Training (PBT)](https://docs.ray.io/en/releases-2.3.0/tune/examples/pbt_guide.html) algorithm to tune hyperparameters using Ray Tune library with MLflow on AzureML or locally. There are a few design decisions you as a user will be able to make when using this example, and this user guide will help you understand how to make those decisions.

## Why Population Based Training?

Ray Tune provides a number of schedulers that can be used to tune hyperparameters for reinforcement learning algorithms. The scheduler we use in this sample, the Population Based Training (PBT) algorithm, is one of the most popular schedulers for reinforcement learning algorithms.

While there are other [schedulers](https://docs.ray.io/en/releases-2.3.0/tune/api/schedulers.html) in Ray that you can leverage for hyperparameter search, we recommend PBT for a few reasons:

1. **Efficient Exploration**: PBT is based on the idea of iteratively exploring and exploiting the hyperparameter search space. This approach enables PBT to quickly explore a wide range of hyperparameters and identify promising ones for further optimization. As a result, PBT can often converge to good hyperparameter settings faster than other methods.
2. **Dynamic Adaptation**: PBT can adapt hyperparameters dynamically during the training process. This means that PBT can change the hyperparameters as the training progresses and adapt to changing conditions. For example, if a particular set of hyperparameters is performing well, PBT can allocate more resources to it, while reducing the resources allocated to poorly performing hyperparameters.
3. **Resource Efficiency**: PBT is resource-efficient because it optimizes hyperparameters in a distributed manner. Instead of training a single model with a fixed set of hyperparameters, PBT trains a population of models with different hyperparameters concurrently. This approach makes PBT more efficient because it can explore the search space faster and avoid getting stuck in local minima.

A good overview of population based training is available [here](https://docs.ray.io/en/releases-2.3.0/tune/examples/pbt_guide.html).

## How to Select Hyperparameter Mutations Space?

The first design decision you will need to make is the `hyperparameter_mutations` dictionary. This dictionary specifies the mutations of the hyperparameters to be tuned by the PBT algorithm. The mutations are specified as a function of the search space for each hyperparameter. For example, the following code snippet shows the default hyperparameter mutations for the PPO algorithm in our sample:

```python
hyperparam_mutations = {
    "lambda": lambda: random.uniform(0.9, 1.0),
    "clip_param": lambda: random.uniform(0.01, 0.5),
    "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    "num_sgd_iter": lambda: random.randint(1, 30),
    "sgd_minibatch_size": lambda: random.randint(128, 16384),
    "train_batch_size": lambda: random.randint(2000, 160000),
}
```

Here `hyperparam_mutations`  is a dictionary where key/value pair specifies a resampling function for each hyperparameter, or a range of values to sample from. For example, the `lambda` hyperparameter is resampled from a uniform distribution between 0.9 and 1.0 whereas the `lr` hyperparameter is resampled from a list of allowed values.


This dictionary is then passed to the `PopulationBasedTraining` scheduler as follows:

```python
pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=120,
    resample_probability=0.25,
    hyperparam_mutations=hyperparam_mutations,
    custom_explore_fn=explore,
)
```

where we additionally specify the attribute to use for time tracking and comparison (`time_attr`), the frequency (as a multiple of `time_attr`, so 120 seconds in this example) at which to continue the trial (exploit) or perturb (explore) the hyperparameters (`perturbation_interval`), the probability of resampling a hyperparameter (vs mutating) from their resampling distribution (`resample_probability`), and a custom function to explore the hyperparameter space (`custom_explore_fn`), which in our example is set to simply ensure that sufficient environment samples have been collected for a training iteration.


## How to Select the Stopping Criteria?

The next design decision you will need to make is the when to stop your experiment, which is specified through the `stopping_criteria` dictionary. This dictionary specifies the stopping criteria for the PBT algorithm. The stopping criteria are specified as a function of the training iteration and the mean episode reward of the model. For example, the following code snippet shows the default stopping criteria for the PPO algorithm in our sample:

```python
stopping_criteria = {"training_iteration": 100, "episode_reward_mean": 300}
```

which specifies that the algorithm should stop when either the number of training iterations reaches 100 or the mean episode reward (summed over episodes and averaged over a training batch, i.e., $\frac{1}{N} \sum_{j=1}^{N}\sum_{i=1}^T r_{i,j}$, where there are $N$ episodes in the training batch and $T$ iterations in each episode) reaches 300. You should set this based on what your desired average reward is (and be aware of the episode horizon) and how long you want to train for. Note that both of these are in terms of training iterations (where each training iteration is one SGD update), which would include a different number of environment steps because we are modifying the `sgd_minibatch_size` in our hyperparameters.

Further guidance on defining stopping criteria for Tune expeirments is available in the [Ray docs](https://docs.ray.io/en/latest/tune/tutorials/tune-stopping.html).

## Instantiating the Tuner

The `Tuner` object is created using the Tuner class from Ray Tune. The `TuneConfig` parameter of the Tuner constructor specifies the tuning configuration, including the metric used for evaluation, the mode of optimization (maximization or minimization), the scheduler object that specifies the PBT algorithm and the number of samples to be evaluated by the PBT algorithm. This is finally fed into the `Tuner` itself, along with the algorithm to be tuned, the initial configuration of the algorithm, and the number of workers to use for sampling:


```python
tuner = tune.Tuner(
    # the algorithm/trainable to be tuned
    "PPO",
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean",
        mode="max",
        scheduler=pbt,
        # the number of hyperparameters to sample
        num_samples=num_tune_samples,
    ),
    # specify the initial config input into the trainer
    # these are the initial samples used, which are then mutated by
    # the population based training algorithm if they are specified in
    # `hyperparam_mutations`.
    # the `num_workers` specifies the number of sample collection workers
    # that are used for gathering samples
    # the `num_cpus` specifies the number of CPUs for each training trial
    # here `num_workers=4` and `num_cpus=1` means we will use 5 cpus
    # if you want to run these trials concurrently, then you will need
    # CLUSTER_CPUS >= 5 x num_tune_samples
    # otherwise the PBT scheduler will round-robin between training each trial
    param_space={
        "env": env_name,
        "kl_coeff": 1.0,
        "num_workers": 4,
        "num_cpus": 1,  # number of CPUs to use per trial
        "num_gpus": 0,  # number of GPUs to use per trial
        # For DiagGaussian action distributions, make the second half of the model
        # outputs floating bias variables instead of state-dependent. This only
        # has an effect is using the default fully connected net.
        # avoid this for non-continuous action spaces
        "model": {"free_log_std": True},
        # These params are tuned from a fixed starting value.
        "lambda": 0.95,
        "clip_param": 0.2,
        "lr": 1e-4,
        # These params start off randomly drawn from a set.
        "num_sgd_iter": tune.choice([10, 20, 30]),
        "sgd_minibatch_size": tune.choice([128, 512, 2048]),
        "train_batch_size": tune.choice([10000, 20000, 40000]),
    },
    # MLFlow callback uses parent_run_id and tracks all hyperparameter
    # runs as child jobs
    run_config=air.RunConfig(
        stop=stopping_criteria,
        callbacks=[
            MLflowLoggerCallback(
                tags={MLFLOW_PARENT_RUN_ID: current_run.info.run_id},
                experiment_name="pbt_ppo",
                save_artifact=True,
            )
        ],
    ),
)
```


Here we are passing the algorithm we want to tune (PPO), the tuning configuration, the initial configuration of the algorithm, and the number of workers to use for sampling.


### Param Space vs Hyperparameter Mutations

In the above `Tuner` call, we pass the `param_space` dictionary, which specifies the initial configuration of the algorithm. This is the configuration that is used to initialize the algorithm and is then mutated by the PBT algorithm. If the parameters were ommitted from `param_space` but were provided in `hyperparam_mutations`, then the initial values would be also sampled from `hyperparam_mutations`. The hyperparameters that are specified in the `param_space` dictionary are not mutated by the PBT algorithm unless they are also in `hyperparam_mutations`. For example, in the above code snippet, the `num_workers` hyperparameter is specified in the `param_space` dictionary but not in `hyperparam_mutations`, and is therefore not mutated by the PBT algorithm. Conversely, the `num_sgd_iter` hyperparameter is specified in the `hyperparam_mutations` dictionary and is mutated by the PBT algorithm.


### Selecting the Number of CPUs and Workers for Tuning

Since PBT can train multiple trials concurrently, you can speed up the trial runtime by leveraging available CPUs to parallelize training. To do so effectively, you will need to specify the number of CPUs and workers to use for tuning. The `num_cpus` parameter specifies the number of CPUs to use for each trial, and the `num_workers` parameter specifies the number of workers to use for sampling from your simulation environments. In the above code snippet, we are using 4 workers and 1 CPU per trial, so we will need 5 CPUs to run _each_ trial. If you do not have enough CPUs to run all the trials concurrently (i.e., `5 * num_tune_samples`), then the PBT scheduler will round-robin between training each subsequent trial. For example, if you only have 10 CPUs (as we have requested in our AML `job.yml` for this experiment), then the PBT scheduler will train 2 trials concurrently, and then train the remaining trials sequentially as resources become available.


## Further Resources

- [Using RLlib with Tune](https://docs.ray.io/en/latest/tune/examples/pbt_ppo_example.html)
- [Visualizing PBT](https://docs.ray.io/en/latest/tune/examples/pbt_visualization/pbt_visualization.html)
- [Guide to Parallelism and Resources for Ray Tune](https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html)
- [A Guide to PBT with Tune](https://docs.ray.io/en/releases-2.3.0/tune/examples/pbt_guide.html)
