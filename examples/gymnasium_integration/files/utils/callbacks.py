'''RLlib callbacks module:
    Common callback methods to be passed to RLlib trainer.
'''

# Define our custom callback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from azureml.core import Run

class MyCallback(DefaultCallbacks):

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        print(
            "Algorithm.train() result: {} -> {} episodes".format(
                algorithm, result["episodes_this_iter"]
            )
        )
        # you can mutate the result dict to add new fields to return
        result["custom_metrics"] = dict([("episode_reward_mean", result["episode_reward_mean"]),
                                         ("episode_len_mean", result["episode_len_mean"]),
                                         ("episodes_total", result["episodes_total"])])
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
        
        run = Run.get_context()
        for k,v in result["custom_metrics"].items():
            run.log(name=k, value=v)

    
    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        """Runs on each episode step.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index: The index of the sub-environment that stepped the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """

        #super().on_episode_step(info, worker, base_env, policies, episode, env_index, **kwargs)

        run = Run.get_context()

        for agent,info in episode._last_infos.items():
            #print("[debug on_episode_step] episode._last_infos: ", k, v)

            for k,v in info.items():
                #episode.custom_metrics[k] = v
                run.log(name=k, value=v)

        pass
        

    def on_evaluate_end(self, *, algorithm, evaluation_metrics: dict, **kwargs, ) -> None:
        """Runs when the evaluation is done.

        Runs at the end of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            evaluation_metrics: Results dict to be returned from algorithm.evaluate().
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        pass