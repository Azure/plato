'''RLlib callbacks module:
    Common callback methods to be passed to RLlib trainer.
'''

# Define our custom callback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from azureml.core import Run

class MyCallback(DefaultCallbacks):

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
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
