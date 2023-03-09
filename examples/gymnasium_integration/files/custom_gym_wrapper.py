import gymnasium as gym
#import gym
from collections import deque
import numpy as np
from sim.simulator_model import SimulatorModel
from ray.rllib.env.base_env import BaseEnv

from training_setup.rl_lesson_init import rl_lesson_init
from training_setup.rl_sim_spec import RLSimSpec

class Gym_Wrapper(gym.Env, gym.utils.EzPickle):
    
    def __init__(self,
                 config = {},
                 **kwargs):
        gym.utils.EzPickle.__init__(
            self,
            config,
        )

        super().__init__()
        
        # save the environmental config parsed from tune.run
        # - this includes the following feats: {, worker=1/11, vector_idx=0, remote=False} -
        self.config = config

        # define episode reset config
        self.rl_lesson_config = self.config.get('rl_lesson_config', {})

        # initialize sim specification
        self.rl_sim_spec = RLSimSpec()

        # define the simulator model
        self.sim = SimulatorModel()

        # dimensions of the grid. TODO: Remove once we validate no need to use kwargs.
        self.XX = kwargs.get('XX',3)

        # get specs for states and actions from the simulator
        self.state_dim, self.action_dim = self.rl_sim_spec.get_gym_specs()

        # configure states
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)
        assert np.array([0]*self.state_dim, np.float32) in self.observation_space
        #self.observation_space = MultiDiscrete([self.width, self.height])

        # configure actions
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)


    def step(self, action):
        ''' apply the supplied action '''

        # take the action
        sim_action = self.rl_sim_spec.gym_action_to_sim(action)
        state_dict = self.sim.step(sim_action)

        # convert the state to a Gym state
        state = self.rl_sim_spec.sim_state_to_gym(state_dict)
        # clip the state to the observation space
        state = np.clip(state, self.observation_space.low, self.observation_space.high)

        # get -1 reward for each step
        # - except at the terminal state which has zero reward
        # - set the 'terminated' flag if we've reached thermal runaway
        terminated = self.sim.termination()
        truncated = self.sim.truncation()
        reward, terminated, truncated = self.rl_sim_spec.compute_reward_term_and_trun(state_dict, terminated, truncated)

        info = {}
        # add states to track in the info dict (for logging)
        # - this is used by the 'monitor' wrapper to record the states
        for state_name in self.rl_sim_spec.get_states_to_log():
            info["state_" + state_name] = state_dict[state_name]

        return state, reward, terminated, truncated, info


    def reset(self, *, seed=None, options=None):
        # seed and options needed for ray 2.3.0 compatibility
        super().reset(seed=seed)
        
        # Setup values for sim config whenever "rl_lesson" has been defined.
        reset_config = rl_lesson_init(self.rl_lesson_config)
        state_dict = self.sim.reset(reset_config)
        # convert the state to a Gym state
        state = self.rl_sim_spec.sim_state_to_gym(state_dict)
        state = np.clip(state, self.observation_space.low, self.observation_space.high)

        info = {}
        return state, info


    def render(self, action=0, reward=0 ):
        # TODO: implement this
        # azaidi: I wonder if we can use gymnasium's visualization wrappers or utils to
        #         render episodes that we could serve or save to aml dashboard or mlflow.
        states=[]
        print(f"(action: {action}): (states: {states}) reward = {reward}")



if __name__ == "__main__":

    # setup rl_lesson config
    import yaml
    with open("training_setup/rl_lesson.yml", "r") as file:
        rl_lesson_config = yaml.safe_load(file)
        config = {"rl_lesson_config": rl_lesson_config}
        print("config: ", config)

        
    # create an instance of our custom environment
    env = Gym_Wrapper(config)

    print(env.action_space)
    print(env.observation_space)
    print(env.reset())
    print(env.step([1]))

    # use the Gymnasium 'check_env' function to check the environment
    # - returns nothing if the environment is verified as ok
    from gymnasium.utils.env_checker import check_env
    check_env(env)

    # use the Ray 'check_env' function to check the environment
    # -- requires pip install opencv-python --
    from ray.rllib.utils.pre_checks.env import check_env as ray_check_env
    ray_check_env(env)


#class ObservationWrapper(gym.ObservationWrapper):
#    def __init__(self, env):
#        super().__init__(env)
#        self.o_min = 0.
#        self.o_max = 5000.
#
#    def observation(self, ob):
#        # Normalize observations
#        ob = (ob - self.o_min)/(self.o_max - self.o_min)
#        return ob
#
#class RewardWrapper(gym.RewardWrapper):
#    def __init__(self, env):
#        super().__init__(env)
#        self.r_min = -500
#        self.r_max = 100
#
#    def reward(self, reward):
#        # Scale rewards:
#        reward = reward/(self.r_max - self.r_min)
#        return reward
#
#class ActionWrapper(gym.ActionWrapper):
#    def __init__(self, env):
#        super().__init__(env)
#
#    def action(self, action):
#        # Scale actions
#        action = action/10
#        return action