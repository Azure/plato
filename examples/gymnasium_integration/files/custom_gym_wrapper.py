import gymnasium as gym
#import gym
from collections import deque
import numpy as np
from sim.simulator_model import SimulatorModel
from ray.rllib.env.base_env import BaseEnv


class Gym_Wrapper(gym.Env):
    
    def __init__(self, config, **kwargs):
        super().__init__()

        # define the simulator model
        self.sim = SimulatorModel()

        # dimensions of the grid
        self.XX = kwargs.get('XX',3)

        # get specs for states and actions from the simulator
        self.state_dim, self.action_dim = self.sim.get_gym_specs()

        # configure states
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)
        assert np.array([0]*self.state_dim, np.float32) in self.observation_space
        #self.observation_space = MultiDiscrete([self.width, self.height])

        # configure actions
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)


    def step(self, action):
        ''' apply the supplied action '''

        # take the action
        self.sim.step(action)

        # convert the state to a Gym state
        state = self.sim.sim_state_to_gym()

        # get -1 reward for each step
        # - except at the terminal state which has zero reward
        # - set the 'terminated' flag if we've reached thermal runaway
        reward, terminated, truncated = self.sim.compute_reward_term_and_trun()

        info = {}
        return state, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        config = {}
        state_dict = self.sim.reset(config)
        # convert the state to a Gym state
        state = self.sim.sim_state_to_gym()

        info = {}
        return state,info


    def render(self, action=0, reward=0 ):
        states=[]
        print(f"(action: {action}): (states: {states}) reward = {reward}")


if __name__ == "__main__":
    # create an instance of our custom environment
    env = Gym_Wrapper({})

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