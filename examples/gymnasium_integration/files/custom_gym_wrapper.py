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

        # the observation will be 4: Tr, Cr, Cref, Tc
        self.observation_space = gym.spaces.Box(low=-100.0, high=600.0, shape=(4,), dtype=np.float32)
        assert np.array([0,0,0,0], np.float32) in self.observation_space
        #self.observation_space = MultiDiscrete([self.width, self.height])

        # there is 1 possible actions: Tc
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)


    def step(self, action):
        ''' apply the supplied action '''

        # take the action
        action_dict = dict([("Tc_adjust", action[0])])
        state_dict = self.sim.step(action_dict)

        # convert the state to a Gym state
        state = self.get_gym_state(state_dict)

        # set the 'terminated' flag if we've reached thermal runaway
        terminated = self.sim.termination()
        truncated = self.sim.truncation()

        # get -1 reward for each step
        # - except at the terminal state which has zero reward
        reward = self.compute_reward(state_dict)
        if terminated:
            reward += -10
        elif truncated:
            reward += 0

        info = {}
        return state, reward, terminated, truncated, info
    

    def get_gym_state(self, state_dict):
        ''' convert the simulator state to a Gym state '''

        # convert the state to a Gym state
        state = [state_dict["Tr"],
                state_dict["Cr"],
                state_dict["Cref"],
                state_dict["Tc"]]
        state = np.array(state, np.float32)

        return state
    

    def compute_reward(self, state_dict):
        ''' compute the reward for the supplied state '''
        
        # compute the concentration error
        cr_error = abs(state_dict["Cr"]-state_dict["Cref"])

        # get the reward from the simulator
        reward = -cr_error

        return reward


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        config = {}
        state_dict = self.sim.reset(config)
        # convert the state to a Gym state
        state = self.get_gym_state(state_dict)

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