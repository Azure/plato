
import random
from typing import Any, Dict, List, Union
import numpy as np

class RLSimSpec:
    """
    Manages the Sim Properties that represents the simulation for this sample.
    Implements the reset and step methods required for a RL-AML simulator.
    """

    # STATES
    # Set the number of states that are provided as input to the RL agent (must match shape in 'sim_state_to_gym' method).
    obs_state_len: int = 4
    # Set all the states to be logged throughout RL training session, including KPI metrics.
    sim_states_to_log: List[str] = ["Tr", "Cr", "Cref", "Tc", "Tc_adjust", "kpi_rms_conc_error"]

    # ACTIONS
    action_len: int = 1

    # RL PROPERTIES
    gamma: int = 1

    def __init__(self,
        debug: bool = False,):
        """ Perform global initialization here if needed before running episodes. """

        # debug functionality
        self.debug = debug
        
        pass


    def compute_reward_term_and_trun(self, state_dict, terminated, truncated) -> Union[float, bool, bool]:
        """ Return the current reward of the simulator. """

        # compute the concentration error
        cr_error = abs(state_dict["Cr"] - state_dict["Cref"])

        # get the reward from the simulator
        reward = np.exp(-cr_error * self.gamma)

        # adjust based on termination and truncation
        if terminated:
            reward += -10
        elif truncated:
            reward += 0

        return reward, terminated, truncated
    

    def sim_state_to_gym(self, state_dict) -> Dict[str, Any]:
        """ Return the current state of the simulator for gym, normalized between [-1 and 1]. """

        # convert the state to a Gym state and normalize it
        state = [state_dict["Tr"] / 600,
                state_dict["Cr"] / 10,
                state_dict["Cref"] / 10,
                state_dict["Tc"] / 600]
        state = np.array(state, np.float32)

        assert state.shape == (self.obs_state_len,), \
        f"The observable state shape for RL training is not correct. Expected: {self.obs_state_len}, Actual: {state.shape}."

        return state
    

    def gym_action_to_sim(self, action) -> Dict[str, Any]:
        """ Return the current state of the simulator for gym, normalized between [-1 and 1]. """

        # convert the Gym action to sim and un-normalize it
        action_dict = dict([("Tc_adjust", action[0] * 10)])

        return action_dict
    

    def get_gym_specs(self) -> float:
        """ Return the current reward of the simulator. """

        return self.obs_state_len, self.action_len


    def get_states_to_log(self):
        """ Return all the sim states to monitor during brain training. """

        return self.sim_states_to_log