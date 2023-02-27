from sim.cstr_sim import CSTRSimulation
from typing import NamedTuple, List, Dict, Any, Union
import numpy as np

class SimulatorModel:
    """
    Manages the LunarLander model that represents the simulation for this sample.
    Implements the reset and step methods required for a Bonsai simulator.
    """

    def __init__(self,
        render: bool = False,
        log_data: Union[bool, str] = False,
        debug: bool = True,):
        """ Perform global initialization here if needed before running episodes. """

        # render functionality
        self.render = render
        # logging features
        self.log_data = log_data
        # debug functionality
        self.debug = debug
        
        # initialize the simulator
        self.sim = CSTRSimulation(render = self.render,
                            log_data = self.log_data,
                            debug = self.debug)
        
        pass


    @property
    def sim_name(self):
        return "CSTR_sim"


    def reset(self, config) -> Dict[str, Any]:
        """ Reset any state from the previous episode and get ready to start a new episode. """
        
        # Start simulation with selected config.
        self.sim.reset(config=config)

        return self.sim.get_state()
    

    def compute_reward_term_and_trun(self) -> List[float, bool, bool]:
        """ Return the current reward of the simulator. """
        state_dict = self.sim.get_state()

        # compute the concentration error
        cr_error = abs(state_dict["Cr"] - state_dict["Cref"])

        # get the reward from the simulator
        reward = -cr_error

        # adjust based on termination and truncation
        terminated = self._termination()
        truncated = self._truncation()
        if terminated:
            reward += -10
        elif truncated:
            reward += 0

        return reward, terminated, truncated
    

    def get_gym_specs(self) -> float:
        """ Return the current reward of the simulator. """

        # the observation will be 4: Tr, Cr, Cref, Tc
        self.state_len = 4
        # there is 1 possible actions: Tc
        self.action_len = 1

        return self.state_len, self.action_len
    

    def sim_state_to_gym(self) -> Dict[str, Any]:
        """ Return the current state of the simulator for gym, normalized between [-1 and 1]. """
        state_dict = self.sim.get_state()

        # convert the state to a Gym state and normalize it
        state = [state_dict["Tr"] / 600,
                state_dict["Cr"] / 10,
                state_dict["Cref"] / 10,
                state_dict["Tc"] / 600]
        state = np.array(state, np.float32)

        return state
    

    def _gym_action_to_sim(self, action) -> Dict[str, Any]:
        """ Return the current state of the simulator for gym, normalized between [-1 and 1]. """

        # convert the Gym action to sim and un-normalize it
        action_dict = dict([("Tc_adjust", action[0] * 10)])

        return action_dict
    

    def _termination(self) -> bool:
        """ Return True if the episode has reached a terminal condition. """
        return self.sim.termination()
    

    def _truncation(self) -> bool:
        """ Return True if the episode has reached a truncation condition. """
        return self.sim.truncation()
    

    def step(self, action) -> Dict[str, Any]:
        """ Apply the specified action and perform one simulation step. """

        # Apply action to sim.
        action_dict = self.sim.gym_action_to_sim(action=action)
        self.sim.step(action=action_dict)

        # If 'sim_halted' is set to True, that indicates that the simulator is unable to continue and
        # the episode will be discarded. This simulator sets it to False because it can always continue.
        return self.sim.get_state()



