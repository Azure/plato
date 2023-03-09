from sim.cstr_sim import CSTRSimulation
from typing import NamedTuple, List, Dict, Any, Union
import numpy as np

class SimulatorModel:
    """
    Manages the CSTR model that represents the simulation for this sample.
    Implements the reset and step methods required for a Bonsai simulator.

    This model relies on the following CSTR sim methods:
    - reset(config: Dict[str, Any])
    - step(action: Dict[str, Any])
    - get_state() -> Dict[str, Any]
    - termination() -> bool
    - truncation() -> bool

    """

    def __init__(self,
        render: bool = False,
        log_data: Union[bool, str] = False,
        debug: bool = False,):
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
        
        print(f"[sim model] Resetting simulator with config: {config}")
        
        # Start simulation with selected config.
        self.sim.reset(config=config)

        return self.sim.get_state()
    

    def step(self, action) -> Dict[str, Any]:
        """ Apply the specified action and perform one simulation step. """

        # Apply action to sim.
        self.sim.step(action)

        # If 'sim_halted' is set to True, that indicates that the simulator is unable to continue and
        # the episode will be discarded. This simulator sets it to False because it can always continue.
        return self.sim.get_state()
    

    def termination(self) -> bool:
        """ Return True if the episode has reached a terminal condition. """
        return self.sim.termination()
    

    def truncation(self) -> bool:
        """ Return True if the episode has reached a truncation condition. """
        return self.sim.truncation()



