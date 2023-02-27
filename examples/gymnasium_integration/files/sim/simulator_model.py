from sim.cstr_sim import CSTRSimulation
from sim.control_linear_mpc import linear_mpc
from sim.control_nonlin_mpc import non_lin_mpc
from typing import NamedTuple, Dict, Any, Union


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

        # Initialize control type from None to default ("direct_control").
        self.control_mode = None
        self.select_control_mode({})
        
        pass


    @property
    def sim_name(self):
        return "CSTR_sim"


    def reset(self, config) -> Dict[str, Any]:
        """ Reset any state from the previous episode and get ready to start a new episode. """
        
        # Start simulation with selected config.
        self.sim.reset(config=config)

        return self.sim.get_state()
    

    def termination(self) -> bool:
        """ Return True if the episode has reached a terminal condition. """
        return self.sim.termination()
    

    def truncation(self) -> bool:
        """ Return True if the episode has reached a truncation condition. """
        return self.sim.truncation()


    def step(self, action) -> Dict[str, Any]:
        """ Apply the specified action and perform one simulation step. """
        # Apply action to sim.
        self.sim.step(action=action)

        # If 'sim_halted' is set to True, that indicates that the simulator is unable to continue and
        # the episode will be discarded. This simulator sets it to False because it can always continue.
        return self.sim.get_state()


    def select_control_mode(self, config):

        # If no control type is given, default to 'direct_control'.
        config = dict([("control_mode", "direct_control")])


        # initialize sim
        if config["control_mode"] == "direct_control":
            self.control_mode = "direct_control"
            self.sim = CSTRSimulation(render = self.render,
                                      log_data = self.log_data,
                                      debug = self.debug)
