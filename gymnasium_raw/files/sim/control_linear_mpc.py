import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gekko import GEKKO
from scipy import interpolate
import math
from typing import Dict, Any, Union
import random

from sim.cstr_sim import CSTRSimulation


class linear_mpc(CSTRSimulation):

    # def __init__():

    def reset(
        self,
        config: Dict[str, Any] = {},
    ):

        super().reset(config)

        # Initialize the control parameters based on new scenario.
        self.init_control()
    
        # Initialize mpc actions.
        self.lin_mpc_Tr_init = 2
        self.lin_mpc_Tc_adjust = 0


    def step(
        self,
        action: Dict[str, Any] = {}
    ):

        if self.debug:
            print(f"[LIN-MPC] STEP: Method called. Received action Dict == {action}")

        # Take in and apply MPC characteristics to be updated.
        self.lin_mpc_Tr_init = 2
        if "lin_mpc_Tr_init" in action.keys():
            self.lin_mpc_Tr_init = action["lin_mpc_Tr_init"]
        else:
            print(f"No valid actions parsed to compute_action function in linear-mpc control. Received action Dict == {action}")

        # Run Linear MPC model.
        new_Tc = self.compute_best_action(self.lin_mpc_Tr_init)
        # Get action for simulation.
        Tc_adjust = new_Tc - self.Tc
        action = dict([("Tc_adjust", Tc_adjust)])
        self.lin_mpc_Tc_adjust = Tc_adjust

        # Run simulation model with recommended action by Linear MPC.
        super().step(action)
        

    def get_state(self):

        # Get sim states.
        states = super().get_state()

        # Last lin_mpc_Tr_init used by MPC.
        states["lin_mpc_Tr_init"] = float(self.lin_mpc_Tr_init)
        # Append recommended value of Tc_adjust (before being capped by solver).
        states["lin_mpc_Tc_adjust"] = float(self.lin_mpc_Tc_adjust)

        return states


    def init_control(self):

        if self.debug:
            print(f"[LIN-MPC] INIT_CONTROL: Method called.")

        self.remote_server = True
        self.display_mpc_vals = False
    
        # GEKKO linear MPC
        self.m = GEKKO(remote=self.remote_server)

        ep_length_secs = 45
        ep_length_its = int(ep_length_secs/self.step_time)
        self.m.time = np.linspace(0, ep_length_secs, num=ep_length_its) #simulate 45 minutes (0.5 minute per controller action)

        # initial conditions
        self.Tc0 = self.Tc
        self.T0 = self.Tr_no_noise
        self.C0 = self.Cr_no_noise

        # Manipulated and Controlled Variables
        self.m.Tc = self.m.MV(value = self.Tc0, lb=273, ub=322) #mv with constraints
        self.m.T = self.m.CV(value = self.T0) #cv

        # Process dynamic model
        self.tau = self.m.Const(value = 3)
        self.Kp = self.m.Const(value = 0.75)
        self.m.Equation(self.tau * self.m.T.dt() == -(self.m.T - self.T0) + self.Kp * (self.m.Tc - self.Tc0))

        # MV tuning - Cooling Temp
        self.m.Tc.STATUS = 1
        self.m.Tc.FSTATUS = 0
        Tc_adjust_limit = 10*self.step_time
        self.m.Tc.DMAXHI = Tc_adjust_limit   # constrain movement up
        self.m.Tc.DMAXLO = -Tc_adjust_limit  # quick action down
        
        # CV tuning - Tr Reactor Temp
        self.m.T.STATUS = 1
        self.m.T.FSTATUS = 1
        self.m.T.SP = 311
        #m.T.UPPER = 400 #Upper constraint
        self.m.T.TR_INIT = 2
        self.m.T.TAU = 1.0 # time constant of trajectory

        self.m.options.CV_TYPE = 2  # the objective is an l2-norm (squared error)
        self.m.options.IMODE = 6    # MPC
        self.m.options.SOLVER = 3

        return
    

    def compute_best_action(self, lin_mpc_Tr_init):

        if self.debug:
            print(f"[LIN-MPC] COMPUTE_BEST_ACTION: Method called. lin_mpc_Tr_init = {lin_mpc_Tr_init}.")

        tr_init_abs = self.Tr + lin_mpc_Tr_init

        # COMPUTATION OF MPC
        # insert measurement
        self.m.T.MEAS = self.Tr_no_noise
        # update setpoint
        self.m.T.SP = self.Tref

        # solve MPC
        self.m.solve(disp=self.display_mpc_vals)
        # change to a fixed starting point for trajectory
        self.m.T.TR_INIT = tr_init_abs

        # retrieve new Tc values
        new_Tc = self.m.Tc.NEWVAL

        return new_Tc
