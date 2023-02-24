import json
import os
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import numpy as np
import sys
from casadi import *
from scipy import interpolate

from typing import Dict, Any, Union

from sim.cstr_sim import CSTRSimulation
from sim.cstr_solver import CSTR_Solver

# Import do_mpc package:
import do_mpc


# time step (seconds) between state updates
#Δt = 1

π = math.pi

class non_lin_mpc(CSTRSimulation):

    # def __init__():

    def reset(
        self,
        config: Dict[str, Any] = {},
    ):

        super().reset(config)

        # Initialize the control parameters based on new scenario.
        self.init_control()
    
        # Initialize mpc actions.
        self.nonlin_mpc_Tc_init = 0
        self.nonlin_mpc_Tc_adjust = 0


    def step(
        self,
        action: Dict[str, Any] = {}
    ):

        if self.debug:
            print(f"[NON-LIN-MPC] STEP: Method called. Received action Dict == {action}")

        # Take in and apply MPC characteristics to be updated.
        nonlin_mpc_Tc_init = None
        if "nonlin_mpc_Tc_init" in action.keys():
            nonlin_mpc_Tc_init = action["nonlin_mpc_Tc_init"]
        else:
            print(f"No valid actions parsed to compute_action function in nonlin-mpc control. Received action Dict == {action}")

        # Run Linear MPC model.
        self.init_control(nonlin_mpc_Tc_init)
        new_Tc = self.compute_best_action(nonlin_mpc_Tc_init)
        # Get action for simulation.
        Tc_adjust = new_Tc - self.Tc
        action = dict([("Tc_adjust", Tc_adjust)])
        self.nonlin_mpc_Tc_adjust = Tc_adjust

        if self.debug:
            print(f"[NON-LIN-MPC] STEP: Call super step. Action Dict == {action}")

        # Run simulation model with recommended action by Linear MPC.
        super().step(action)
        
        if self.debug:
            print(f"[NON-LIN-MPC] STEP: Exit step function.")
        

    def get_state(self):

        if self.debug:
            print(f"[NON-LIN-MPC] GET_STATE: Method called.")

        # Get sim states.
        states = super().get_state()

        if self.debug:
            print(f"[NON-LIN-MPC] GET_STATE: Append extra states to received states: {states}.")

        # Last Tc_adjust used by MPC.
        states["nonlin_mpc_Tc_init"] = float(self.nonlin_mpc_Tc_init)
        # Append recommended value of Tc_adjust (before being capped by solver).
        states["nonlin_mpc_Tc_adjust"] = float(self.nonlin_mpc_Tc_adjust)

        if self.debug:
            print(f"[NON-LIN-MPC] GET_STATE: Retrieve final states after addition: {states}.")

        return states



    def init_control(self, nonlin_mpc_Tc_init = None):

        if self.debug:
            print(f"[NON-LIN-MPC] INIT_CONTROL: Method called.")

        Cr0 = self.Cr
        Tr0 = self.Tr
        Tc0 = self.Tc
        if nonlin_mpc_Tc_init != None:
            self.nonlin_mpc_Tc_init = nonlin_mpc_Tc_init
            Tc0 += nonlin_mpc_Tc_init
        else:
            self.nonlin_mpc_Tc_init = 0

        # Call the CSTR solver for mpc modeling.
        solver = CSTR_Solver(Tr = Tr0,
                             Cr = Cr0,
                             Tc = Tc0,
                             ΔTc = 0,
                             step_time = self.step_time,
                             edo_solver_n_its = 1,
                             debug=False)
        
        # MPC MODEL
        self.model_type = 'continuous' # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(self.model_type)
        
        # State - Environment variables CV
        Cr = self.model.set_variable(var_type='_x', var_name='Cr', shape=(1,1)) # Concentration
        Tr = self.model.set_variable(var_type='_x', var_name='Tr', shape=(1,1)) # Temperature

        # define measurements:
        self.model.set_meas('Cr', Cr, meas_noise=True)
        self.model.set_meas('Tr', Tr, meas_noise=True)

        # Input - Manipulated Variable MV
        Tc = self.model.set_variable(var_type='_u', var_name='Tc') # Coolant temperature

        # Non-fixed Time Varying parameters:
        Cref = self.model.set_variable(var_type='_tvp', var_name='Cref')
        Tref = self.model.set_variable(var_type='_tvp', var_name='Tref')

        # Process Dynamics Model
        self.model.set_rhs('Cr', solver.model_Cr(Cr, Tr) )
        self.model.set_rhs('Tr', solver.model_Tr(Cr, Tr, Tc) )

        # Build the model
        self.model.setup()

        # SETUP CONTROLLER
        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {
            'n_horizon': 20,
            'n_robust': 1,
            'open_loop': 0,
            #'t_step': 0.1, #<version NLMM.py>
            't_step': self.step_time,
            'store_full_solution': True,
            # Use MA27 linear solver in ipopt for faster calculations:
            #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'} #<version NLMM.py>
        }

        self.mpc.set_param(**setup_mpc)
        # scaling variables
        self.mpc.scaling['_x', 'Tr'] = 100
        self.mpc.scaling['_u', 'Tc'] = 100
        #mpc.scaling['_u', 'ω_s'] = 1

        # SETUP OBJECTIVE - COST FUNCTION
        _x = self.model.x
        _tvp = self.model.tvp
        _u = self.model.u #<version NLMM.py>
        # <version cstr_mpc_simulator.py (CMS.py)>
        mterm = ((_x['Cr'] - _tvp['Cref']))**2 # terminal cost
        lterm = ((_x['Cr'] - _tvp['Cref']))**2 # stage cost
        # <version non_linear_mpc_model.py (NLMM.py)>
        #mterm = ((_x['Cr'] - CrSP))**2 # terminal cost
        #lterm = ((_x['Cr'] - CrSP))**2 # stage cost

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(Tc = 1.5 * 1) # input penalty 1e-2 , 1000

        # SETUP CONSTRAINTS
        # constraints on states
        self.mpc.bounds['lower', '_x', 'Cr'] = 0.1
        self.mpc.bounds['upper', '_x', 'Cr'] = 12
        
        self.mpc.bounds['upper', '_x', 'Tr'] = 400
        self.mpc.bounds['lower', '_x', 'Tr'] = 100

        #mpc.set_nl_cons('Cr', _x['Cr'], ub=2, soft_constraint=True, penalty_term_cons=1e4) #<version NLMM.py>

        # lower bounds of the inputs
        #mpc.bounds['lower', '_u', 'dTc'] = -10 #<version NLMM.py>
        self.mpc.bounds['lower', '_u', 'Tc'] = 273 #273

        # upper bounds of the inputs
        #mpc.bounds['upper', '_u', 'dTc'] = 10 #<version NLMM.py>
        self.mpc.bounds['upper', '_u', 'Tc'] = 322

        # TIME VARYING PARAMETERS
        # in optimizer configuration:
        tvp_temp = self.mpc.get_tvp_template()
        tvp_temp['_tvp', :] = np.array([2])

        def tvp_fun(t_now):
            aux_Cref, aux_Tref = self.update_references(t_now+self.it_time)
            y = float(aux_Cref)
            tvp_temp['_tvp', :] = np.array([y])
            return tvp_temp

        self.mpc.set_tvp_fun(tvp_fun)

        # Initialize MPC setup
        self.mpc.setup()

        # ESTIMATOR
        self.estimator = do_mpc.estimator.StateFeedback(self.model)

        # SIMULATOR
        self.simulator = do_mpc.simulator.Simulator(self.model)
        params_simulator = {
            #'integration_tool': 'cvodes', #<version NLMM.py>
            #'abstol': 1e-10, #<version NLMM.py>
            #'reltol': 1e-10, #<version NLMM.py>
            't_step': self.step_time
        }

        self.simulator.set_param(**params_simulator)

        # SETUP TV parameters
        p_num = self.simulator.get_p_template()
        tvp_num = self.simulator.get_tvp_template()

        # function for time-varying parameters
        def tvp_fun(t_now):
            return tvp_num

        # uncertain parameters
        def p_fun(t_now):
            return p_num

        self.simulator.set_tvp_fun(tvp_fun)
        self.simulator.set_p_fun(p_fun)

        # Initialize simulator setup
        self.simulator.setup()

        # INITIALIZE ENVIRONMENT STATES
        # Set the initial state (Cr, Tr) of mpc, simulator and estimator:
        x0 = self.simulator.x0
        x0['Cr'] = Cr0
        x0['Tr'] = Tr0

        self.mpc.x0 = x0
        self.simulator.x0 = x0
        self.estimator.x0 = x0

        # Set the initial action (Tc) of mpc, simulator and estimator:
        u0 = self.simulator.u0
        u0['Tc'] = Tc0

        self.mpc.u0 = u0
        self.simulator.u0 = u0
        self.estimator.u0 = u0

        # Initialize MPC Control
        self.mpc.set_initial_guess()



    def compute_best_action(self, XX):

        if self.debug:
            print(f"[NON-LIN-MPC] COMPUTE_BEST_ACTION: Method called. XX == {XX}")

        k = int(self.it_time/self.step_time)
        
        # INITIALIZE ENVIRONMENT STATES
        # Set the initial state (Cr, Tr) of mpc, simulator and estimator:
        x0 = self.simulator.x0
        x0['Cr'] = self.Cr
        x0['Tr'] = self.Tr
        self.u0 = self.mpc.make_step(x0) # get MPC next action
        #limit controller actions from -10 to 10 degrees Celsius
        Tc_adjust_limit = 10*self.step_time
        if self.u0[0][0] - self.Tc >= Tc_adjust_limit:
            self.u0 = np.array([[self.Tc + Tc_adjust_limit]])
        elif self.u0[0][0] - self.Tc <= -Tc_adjust_limit:
            self.u0 = np.array([[self.Tc - Tc_adjust_limit]])
        
        #Add Noise
        #For random samples from N(mu,sigma**2), use: /
        #sigma * np.random.randn(...) + mu
        #error_var = 0.1
        error_var = self.noise_percentage
        σ_max1 = error_var * (8.5698 -2)
        σ_max2 = error_var * ( 373.1311 - 311.2612)
        mu = 0
        #v0 = mu + σ_max * np.random.randn(0, 1)
        v0 = np.array([mu + σ_max1* np.random.randn(1, 1)[0],mu + σ_max2* np.random.randn(1, 1)[0]])
        
        y_next = self.simulator.make_step(self.u0) #,v0=v0) # MPC

        # Get all state values.
        state_ops = y_next.reshape((1,2))
        
        self.x0 = self.estimator.make_step(y_next) # Simulation next step

        return self.u0[0][0]



#print(non_lin_mpc(noise = 0, CrSP = 8.57 , Cr0 = 8.5698 , Tr0 = 311.2639 , Tc0 = 292))

        

