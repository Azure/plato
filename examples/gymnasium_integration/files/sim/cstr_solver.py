import traceback
from dataclasses import dataclass
import math
from math import exp
import numpy as np
from scipy.integrate import odeint

π = math.pi


@dataclass
class CSTR_Solver:

    # Problem states & actions.
    Cr: float       # Concentration at the reactor's output (kmol/m3)
    Tr: float       # Temperature at the reactor's output (Kelvin)
    Tc: float       # Temperature of the coolant (Kelvin)
    ΔTc: float      # Delta action to apply to the temperature of the coolant.


    # Running parameters
    edo_solver_n_its: int = 1      # Number of runs to perform to resolve the ODE: Ordinary Differential Equation.
    step_time: int = 1             # Step time to run the simulation for at each simulation execution.
    

    # Problem constants.
    F: float = 1 # Volumetric flow rate (m3/h)
    V: float = 1 # Reactor volume (m3)
    k0: float = 34930800    # Pre-exponential nonthermal factor (1/h)
    E: float = 11843        # Activation energy per mole (kcal/kmol)
    R: float = 1.98588  # Boltzmann's ideal gas constant (kcal/(kmol·K))
    ΔH: float = -5960   # Heat of reaction per mole kcal/kmol
    phoCp: float = 500  # Density multiplied by heat capacity (kcal/(m3·K))
    UA: float = 150     # Overall heat transfer coefficient multiplied by tank area (kcal/(K·h))
    Cafin: float = 10   # Feed Concentration (kmol/m^3)
    Tf: float = 298.2   # Feed Temperature (K)

    # debug functionality through print statements (old school)
    debug: bool = False


    def __post_init__(self):
        """""
        By default, resolve the ODE right after the object is generated.
        """""
        self.run_sim()


    def run_sim(self):
        """""
        Method that queues the computation of the ODE (ordinary differential equation).
        """""

        # Setup current conditions.
        # TODO: Question for Octavio: Why was the solver run for 3 iterations (edo_solver_n_its == 2)?
        action_limit = int(10*self.step_time)
        if abs(self.ΔTc) > action_limit:
            try:
                raise ValueError(f"Current provided value of {self.ΔTc} exceeds the max ramp value of {10*self.step_time}.")
            except Exception:
                print(traceback.format_exc())
                self.ΔTc = action_limit if self.ΔTc > 0 else -action_limit
                print(f"Execution continues, action has been capped at {self.ΔTc}.\n")
        elif self.ΔTc is None:
            try:
                raise ValueError(f"Current provided value of {self.ΔTc} is None.")
            except Exception:
                self.ΔTc = 0
                print(f"Execution continues, action has been set to {0}.\n")
        elif math.isnan(self.ΔTc):
            try:
                raise ValueError(f"Current provided value of {self.ΔTc} is NaN.")
            except Exception:
                self.ΔTc = 0
                print(f"Action provided is NAN. Execution continues, action has been set to {self.ΔTc}.\n")
        

        # Setup current conditions.
        z0 = [self.Cr, self.Tr]
        t = np.linspace(0, self.step_time, self.edo_solver_n_its+1)
        
        # Setup the changing conditions to the coolant temp.
        #u = [0,self.ΔTc]
        u = np.zeros(self.edo_solver_n_its+1)
        u[0] = self.ΔTc #-1 0

        # Setup output variables to store intermediate results.
        x = np.empty_like(t)
        y = np.empty_like(t)
        # Initialize the first array values to whichever current conditions.
        x[0] = z0[0]
        y[0] = z0[1]

        # Run the solver for 'n' steps to solve the ODE (Ordinary Differential Equation).
        for i in range(0, self.edo_solver_n_its):
            tspan = [t[i], t[i+1]]
            if self.debug:
                print(f"aux: z0 {z0}, tspan {tspan}, self.Tc {self.Tc}, u[i] {self.Tc+u[i]}")
            z = odeint(self.model, z0, tspan, args=(u[i],))
            x[i+1] = z[-1][0]
            y[i+1] = z[-1][1]
            z0 = z[-1]
            if self.debug:
                print(z0)
        
        # Extract the upadted values for reactor temp & conc, as well as coolant temp.
        self.Cr = x[-1]
        self.Tr = y[-1]
        if self.debug:
            print("self.Cr", self.Cr)
            print("self.Tr", self.Tr)
        self.Tc += self.ΔTc
    

    def model(self, z, t, u):
        """""
        Method that resolves the ODE (ordinary differential equation).
        """""
        x = z[0]  # Reactor Concentration
        y = z[1]  # Reactor Temperature
        
        ## COMMENT OUT PREVIOUS DIRECT COMPUTATION.
        ## reaction rate
        #rA = self.k0 * exp(-self.E/(self.R*y))*x
        #
        ## Calculate concentration derivative
        #dxdt = (self.F/self.V * (self.Cafin - x)) - rA
        ## Calculate temperature derivative
        #dydt = (self.F/self.V *(self.Tf-y)) \
        #       - ((self.ΔH/self.phoCp)*rA) \
        #       - ((self.UA /(self.phoCp*self.V)) * (y - (self.Tc + u)))

        # Calculate concentration derivative
        dxdt = self.model_Cr(x, y)

        # Calculate temperature derivative
        dydt = self.model_Tr(x, y, self.Tc + u)
        
        debug_type = False
        if debug_type:
            print("type(self.k0)", (self.k0))
            print("type(self.E)", (self.E))
            print("type(self.R)", (self.R))
            print("type(y)", (y))
            print("type(x)", (x))
            rA = self.model_rA(y, x)
            print("type(rA)", (rA))
            print("type(self.F)", (self.F))
            print("type(self.V)", (self.V))
            print("type(self.Tf)", (self.Tf))
            print("type(self.ΔH)", (self.ΔH))
            print("type(self.phoCp)", (self.phoCp))
            print("type(self.UA)", (self.UA))
            print("type(self.Tc + u)", (self.Tc + u))
            print("type(dxdt)", (dxdt))
            print("type(dydt)", (dydt))
            a = 1/0

        dzdt = [dxdt,dydt]
        return dzdt
    
    
    def model_rA(self, Tr, Cr):
        # reaction rate
        rA = self.k0 * np.exp(-self.E/(self.R*Tr))*Cr
        return rA
    
    
    def model_Cr(self, Cr, Tr):
        # Calculate concentration derivative
        rA = self.model_rA(Tr, Cr)
        dCr = (self.F/self.V * (self.Cafin - Cr)) - rA
        return dCr
    
    
    def model_Tr(self, Cr, Tr, Tc):
        # Calculate temperature derivative
        rA = self.model_rA(Tr, Cr)
        dTr = (self.F/self.V *(self.Tf-Tr)) \
               - ((self.ΔH/self.phoCp)*rA) \
               - ((self.UA /(self.phoCp*self.V)) * (Tr - Tc))
        return dTr

        
        

