import numpy as np
import scipy.integrate as siint
from timeit import default_timer as timer
from scipy import signal
from sympy import *
from sympy.functions import sign
import matplotlib.pyplot as plt
import time
from Code.S02_Allgemein.plot_results import plot_only_sim

class tank:
    """ class for simulation of tank systems with one or two tanks """

    def __init__(self, A, q, dt, noise_sigma, deadtime, model_qual=1):
        """ initialise tank object """
 
        self.model_qual = model_qual # quaility of model, only applies if model based detection is used
        self.q = q # m**2 cross section area of pipe
        self.A = A # m**2 cross section area of tank
        self.g = 9.81 # gravitational constant
        self.dt = dt # time step size of simulation
        self.noise_sigma = noise_sigma # measurement noise

        self.valve_index = 0 # start index to find valve position
        self.valve_now = 1 # position of valve

        # check if measurement noise is on or not
        if self.noise_sigma != 0:
            self.noise_on = 1
        else:
            self.noise_on = 0

        self.deadtime = deadtime # deadtime from pump to tank
        self.dist_setpoint = None # changed setpoint in case of disturbance
    
    def init_controller(self, pi0):
        """ initialise controller module for up to two tanks """

        self.pi = {}
        self.pi['0'] = pi0 # controller for tank 1
        self.ndeadtime = int(self.deadtime/ self.pi['0'].dt+1) # deadtime in simulatin steps

    def init_detect(self, detect_obj, ref_sys=None):
        """ initialise detection module """

        self.pdist_forecast = 0 # probability of detection based on forecast, only applied if global forecast is used
        self.detect = detect_obj # init detection object as part of tank class
        self.detect_mode = detect_obj.name # get detection mode
        self.detect.ref_sys = ref_sys # save reference system as part of tank class, only applies if model-based detection is used

    def init_react(self, react_obj):
        """ initialise reaction module """

        self.react = react_obj # save reaction module as part of tank class

    def init_data(self):
        """ initialise store for simulation data """
        
        data = {}
        data['tank_count'] = self.tank_count # numer of tanks
        data['pi'] = self.pi # controller
        data['t'] = [] # timesteps

        for i in range(0, self.tank_count):
            data['y' + str(i)] = [] # water height of tank i
            data['y' + str(i) + '_filt'] = [] # filtered water height of tank 1
            data['dy' + str(i)] = [] # gradient of water height of tank i
            data['u' + str(i)] = [] # controller value
            data['dist' + str(i)] = [] # state of valve (disturbance)
            data['pdist' + str(i)] = [] # "probability" of disturbance based on detection module
            data['pi' + str(i)] = [] # setpoint of controller
            data['y' + str(i) + '_mod'] = [] # water height of parallel simulation
            data['mod_error'] = [] # difference between tank and parallel simulation
            data['dy'+str(i)+'thres'] = [] # threshold fpr gradient method

            return data

    def simulate(self, t_eval, y0, valve_state0, valve_state1 = None):
        """ simulate tanksystem based on Toriccelli """
       
        start_sim = time.time()
        data = self.init_data() # initalise store for data
        
        y_mes = list(y0) # "measured" value
        y_filt = y_mes
        y = list(y0) # real value
        u = [0] * self.tank_count # controller value
        u_sys = [0] * self.tank_count # controller value after consideration of deadtime
 
        self.pdist = [0, 0] # inital value disturbance probability
        self.mod_error = []

        # loop throut timesteps
        for i in range(0, len(t_eval)-1):

            t_span = [t_eval[i], t_eval[i+1]] # start and end point for iteration

            # use pi conroller
            for i in range(0, self.tank_count):
                u[i] = self.pi[str(i)].get_u(y_filt[-1]) # berechne Stellgröße
                data['u' + str(i)].append(u[i])

            # deadtime of n timesteps
            for i in range(0, self.tank_count):
                if len(data['u'+str(i)]) > self.ndeadtime:
                    u_sys[i] = data['u'+str(i)][-self.ndeadtime]
                else:
                    u_sys[i] = 0

            # integrate equation
            if self.tank_count == 1:

                # get status of dist on entry valve
                valve1 = self.valve_fun(t_span[0], valve_state0)

                sol = siint.solve_ivp(fun = lambda t,y: eintank_eq(t, y, valve1, u_sys[0], self), t_span=t_span, y0=y, t_eval=t_span)
           
            # integrate parallel sim, if model based detection
            if 'model' in self.detect_mode:
                if self.tank_count == 1:
                    sol_mod = siint.solve_ivp(fun = lambda t,y: eintank_eq(t, y, 1, u_sys[0], self.detect.ref_sys), t_span=t_span, y0=[y_filt[-1]], t_eval=t_span)
                
                for i in range(0, self.tank_count):
                    data['y' + str(i) + '_mod'].append(sol_mod.y[i][-1])

            # add measurement noise
            for i in range(0, self.tank_count):
                y[i] = sol.y[i][-1]
                if self.noise_on == 1:
                    noise = np.random.normal(0, self.noise_sigma)
                else:
                    noise = 0
                y_mes[i] = y[i] + noise

            self.t = sol.t[-1]

            # append data
            for i in range(0, self.tank_count):
                data['y' + str(i)].append(y_mes[i])
            data['t'].append(self.t)
            data['dist0'].append(valve1)

            # filter data (moving mean over 0.2 s)
            """if len(data['y0'])>15:
                num, den = signal.butter(4, 1.5, 'low', analog=False, fs=1/self.dt)
                y_filt = [signal.filtfilt(num, den, data['y0'])][-1]"""
            if len(data['y0']) > int(0.25/self.dt):
                y_filt = [np.mean(data['y0'][-int(0.2/self.dt):])]
            else:
                y_filt = [data['y0'][-1]]

            data['y0_filt'].append(y_filt[-1])

            # determine probability of disturbance (not used)
            self.detect.detect_dist(data=data, tank=self)

            # react to disturbance based on probability
            self.react.adjust_setpoint(data, self)
            self.react.react_dist(tank=self, u=u)

            # append data
            for i in range(0, self.tank_count):
                if 'gradient' in self.detect_mode:        
                    data['dy'+str(i)].append(self.dy[i])
                    data['dy'+str(i)+'thres'].append(self.detect.dy_threshold)
                data['pdist'+str(i)].append(self.pdist[i])
                data['pi'+str(i)].append(self.pi[str(i)].y_s)

        self.data = data
        self.data['mod_error'] = self.mod_error
        self.sim_time = time.time() - start_sim

    def valve_fun(self, t, valve_state):
        """ get status of valve (disturbance) """

        for i in valve_state[self.valve_index:]:
            if t >= i[0]:
                self.valve_now = i[1]
                self.valve_index += 1
            else:
                return  self.valve_now
                
        return self.valve_now

class eintank(tank):
    """ child class for one tank system only """

    def __init__(self, A, q, dt, noise_sigma, deadtime, model_qual=1):
        super().__init__(A, q, dt, noise_sigma, deadtime, model_qual)

        self.tank_count = 1

def eintank_eq(t, y, valve, u, tank):
    """ tank equation for integration """

    # no negative water height
    if y[0] < 0:
        y[0] = 0

    if tank.model_qual == 1:
        return 1/tank.A * (u*valve - tank.q*np.sqrt(2*tank.g*y[0]))
    else:
        return 1/(tank.A) * (u*valve - tank.q*(2*tank.g*y[0])**(tank.model_qual))
