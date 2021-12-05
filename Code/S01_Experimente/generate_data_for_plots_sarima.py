# first monte-carlo studie, evaluate performance of methods

import multiprocessing as mp
import numpy as np
import scipy.integrate as siint
import matplotlib.pyplot as plt
import math
import itertools
from itertools import repeat
import h5py
import time # not needed
from datetime import datetime

from Code.S02_Allgemein.save_object import save_object
from Code.S01_Experimente.Monte_Carlo.create_disturbances import create_disturbances, get_signalpower
from Code.S02_Allgemein.tank import eintank
from Code.S02_Allgemein.pi import PI
from Code.S02_Allgemein.save_object import save_object
from Code.S02_Allgemein.plot_results import plot_results_full
from Code.S03_Erkennen.detect_dummy import detect_dummy
from Code.S03_Erkennen.detect_gradient import detect_gradient
from Code.S03_Erkennen.detect_model_algo import detect_model_algo
from Code.S03_Erkennen.detect_model_ar_global import detect_model_ar
from Code.S03_Erkennen.detect_model_arima import detect_model_arima
from Code.S03_Erkennen.detect_dtw_mean_algo import detect_dtw_mean_algo
from Code.S03_Erkennen.detect_pelt import detect_pelt
from Code.S03_Erkennen.detect_pelt_minmax import detect_pelt_minmax
from Code.S04_Reagieren.react_move_setpoint import react_move_setpoint
from Code.S04_Reagieren.react_dummy import react_dummy
from Code.S06_Quantifizieren.evaluate_all_metrices import evaluate_all, evaluate_hdf5
from Code.S02_Allgemein.disturbance_generator import dist_generator

import warnings

def main():

    print('-')
    print('-')
    print('-')
    print('-')
    print('-')
    print('start main: ' + datetime.now().strftime('%Y.%m.%d %H:%M,%S'))

    # get simulation timesteps
    t0 = 0
    te = 600
    dt = 0.01
    t = [t0, te]
    t_eval = np.linspace(t0, te, int((te-t0)/dt+1)) 

    # tank and controller parameters
    y0 = tuple([0]) # inital values
    A = 254 # cm**2 cross section area tank
    q = np.pi*(1.4/2)**2 # cm**2 cross section area pipe
    y_s = 20 # cm set point
    u_max = 150
    kp = 189.743590
    ki = 15
    kd = 0

    noise_sigma = [0.05]
    deadtime = [0.5]
    model_quality = [1]

    # create disturbances
    duration_dist = 40
    dist_start_intervall = 100 # start first disturbance

    num_dist = 10 # number of disturbances

    dist_list_start = []
    dist_list = []


    dist_list_start = [100, 200, 300, 400, 500, 600]
    
    dist_list.append([0, 1])
    dist_list = dist_list + dist_generator('cos', 100, 140, dt, tperiod=8, amp=0.5, offset =0.5)
    dist_list = dist_list + dist_generator('cos', 200, 240, dt, tperiod=8, amp=0.5, offset =0.5)
    dist_list = dist_list + dist_generator('cos', 300, 340, dt, tperiod=8, amp=0.5, offset =0.5)
    dist_list = dist_list + dist_generator('cos', 400, 440, dt, tperiod=8, amp=0.5, offset =0.5)
    dist_list = dist_list + dist_generator('cos', 500, 540, dt, tperiod=8, amp=0.5, offset =0.5)
    dist_list = dist_list + dist_generator('cos', 600, 640, dt, tperiod=8, amp=0.5, offset =0.5)

    # recombine disturbances --> 2 disturbances/ simulation
    dist_list = [dist_list]
    dist_list_name = ['cos_1_40__' + str(num_dist)]
    dist_id = [1]

    sel = 0 # select disturbance from dist_list

    # simulate selected disturbance
    call_all_systems(dist_id[sel], dist_list_name[sel], dist_list[sel], dist_list_start, kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, model_quality)
    
def call_all_systems(dist_id, dist_name, valve_state1, dist_list_start, kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, model_quality):
    """simualte different systems with selected parameters"""
    
    # repeat simulations to reduce influece of noise
    for repeat in range(0, 1):

        # apply disturbance to different systems
        for noise_sel in noise_sigma:
            for deadtime_sel in deadtime:
                for model_qual in model_quality:

                    # fifth object: model arima

                    # init object with selected parameters
                    tanksys5 = eintank(A, q, dt, noise_sel, deadtime_sel)
                    tanksys5.sigma = 0
                    #kp = 100
                    pi5 = PI(kp, ki, kd, y_s, dt, u_max)
                    tanksys5.init_controller(pi5)
                    detect_model5 = detect_model_arima(global_forecast=0)
                    tanksys5_ref = eintank(A, q, dt, 0, deadtime_sel, model_qual)
                    tanksys5.init_detect(detect_model5, ref_sys=tanksys5_ref)
                    react_move_setpoint5 = react_move_setpoint()
                    tanksys5.init_react(react_move_setpoint5)
                    tanksys5.dist_id = dist_id
                    tanksys5.dist_name = dist_name
                    tanksys5.dist_list_start = dist_list_start

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tanksys5.data = tanksys5.simulate(t_eval, y0, valve_state1)
                    save_object('sarima_vergleich_ref_arima', tanksys5)

                    # sixth object: model sarima

                    # init object with selected parameters
                    tanksys6 = eintank(A, q, dt, noise_sel, deadtime_sel)
                    tanksys6.sigma = 0
                    pi6 = PI(kp, ki, kd, y_s, dt, u_max)
                    tanksys6.init_controller(pi6)
                    detect_model6 = detect_model_arima(global_forecast=1)
                    tanksys6_ref = eintank(A, q, dt, 0, deadtime_sel, model_qual)
                    tanksys6.init_detect(detect_model6, ref_sys=tanksys6_ref)
                    react_move_setpoint6 = react_move_setpoint()
                    tanksys6.init_react(react_move_setpoint6)
                    tanksys6.dist_id = dist_id
                    tanksys6.dist_name = dist_name
                    tanksys6.dist_list_start = dist_list_start

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tanksys6.data = tanksys6.simulate(t_eval, y0, valve_state1)
                    save_object('sarima_vergleich_sarima', tanksys6)
                        

    print('dist_id ' + str(dist_id) +' finished')

if __name__ == '__main__':
    main()