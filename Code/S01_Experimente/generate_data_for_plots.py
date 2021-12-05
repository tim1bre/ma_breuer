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
    te = 300
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
    t_dist1_0 = 100 # start first disturbance
    t_dist2_0 = 200 # start second disturbance
    dist_list_start = [t_dist1_0, t_dist2_0]
    dist1, dist1_name, dist2, dist2_name = create_disturbances(t_dist1_0, t_dist2_0, duration_dist, dt)

    # recombine disturbances --> 2 disturbances/ simulation
    dist_list = [ [[0,1]] + i[0] + i[1] for i in itertools.product(dist1, dist2)]
    dist_list_name = [i for i in itertools.product(dist1_name, dist2_name)]
    dist_id = [i for i in range(1,1+len(dist_list))]

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

                """# first object: gradient algo

                # init object with selected parameters
                tanksys1 = eintank(A, q, dt, noise_sel, deadtime_sel)
                tanksys1.sigma = 0
                pi1 = PI(kp, ki, kd, y_s, dt, u_max)
                tanksys1.init_controller(pi1)
                detect_grad1 = detect_gradient(0.99, 70)
                tanksys1.init_detect(detect_grad1)
                react_move_setpoint1 = react_move_setpoint()
                tanksys1.init_react(react_move_setpoint1)
                tanksys1.dist_id = dist_id
                tanksys1.dist_name = dist_name
                tanksys1.dist_list_start = dist_list_start

                tanksys1.data = tanksys1.simulate(t_eval, y0, valve_state1)
                save_object('implementierte_methoden_gradient', tanksys1)"""
                
                
                for model_qual in model_quality:
                    
                    """# second object: model algo

                    # init object with selected parameters
                    tanksys2 = eintank(A, q, dt, noise_sel, deadtime_sel)
                    tanksys2.sigma = 0
                    pi2 = PI(kp, ki, kd, y_s, dt, u_max)
                    tanksys2.init_controller(pi2)
                    detect_model2 = detect_model_algo('full')
                    tanksys2_ref = eintank(A, q, dt, 0, deadtime_sel, model_qual)
                    tanksys2.init_detect(detect_model2, ref_sys=tanksys2_ref)
                    react_move_setpoint2 = react_move_setpoint()
                    tanksys2.init_react(react_move_setpoint2)
                    tanksys2.dist_id = dist_id
                    tanksys2.dist_name = dist_name
                    tanksys2.dist_list_start = dist_list_start

                    
                    tanksys2.data = tanksys2.simulate(t_eval, y0, valve_state1)
                    save_object('implentierte_methoden_modellbasier', tanksys2)"""
                    
                    """
                    # third object: model ar

                    # init object with selected parameters
                    tanksys3 = eintank(A, q, dt, noise_sel, deadtime_sel)
                    tanksys3.sigma = 0
                    pi3 = PI(kp, ki, kd, y_s, dt, u_max)
                    tanksys3.init_controller(pi3)
                    detect_model3 = detect_model_ar(global_forecast=0)
                    tanksys3_ref = eintank(A, q, dt, 0, deadtime_sel, model_qual)
                    tanksys3.init_detect(detect_model3, ref_sys=tanksys3_ref)
                    react_move_setpoint3 = react_move_setpoint()
                    tanksys3.init_react(react_move_setpoint3)
                    tanksys3.dist_id = dist_id
                    tanksys3.dist_name = dist_name
                    tanksys3.dist_list_start = dist_list_start

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            tanksys3.data = tanksys3.simulate(t_eval, y0, valve_state1)
                        tanksys3 = evaluate_all(tanksys3, 95) 
                        tanksys3.success = 1
                    except Exception as e:
                        tanksys3.success = 0
                        print('Sys 3 failed with disturbance ' + str(dist_id))
                        print(e)

                    if server_run == 1:  
                        queue.put(evaluate_hdf5(tanksys3))"""

                    """ 
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
                    save_object('implentierte_methoden_arima', tanksys5)"""
                        
                """
                # fourth object: dtw algo

                # init object with selected parameters
                tanksys4 = eintank(A, q, dt, noise_sel, deadtime_sel)
                tanksys4.sigma = 0
                pi4 = PI(kp, ki, kd, y_s, dt, u_max)
                tanksys4.init_controller(pi4)
                detect_model4 = detect_dtw_algo()
                tanksys4.init_detect(detect_model4)
                react_move_setpoint4 = react_move_setpoint()
                tanksys4.init_react(react_move_setpoint4)
                tanksys4.dist_id = dist_id
                tanksys4.dist_name = dist_name
                tanksys4.dist_list_start = dist_list_start

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tanksys4.data = tanksys4.simulate(t_eval, y0, valve_state1)
                    tanksys4 = evaluate_all(tanksys4, 95) 
                    tanksys4.success = 1
                except Exception as e:
                    tanksys4.success = 0
                    print('Sys 4 failed with disturbance ' + str(dist_id))
                    print(e)

                if server_run == 1:            
                    queue.put(evaluate_hdf5(tanksys4))
                """
                # sixth object: dtw algo mean free

                # init object with selected parameters
                tanksys6 = eintank(A, q, dt, noise_sel, deadtime_sel)
                tanksys6.sigma = 0
                pi6 = PI(kp, ki, kd, y_s, dt, u_max)
                tanksys6.init_controller(pi6)
                detect_dtw_grad1 = detect_dtw_mean_algo()
                tanksys6.init_detect(detect_dtw_grad1)
                react_move_setpoint6 = react_move_setpoint()
                tanksys6.init_react(react_move_setpoint6)
                tanksys6.dist_id = dist_id
                tanksys6.dist_name = dist_name
                tanksys6.dist_list_start = dist_list_start

                
                tanksys6.data = tanksys6.simulate(t_eval, y0, valve_state1)
                save_object('implentierte_methoden_dtw', tanksys6)
                
                """
                # seventh object: dummy - no methods used

                # init object with selected parameters
                tanksys7 = eintank(A, q, dt, noise_sel, deadtime_sel)
                tanksys7.sigma = 0
                pi7 = PI(kp, ki, kd, y_s, dt, u_max)
                tanksys7.init_controller(pi7)
                detect_dummy7 = detect_dummy()
                tanksys7.init_detect(detect_dummy7)
                react_dummy7 = react_dummy()
                tanksys7.init_react(react_dummy7)
                tanksys7.dist_id = dist_id
                tanksys7.dist_name = dist_name
                tanksys7.dist_list_start = dist_list_start

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tanksys7.data = tanksys7.simulate(t_eval, y0, valve_state1)
                    tanksys7 = evaluate_all(tanksys7, 95) 
                    tanksys7.success = 1
                except Exception as e:
                    tanksys7.success = 0
                    print('Sys 7 failed with disturbance ' + str(dist_id))
                    print(e)

                if server_run == 1:            
                    queue.put(evaluate_hdf5(tanksys7))
                """

                """# 8th object: pelt old

                # init object with selected parameters
                tanksys8 = eintank(A, q, dt, noise_sel, deadtime_sel)
                tanksys8.sigma = 0
                pi8 = PI(kp, ki, kd, y_s, dt, u_max)
                tanksys8.init_controller(pi8)
                detect_pelt8 = detect_pelt(10)
                tanksys8.init_detect(detect_pelt8)
                react_move_setpoint8 = react_move_setpoint()
                tanksys8.init_react(react_move_setpoint8)
                tanksys8.dist_id = dist_id
                tanksys8.dist_name = dist_name
                tanksys8.dist_list_start = dist_list_start

                tanksys8.data = tanksys8.simulate(t_eval, y0, valve_state1)
                save_object('implentierte_methoden_PELT', tanksys8)
                

                # 9th object: pelt new with classification via min/max

                # init object with selected parameters
                tanksys9 = eintank(A, q, dt, noise_sel, deadtime_sel)
                tanksys9.sigma = 0
                pi9 = PI(kp, ki, kd, y_s, dt, u_max)
                tanksys9.init_controller(pi9)
                detect_pelt9 = detect_pelt_minmax(5)
                tanksys9.init_detect(detect_pelt9)
                react_move_setpoint9 = react_move_setpoint()
                tanksys9.init_react(react_move_setpoint9)
                tanksys9.dist_id = dist_id
                tanksys9.dist_name = dist_name
                tanksys9.dist_list_start = dist_list_start

                tanksys9.data = tanksys9.simulate(t_eval, y0, valve_state1)
                save_object('implentierte_methoden_PELTmitKl', tanksys9)
                """
                   
    print('dist_id ' + str(dist_id) +' finished')

if __name__ == '__main__':
    main()