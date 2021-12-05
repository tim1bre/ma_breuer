# fourth monte-carlo, evaluate learning

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
from Code.S02_Allgemein.disturbance_generator import dist_generator

from Code.S01_Experimente.Monte_Carlo.create_disturbances import get_signalpower
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

    # select if multi or single core
    server_run = 1
    if server_run == 1:
        print('mode: multi core')
    elif server_run == 0:
        print('mode: single core')
    h5_filename = datetime.now().strftime('%Y_%m_%d__%H_%M_%S') + '_Monte_Carlo_Data.hdf5'
    print('filename: '+str(h5_filename))

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
    t_dist1_0 = 100 # start first disturbance
    t_dist2_0 = 200 # start second disturbance
    dist_list_start = [t_dist1_0, t_dist2_0]

    dist1, dist1_name, dist2, dist2_name = create_disturbances_MC4(t_dist1_0, t_dist2_0, dt)

    # if server run, initialize hdf5 and fill in disturbance information
    if server_run == 1:
        with h5py.File(h5_filename, 'w') as hdf:
            G_dist = hdf.create_group("disturbances")
            G_dist1 = G_dist.create_group("dist1")
            G_dist2 = G_dist.create_group("dist2")

            for i,j in zip(dist1, dist1_name):
                G_data1 = G_dist1.create_dataset(j, shape=(len(i),len(i[0])), data=i)
                G_data1.attrs['signalpower'] = get_signalpower(i)
            
            for i,j in zip(dist2, dist2_name):
                G_data2 = G_dist2.create_dataset(j, shape=(len(i),len(i[0])), data=i)
                G_data2.attrs['signalpower'] = get_signalpower(i)

    # recombine disturbances --> 2 disturbances/ simulation
    dist_list = [ [[0,1]] + i[0] + i[1] for i in itertools.product(dist1, dist2)]
    dist_list_name = [i for i in itertools.product(dist1_name, dist2_name)]
    dist_id = [i for i in range(1,1+len(dist_list))]

    print('total disturbances: ' + str(dist_id[-1]))

    # multi processor
    if server_run == 1:

        with mp.Manager() as manager:
            pool = mp.Pool(5) # init pool
            message_queue = manager.Queue() # init queue to transfer data from worker to listener
            terminate_queue = manager.Queue() # init queue to terminate listener process

            # distribute tasks to processors
            listener_done = pool.apply_async(listener, (message_queue, terminate_queue, h5_filename))
            pool.starmap(call_all_systems, zip(dist_id, dist_list_name, dist_list, repeat(dist_list_start), repeat(kp), repeat(ki), repeat(kd), repeat(y_s), repeat(A), repeat(q), repeat(y0), repeat(dt), repeat(u_max), repeat(t_eval), repeat(noise_sigma), repeat(deadtime), repeat(model_quality), repeat(server_run), repeat(message_queue)))

            # terminate listener
            message_queue.put('terminate')

            while True:
                if not terminate_queue.empty():
                    if terminate_queue.get() == 'exit loop':
                        break
                else:
                    pass

            pool.close()
            pool.join()
    
    # single core
    else:

        sel = 1 # select disturbance from dist_list

        # simulate selected disturbance
        call_all_systems(dist_id[sel], dist_list_name[sel], dist_list[sel], dist_list_start, kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, model_quality, server_run, 0)

    print('end main: ' + datetime.now().strftime('%Y.%m.%d %H:%M,%S'))
    
    # log end time in seperate hdf5 (workaround)
    end_time = datetime.now().strftime('%Y_%m_%d__%H_%M_%S') + '_end_time.hdf5'
    with h5py.File(end_time, 'w') as hdf:
        pass
        

def listener(q, terminate_queue, h5_filename):
    """listens for messages in queue, writes in hdf5"""

    count_write = 1 # count of how many simulations have been written to hdf5

    with h5py.File(h5_filename, 'a') as hdf:
        G_exp = hdf.create_group("experiments")

        while True:
            if not q.empty():
                m = q.get()

                if isinstance(m, list):

                    [attributes, data_h5, metrices] = m
            
                    # add new experiment to hdf5
                    exp_id = len(hdf["experiments"])+1
                    new_exp = G_exp.create_group("exp"+str(exp_id))

                    # add data of experiment to hdf5
                    new_exp_data = new_exp.create_group("data")
                    for i in data_h5.keys():
                        new_exp_data.create_dataset(i, shape=(1, len(data_h5[i])), data=data_h5[i])

                    # add metrices of experiment to hdf5
                    new_exp_metrices = new_exp.create_group("metrices")
                    for i in metrices.keys():
                        if (isinstance(metrices[i], list)):
                            new_exp_metrices.create_dataset(i, shape=(1, len(metrices[i])), data=metrices[i])
                        else:
                            new_exp_metrices.create_dataset(i, data=metrices[i])

                    # add attributes to hdf5
                    for i in attributes.keys():
                        new_exp.attrs[i] = attributes[i]
                    new_exp.attrs['exp_id'] = exp_id

                    print(str(count_write) + " simulations finished")
                    count_write += 1

                # check if workers are already finished
                elif isinstance(m, str):
                    if m == 'terminate':
                        terminate_queue.put('exit loop')
                        break
    
def call_all_systems(dist_id, dist_name, valve_state1, dist_list_start, kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, model_quality, server_run, queue):
    """simualte different systems with selected parameters"""
    
    # repeat simulations to reduce influece of noise
    for repeat in range(0, 1):

        # apply disturbance to different systems
        for noise_sel in noise_sigma:
            for deadtime_sel in deadtime:

                # first object: gradient algo

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

                try:
                    tanksys1.data = tanksys1.simulate(t_eval, y0, valve_state1)
                    tanksys1 = evaluate_all(tanksys1, 95)
                    tanksys1.success = 1
                except Exception as e:
                    tanksys1.success = 0
                    print('Sys 1 failed with disturbance ' + str(dist_id))
                    print(e)

                if server_run == 1:
                    queue.put(evaluate_hdf5(tanksys1))
                
                for model_qual in model_quality:
                    
                    # second object: model algo

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

                    try:
                        tanksys2.data = tanksys2.simulate(t_eval, y0, valve_state1)
                        tanksys2 = evaluate_all(tanksys2, 95) 
                        tanksys2.success = 1
                    except Exception as e:
                        tanksys2.success = 0
                        print('Sys 2 failed with disturbance ' + str(dist_id))
                        print(e)

                    if server_run == 1:
                        queue.put(evaluate_hdf5(tanksys2))

                    # fifth object: model arima

                    # init object with selected parameters
                    tanksys5 = eintank(A, q, dt, noise_sel, deadtime_sel)
                    tanksys5.sigma = 0
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

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            tanksys5.data = tanksys5.simulate(t_eval, y0, valve_state1)
                        tanksys5 = evaluate_all(tanksys5, 95) 
                        tanksys5.success = 1
                    except Exception as e:
                        tanksys5.success = 0
                        print('Sys 5 failed with disturbance ' + str(dist_id))
                        print(e)

                    if server_run == 1:           
                        queue.put(evaluate_hdf5(tanksys5))

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

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tanksys6.data = tanksys6.simulate(t_eval, y0, valve_state1)
                    tanksys6 = evaluate_all(tanksys6, 95) 
                    tanksys6.success = 1
                except Exception as e:
                    tanksys6.success = 0
                    print('Sys 6 failed with disturbance ' + str(dist_id))
                    print(e)

                if server_run == 1:            
                    queue.put(evaluate_hdf5(tanksys6))

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

                # 8th object: pelt old

                # init object with selected parameters
                tanksys8 = eintank(A, q, dt, noise_sel, deadtime_sel)
                tanksys8.sigma = 0
                pi8 = PI(kp, ki, kd, y_s, dt, u_max)
                tanksys8.init_controller(pi8)
                detect_pelt8 = detect_pelt(5)
                tanksys8.init_detect(detect_pelt8)
                react_move_setpoint8 = react_move_setpoint()
                tanksys8.init_react(react_move_setpoint8)
                tanksys8.dist_id = dist_id
                tanksys8.dist_name = dist_name
                tanksys8.dist_list_start = dist_list_start

                try:
                    tanksys8.data = tanksys8.simulate(t_eval, y0, valve_state1)
                    tanksys8 = evaluate_all(tanksys8, 95)
                    tanksys8.success = 1
                except Exception as e:
                    tanksys8.success = 0
                    print('Sys 8 failed with disturbance ' + str(dist_id))
                    print(e)

                if server_run == 1:
                    queue.put(evaluate_hdf5(tanksys8))

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

                try:
                    tanksys9.data = tanksys9.simulate(t_eval, y0, valve_state1)
                    tanksys9 = evaluate_all(tanksys9, 95)
                    tanksys9.success = 1
                except Exception as e:
                    tanksys9.success = 0
                    print('Sys 9 failed with disturbance ' + str(dist_id))
                    print(e)

                if server_run == 1:
                    queue.put(evaluate_hdf5(tanksys9))

    print('dist_id ' + str(dist_id) +' finished')

def create_disturbances_MC4(t_dist1_0, t_dist2_0, dt):
    """ creates disturbances, returns dist1, dist2 and names """

    # create first disturbances
    dist1 = []
    dist1_name = []
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+40, dt, tperiod=5, amp=0.5, offset =0.5))
    dist1_name.append('cosinus_1_40')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+40, dt, tperiod=8, amp=0.5, offset =0.5))
    dist1_name.append('cosinus_2_40')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+40, dt, tperiod=10, amp=0.5, offset =0.5))
    dist1_name.append('cosinus_3_40')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+40, dt, tperiod=20, amp=0.5, offset =0.5))
    dist1_name.append('cosinus_4_40')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+40, dt, tperiod=5, amp=0.4, offset =0.6))
    dist1_name.append('cosinus_5_40')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+40, dt, tperiod=8, amp=0.4, offset =0.6))
    dist1_name.append('cosinus_6_40')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+40, dt, tperiod=10, amp=0.4, offset =0.6))
    dist1_name.append('cosinus_7_40')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+40, dt, tperiod=20, amp=0.4, offset =0.6))
    dist1_name.append('cosinus_8_40')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+50, dt, tperiod=5, amp=0.5, offset =0.5))
    dist1_name.append('cosinus_9_50')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+50, dt, tperiod=10, amp=0.5, offset =0.5))
    dist1_name.append('cosinus_10_50')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+50, dt, tperiod=25, amp=0.5, offset =0.5))
    dist1_name.append('cosinus_11_50')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+50, dt, tperiod=5, amp=0.4, offset =0.6))
    dist1_name.append('cosinus_12_50')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+50, dt, tperiod=10, amp=0.4, offset =0.6))
    dist1_name.append('cosinus_13_50')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_0+50, dt, tperiod=25, amp=0.4, offset =0.6))
    dist1_name.append('cosinus_14_50')

    dist2 = []
    dist2_name = []
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+40, dt, tperiod=5, amp=0.5, offset =0.5))
    dist2_name.append('cosinus_1_40')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+40, dt, tperiod=8, amp=0.5, offset =0.5))
    dist2_name.append('cosinus_2_40')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+40, dt, tperiod=10, amp=0.5, offset =0.5))
    dist2_name.append('cosinus_3_40')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+40, dt, tperiod=20, amp=0.5, offset =0.5))
    dist2_name.append('cosinus_4_40')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+40, dt, tperiod=5, amp=0.4, offset =0.6))
    dist2_name.append('cosinus_5_40')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+40, dt, tperiod=8, amp=0.4, offset =0.6))
    dist2_name.append('cosinus_6_40')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+40, dt, tperiod=10, amp=0.4, offset =0.6))
    dist2_name.append('cosinus_7_40')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+40, dt, tperiod=20, amp=0.4, offset =0.6))
    dist2_name.append('cosinus_8_40')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+50, dt, tperiod=5, amp=0.5, offset =0.5))
    dist2_name.append('cosinus_9_50')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+50, dt, tperiod=10, amp=0.5, offset =0.5))
    dist2_name.append('cosinus_10_50')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+50, dt, tperiod=25, amp=0.5, offset =0.5))
    dist2_name.append('cosinus_11_50')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+50, dt, tperiod=5, amp=0.4, offset =0.6))
    dist2_name.append('cosinus_12_50')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+50, dt, tperiod=10, amp=0.4, offset =0.6))
    dist2_name.append('cosinus_13_50')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_0+50, dt, tperiod=25, amp=0.4, offset =0.6))
    dist2_name.append('cosinus_14_50')

    return dist1, dist1_name, dist2, dist2_name

if __name__ == '__main__':
    main()