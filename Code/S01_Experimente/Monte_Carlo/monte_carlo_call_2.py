# second monte-carlo studie, evaluate performance of model-based detection

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
import traceback
import random

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

    # create disturbances
    duration_dist = 40
    t_dist1_0 = 100 # start first disturbance
    t_dist2_0 = 200 # start second disturbance
    dist_list_start = [t_dist1_0, t_dist2_0]
    dist1, dist1_name, dist2, dist2_name = create_disturbances(t_dist1_0, t_dist2_0, duration_dist, dt)

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

    # just use first dist1
    """num_dist = 5
    dist1 = dist1[:num_dist]
    dist2 = dist2[:num_dist]
    dist1_name = dist1_name[:num_dist]
    dist2_name = dist2_name[:num_dist]
    dist_list[:num_dist]
    """

    dist_list = [i + j for i,j in zip(dist1, dist2)]
    dist_list_name = [[i, j] for i,j in zip(dist1_name, dist2_name)]

    dist_id = [i for i in range(1, 1+len(dist_list))]

    print('total disturbances: ' + str(dist_id[-1]))

    # multi processor
    if server_run == 1:

        with mp.Manager() as manager:
            pool = mp.Pool() # init pool
            message_queue = manager.Queue() # init queue to transfer data from worker to listener
            terminate_queue = manager.Queue() # init queue to terminate listener process

            # distribute tasks to processors
            listener_done = pool.apply_async(listener, (message_queue, terminate_queue, h5_filename))
            pool.starmap(call_all_systems, zip(dist_id, dist_list_name, dist_list, repeat(dist_list_start), repeat(kp), repeat(ki), repeat(kd), repeat(y_s), repeat(A), repeat(q), repeat(y0), repeat(dt), repeat(u_max), repeat(t_eval), repeat(noise_sigma), repeat(deadtime), repeat(server_run), repeat(message_queue)))

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
        call_all_systems(dist_id[sel], dist_list_name[sel], dist_list[sel], dist_list_start, kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, server_run, 0)

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
    
def call_all_systems(dist_id, dist_name, valve_state1, dist_list_start, kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, server_run, queue):
    """simualte different systems with selected parameters"""
    
    parameter_deviation = [0, 0.1, 0.15, 0.25, 0.5, 0.75] # parameter deviation in 1

    # repeat simulations to reduce influece of noise
    for repeat in range(0, 8):

        # apply disturbance to different systems
        for noise_sel in noise_sigma:
            for deadtime_sel in deadtime:
                
                for para_dev in parameter_deviation:
                    
                    # second object: model algo

                    try:                      

                        # init object with selected parameters
                        tanksys2 = eintank(A, q, dt, noise_sel, deadtime_sel)
                        tanksys2.sigma = para_dev
                        pi2 = PI(kp, ki, kd, y_s, dt, u_max)
                        tanksys2.init_controller(pi2)
                        detect_model2 = detect_model_algo('full')

                        # get parameter forr parallel simulation
                        
                        A_new = random.choice([ (1+para_dev)*A, (1-para_dev)*A ])
                        q_new = random.choice([ (1+para_dev)*q, (1-para_dev)*q ])
                        exponent = random.choice([ (1+para_dev)*0.5, (1-para_dev)*0.5 ])

                        tanksys2_ref = eintank(A_new, q_new, dt, 0, deadtime_sel, exponent)
                        tanksys2.init_detect(detect_model2, ref_sys=tanksys2_ref)
                        react_dummy2 = react_dummy()
                        tanksys2.init_react(react_dummy2)
                        tanksys2.dist_id = dist_id
                        tanksys2.dist_name = dist_name
                        tanksys2.dist_list_start = dist_list_start
                    except Exception as e:
                        tanksys2.success = 0
                        print('could not fit sigma ' + str(para_dev))
                        traceback.print_exc()

                    try:
                        tanksys2.data = tanksys2.simulate(t_eval, y0, valve_state1)
                        tanksys2 = evaluate_all(tanksys2, 95)
                        tanksys2.success = 1

                        if server_run == 1:
                            queue.put(evaluate_hdf5(tanksys2))
                    except Exception as e:
                        tanksys2.success = 0
                        print('Sys 2 failed with disturbance ' + str(dist_id))
                        traceback.print_exc()

    print('dist_id ' + str(dist_id) +' finished')

if __name__ == '__main__':
    main()