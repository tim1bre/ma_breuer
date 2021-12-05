# monte-carlo simulation paper

import multiprocessing as mp
import numpy as np
from itertools import repeat
import h5py
from datetime import datetime
import matplotlib.pyplot as plt 
import warnings
import time

import warnings
import numpy as np
from Code.S02_Allgemein.tank import eintank
from Code.S02_Allgemein.pi import PI
from Code.S03_Erkennen.detect_dummy import detect_dummy
from Code.S03_Erkennen.detect_model_algo import detect_model_algo
from Code.S03_Erkennen.detect_dtw_mean_algo import detect_dtw_mean_algo
from Code.S03_Erkennen.detect_model_arima import detect_model_arima
from Code.S03_Erkennen.detect_gradient import detect_gradient
from Code.S03_Erkennen.detect_pelt_minmax import detect_pelt_minmax
from Code.S04_Reagieren.react_dummy import react_dummy
from Code.S04_Reagieren.react_move_setpoint import react_move_setpoint
from Code.S01_Experimente.Paper.create_disturbances_paper import create_disturbances, get_signalpower
from Code.S06_Quantifizieren.evaluate_paper import metrices_to_hdf5, evaluate_metrices_paper
from Code.S06_Quantifizieren.process_hdf5_paper import hdf5_to_df

# for debugging
from Code.S02_Allgemein.plot_results import plot_only_sim 
from Code.S02_Allgemein.save_object import save_object

def main():
    print('-----')
    print('start main: ' + datetime.now().strftime('%Y.%m.%d %H:%M,%S'))

    # select if multi or single core
    server_run = 1
    if server_run == 1:
        print('mode: multi core')
    elif server_run == 0:
        print('mode: single core')
    h5_filename = datetime.now().strftime('%Y_%m_%d__%H_%M_%S') + '_Monte_Carlo_Data.hdf5'

    # simulation parameters
    t0 = 0
    te = 900
    dt = 0.05
    t_eval = np.linspace(t0, te, int((te-t0)/dt+1))

    # tank and controller parameters
    y0 = tuple([0]) # inital values filling level in cm
    A = 0.0254 # m**2 cross section area tank
    q = np.pi*(0.014/2)**2 # m**2 cross section area pipe
    y_s = 0.2 # m set point
    u_max = 0.0005 # maximale Stellgröße cm**3/s
    kp = 0.05
    ki = 0.01
    kd = 0

    noise_sigma = 0.0005
    deadtime = 0.2 # in s
    model_quality = 1 # model quality for modelbased detection

    # create disturbances
    duration_dist = 400 # in s
    t_dist1_0 = 100 # start first disturbance in s
    dist1, dist1_name = create_disturbances(t_dist1_0, duration_dist, dt)

    # if server run, initialize hdf5 and fill in disturbance information
    if server_run == 1:
        with h5py.File(h5_filename, 'w') as hdf:
            G_dist = hdf.create_group("disturbances")
            G_dist1 = G_dist.create_group("dist1")

            for i,j in zip(dist1, dist1_name):
                G_data1 = G_dist1.create_dataset(j, shape=(len(i),len(i[0])), data=i)
                G_data1.attrs['signalpower'] = get_signalpower(i)
            
    # disturbances
    dist_list = dist1
    dist_list_name = dist1_name
    dist_id = [i for i in range(1,1+len(dist_list))]

    print('total disturbances: ' + str(dist_id[-1]))

    # multi processor
    if server_run == 1:

        with mp.Manager() as manager:
            pool = mp.Pool() # init pool all processor cores
            message_queue = manager.Queue() # init queue to transfer data from worker to listener
            terminate_queue = manager.Queue() # init queue to terminate listener process

            # distribute tasks to processors
            listener_done = pool.apply_async(listener, (message_queue, terminate_queue, h5_filename))
            pool.starmap(call_all_systems, zip(dist_id, dist_list_name, dist_list, repeat(kp), repeat(ki), repeat(kd), repeat(y_s), repeat(A), repeat(q), repeat(y0), repeat(dt), repeat(u_max), repeat(t_eval), repeat(noise_sigma), repeat(deadtime), repeat(model_quality), repeat(server_run), repeat(message_queue)))

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
    
    # single core if not server
    else:

        sel = 0 # select disturbance from dist_list

        # simulate selected disturbance
        call_all_systems(dist_id[sel], dist_list_name[sel], dist_list[sel], kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, model_quality, server_run, 0)

    hdf5_to_df(h5_filename)
    print('end main: ' + datetime.now().strftime('%Y.%m.%d %H:%M,%S'))
            
def listener(q, terminate_queue, h5_filename):
    """listens for messages in queue, writes in hdf5"""

    count_write = 1 # count of how many simulations have been written to hdf5

    with h5py.File(h5_filename, 'a') as hdf:
        G_exp = hdf.create_group("experiments")

        while True:
            if not q.empty():
                m = q.get()

                if isinstance(m, list): # wenn Liste dann neue Simulationsergebnisse

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

                    #print(str(count_write) + " simulations finished")
                    count_write += 1

                # check if workers are already finished
                elif isinstance(m, str):
                    if m == 'terminate':
                        terminate_queue.put('exit loop')
                        break
    
def call_all_systems(dist_id, dist_name, valve_state1, kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, model_quality, server_run, queue):
    """simualte different systems with selected parameters"""

    dist_dec_model = 0.0005
    dist_dec_dtw = 0.000175
    dist_dec_pelt = 0.00025
    dist_dec_grad = 0.00012
    
    # repeat simulations to reduce influece of noise
    for repeat in range(0, 1):

        # method with model-based detection
        tanksys1 = eintank(A, q, dt, noise_sigma, deadtime)
        pi1 = PI(kp, ki, kd, y_s, dt, u_max)
        tanksys1.init_controller(pi1)
        detect1 = detect_model_algo('full')
        tanksys1_ref = eintank(A, q, dt, 0, deadtime, model_quality)
        tanksys1.init_detect(detect1, ref_sys=tanksys1_ref)
        react1 = react_move_setpoint(dist_dec_model)
        tanksys1.init_react(react1)
        tanksys1.dist_id = dist_id
        tanksys1.dist_name = dist_name

        try:
            tanksys1.simulate(t_eval, y0, valve_state1)
            #print("model-based: " + str(tanksys1.sim_time))
        except Exception as e:
            print('model-based failed')
            print(e)

        if server_run == 1: # if server write results in queue
            queue.put(metrices_to_hdf5(tanksys1))
        else:
            save_object("test_obj", tanksys1)
            print("fertig!")

        # detection based on dtw
        tanksys2 = eintank(A, q, dt, noise_sigma, deadtime)
        pi2 = PI(kp, ki, kd, y_s, dt, u_max)
        tanksys2.init_controller(pi2)
        detect2 = detect_dtw_mean_algo(3, 10)
        tanksys2.init_detect(detect2)
        react2 = react_move_setpoint(dist_dec_dtw)
        tanksys2.init_react(react2)
        tanksys2.dist_id = dist_id
        tanksys2.dist_name = dist_name
        try:
            tanksys2.simulate(t_eval, y0, valve_state1)
            #print("dtw: " +  str(tanksys2.sim_time))
        except Exception as e:
            print('dtw failed')
            print(e)

        if server_run == 1: # if server write results in queue
            queue.put(metrices_to_hdf5(tanksys2))
        else:
            plot_only_sim(tanksys2.data)

        # pelt-based detection with min/max classification
        tanksys3 = eintank(A, q, dt, noise_sigma, deadtime)
        pi3 = PI(kp, ki, kd, y_s, dt, u_max)
        tanksys3.init_controller(pi3)
        detect_pelt3 = detect_pelt_minmax(5)
        tanksys3.init_detect(detect_pelt3)
        react_move_setpoint3 = react_move_setpoint(dist_dec_pelt)
        tanksys3.init_react(react_move_setpoint3)
        tanksys3.dist_id = dist_id
        tanksys3.dist_name = dist_name
        try:
            tanksys3.simulate(t_eval, y0, valve_state1)
            #print("pelt: " +  str(tanksys3.sim_time))
        except Exception as e:
            print('pelt failed')
            print(e)
        if server_run == 1: # if server write results in queue
            queue.put(metrices_to_hdf5(tanksys3))

        # model-based detection with arima forecast
        tanksys4 = eintank(A, q, dt, noise_sigma, deadtime)
        pi4 = PI(kp, ki, kd, y_s, dt, u_max)
        tanksys4.init_controller(pi4)
        detect_model4 = detect_model_arima(global_forecast=0, downsample_local=5)
        tanksys4_ref = eintank(A, q, dt, 0, deadtime, model_quality)
        tanksys4.init_detect(detect_model4, ref_sys=tanksys4_ref)
        react_move_setpoint5 = react_move_setpoint(dist_dec_model)
        tanksys4.init_react(react_move_setpoint5)
        tanksys4.dist_id = dist_id
        tanksys4.dist_name = dist_name
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tanksys4.simulate(t_eval, y0, valve_state1)
                #print("arima: " +  str(tanksys4.sim_time))
        except Exception as e:
            print('arima failed')
            print(e)
        except:
            pass

        if server_run == 1: # if server write results in queue
            queue.put(metrices_to_hdf5(tanksys4))
        else:
            plot_only_sim(tanksys4.data)

        # gradient-based detection
        tanksys5 = eintank(A, q, dt, noise_sigma, deadtime)
        pi5 = PI(kp, ki, kd, y_s, dt, u_max)
        tanksys5.init_controller(pi5)
        detect_grad5 = detect_gradient(factor_y_min=0.97, pdist_dec=0.025)
        tanksys5.init_detect(detect_grad5)
        react_move_setpoint5 = react_move_setpoint(dist_dec_grad)
        tanksys5.init_react(react_move_setpoint5)
        tanksys5.dist_id = dist_id
        tanksys5.dist_name = dist_name
        try:
            tanksys5.simulate(t_eval, y0, valve_state1)
            #print("gradient: " +  str(tanksys5.sim_time))
        except Exception as e:
            print('gradient failed')
            print(e)

        if server_run == 1: # if server write results in queue
            queue.put(metrices_to_hdf5(tanksys5))
        else:
            plot_only_sim(tanksys5.data)

        # dummy
        tanksys6 = eintank(A, q, dt, noise_sigma, deadtime)
        pi6 = PI(kp, ki, kd, y_s, dt, u_max)
        tanksys6.init_controller(pi6)
        detect_dummy6 = detect_dummy()
        tanksys6.init_detect(detect_dummy6)
        react_dummy6 = react_dummy()
        tanksys6.init_react(react_dummy6)
        tanksys6.dist_id = dist_id
        tanksys6.dist_name = dist_name

        try:
            tanksys6.simulate(t_eval, y0, valve_state1)
            #print("dummy: " +  str(tanksys6.sim_time))
        except Exception as e:
            print('dummy failed')
            print(e)

        if server_run == 1: # if server write results in queue
            queue.put(metrices_to_hdf5(tanksys6))

    print('dist_id ' + str(dist_id) +' finished')

if __name__ == '__main__':
    main()