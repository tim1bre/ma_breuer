from multiprocessing import Pool
import numpy as np
import scipy.integrate as siint
import matplotlib.pyplot as plt
import math

from Code.S02_Allgemein.disturbance_generator import dist_generator, chirp_generator
from Code.S02_Allgemein.tank import eintank
from Code.S02_Allgemein.pi import PI
from Code.S02_Allgemein.save_object import save_object
from Code.S02_Allgemein.plot_results import plot_results_full, plot_only_sim
from Code.S03_Erkennen.detect_model_arima_multimodel import detect_model_arima_multimodel
from Code.S04_Reagieren.react_move_setpoint import react_move_setpoint
from Code.S06_Quantifizieren.evaluate import eval_resloss, t_recover, eval_y_exceed

import warnings

def main():

    print('-')
    print('-')
    print('-')
    print('-')
    print('-')
    print('-- start main --')

    # simulate timesteps
    t0 = 0
    te = 850
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

    noise_sigma = 0.05
    deadtime = 0.5
    model_quality = 1

    valve_state1 = [[0, 1]]

    dist1 = []

    dist1 = dist_generator('cos', 100, 160, dt, tperiod=10, amp=0.5, offset =0.5)
    dist1 += dist_generator('cos', 200, 260, dt, tperiod=5, amp=0.4, offset =0.6)
    dist1 += dist_generator('cos', 300, 360, dt, tperiod=10, amp=0.5, offset =0.5)
    dist1 += dist_generator('cos', 400, 460, dt, tperiod=5, amp=0.4, offset =0.6)
    dist1 += dist_generator('cos', 500, 560, dt, tperiod=10, amp=0.5, offset =0.5)
    dist1 += dist_generator('cos', 600, 660, dt, tperiod=5, amp=0.4, offset =0.6)
    dist1 += dist_generator('cos', 700, 760, dt, tperiod=10, amp=0.5, offset =0.5)
    valve_state1 = dist1
    
    dist_id = 1
    dist_list = 1

    call_all_systems(dist_id, dist_list, kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, valve_state1, model_quality)

    print('-- end main --')
    
def call_all_systems(dist_id, dist_list, kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, valve_state1, model_quality):

    tanksys1 = eintank(A, q, dt, noise_sigma, deadtime)
    pi1 = PI(kp, ki, kd, y_s, dt, u_max)
    tanksys1.init_controller(pi1)
    detect_model1 = detect_model_arima_multimodel()
    tanksys1_ref = eintank(A, q, dt, 0, deadtime, model_quality)
    tanksys1.init_detect(detect_model1, ref_sys=tanksys1_ref)
    react_move_setpoint1 = react_move_setpoint(limit_reserve=0)
    tanksys1.init_react(react_move_setpoint1)
    tanksys1.dist_id = dist_id
    tanksys1.dist_list = dist_list

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tanksys1.data = tanksys1.simulate(t_eval, y0, valve_state1)
    save_object('implementierte_methoden_multimodel', tanksys1)

if __name__ == '__main__':
    main()