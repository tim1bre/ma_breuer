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
from Code.S03_Erkennen.detect_dummy import detect_dummy
from Code.S04_Reagieren.react_dummy import react_dummy
import warnings

def main():

    print('-- start main --')
    print('-')
    print('-')
    print('-')
    print('-')
    print('-')

    # simulate timesteps
    t0 = 0
    te = 180
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

    dist_id = 1
    dist_list = 1
    
    valve_state1 = [[0, 1]]
    valve_state1 += dist_generator('cos', 100, 130, dt, tperiod=5, amp=0.5, offset =0.5)
    
    call_all_systems(dist_id, dist_list, kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, valve_state1, model_quality)

def call_all_systems(dist_id, dist_list, kp, ki, kd, y_s, A, q, y0, dt, u_max, t_eval, noise_sigma, deadtime, valve_state1, model_quality):

    tanksys1 = eintank(A, q, dt, noise_sigma, deadtime)
    pi1 = PI(kp, ki, kd, y_s, dt, u_max)
    tanksys1.init_controller(pi1)
    detect_model1 = detect_dummy()
    tanksys1_ref = eintank(A, q, dt, 0, deadtime)
    tanksys1.init_detect(detect_model1)
    react_move_setpoint1 = react_dummy()
    tanksys1.init_react(react_move_setpoint1)
    tanksys1.dist_id = dist_id
    tanksys1.dist_list = dist_list

    tanksys1.data = tanksys1.simulate(t_eval, y0, valve_state1)
    plot_only_sim(tanksys1.data)
    plt.show()

    save_object('Exp_E0_Motivaton_Thesis', tanksys1)

if __name__ == '__main__':
    main()