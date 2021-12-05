import numpy as np
from Code.S02_Allgemein.disturbance_generator import dist_generator, chirp_generator, sin_var_amp_generator, random_walk_dist, noisy_cos, moving_mean, triangle_generator
import dill
import os
from os import listdir
from os.path import isfile, join

def create_disturbances(t_dist0, duration_dist, dt):
    """ creates disturbances, returns dist, dist2 and names """

    # end of individual disturbances
    t_diste = t_dist0 + duration_dist

    # create first disturbances
    dist = []
    dist_name = []

    # cos disturbances 
    cos_num = 1
    for i in [10, 16, 20, 25, 40, 50]:
        for amp in np.linspace(0.15, 0.3, 14): 
            dist.append(dist_generator('cos', t_dist0, t_diste, dt, tperiod=i, amp=amp, offset=1-amp))
            dist_name.append('cosinus_' + str(cos_num))
            cos_num = cos_num + 1

    
    # rectangle disturbances
    rectangle_num = 1
    for i in [8, 10]:
        for amp in np.linspace(0.1, 0.25, 14): 
            dist.append(dist_generator('sq sin', t_dist0, t_diste, dt, tperiod=i, amp=amp, offset=1-amp))
            dist_name.append('rectangle_'+str(rectangle_num))
            rectangle_num = rectangle_num + 1

    # chirp disturbances
    chirp_num = 1
    for amp in np.linspace(0.175, 0.25, 8):
        for f0 in [0.02, 0.05, 0.1]:
            for f1_multi in [1.5, 2]:
                dist.append(chirp_generator(t_dist0, t_diste, dt, amp=amp, offset=1-amp, f0=0.1, t1=t_diste, f1=f0*f1_multi))
                dist_name.append('chirp_'+str(chirp_num))
                chirp_num = chirp_num +1

    # dreick
    triang_num = 1
    for tperiod in [20, 25]:
        for amp in [0.15, 0.175, 0.2, 0.25, 0.25, 0.275, 0.3]:
            dist.append(triangle_generator(t_dist0, t_diste, dt, tperiod, amp))
            dist_name.append('triag_'+str(triang_num))
            triang_num = triang_num + 1
    for tperiod in [40, 50]:
        for amp in [0.10, 0.125, 0.15, 0.175, 0.2]:
            dist.append(triangle_generator(t_dist0, t_diste, dt, tperiod, amp))
            dist_name.append('triag_'+str(triang_num))
            triang_num = triang_num + 1
    
    # var amp
    varamp_num = 1
    for i in [10, 16, 20, 25, 40, 50]:
        for amp in [0.25, 0.275, 0.3]:
            for var in [-0.075, -0.05, -0.025]:
                ampe = amp + var
                dist.append(sin_var_amp_generator(t_dist0, t_diste, dt, amp0=amp, ampe=ampe, off0=1-amp, offe=1-ampe, tperiod=i))
                dist_name.append('varamp_'+str(varamp_num))
                varamp_num = varamp_num + 1
        for amp in [0.175, 0.2, 0.225]:
            for var in [0.075, 0.05, 0.025]:
                ampe = amp + var
                dist.append(sin_var_amp_generator(t_dist0, t_diste, dt, amp0=amp, ampe=ampe, off0=1-amp, offe=1-ampe, tperiod=i))
                dist_name.append('varamp_'+str(varamp_num))
                varamp_num = varamp_num + 1
    
    """
    amp= 0.35
    dist.append(dist_generator('cos', t_dist0, t_diste, dt, tperiod=40, amp=0.2, offset =1-amp))
    dist_name.append('cosinus_1')
    dist.append(dist_generator('cos', t_dist0, t_diste, dt, tperiod=50, amp=0.3, offset =0.7))
    dist_name.append('cosinus_2')
    dist.append(dist_generator('cos', t_dist0, t_diste, dt, tperiod=50, amp=0.35, offset =0.65))
    dist_name.append('cosinus_3')
    dist.append(dist_generator('cos', t_dist0, t_diste, dt, tperiod=40, amp=0.25, offset =0.75))
    dist_name.append('cosinus_4')
    dist.append(dist_generator('cos', t_dist0, t_diste, dt, tperiod=40, amp=0.3, offset =0.7))
    dist_name.append('cosinus_5')
    """

    # add preselected random disturbances
    randdist_path =  os.path.dirname(os.path.realpath(__file__)) + '/random_dist/'
    f = listdir(randdist_path)
    pklfiles = [f for f in listdir(randdist_path) if (isfile(join(randdist_path, f)) and 'random_dist' in f and '.pkl' in f)]
    
    rand_num = 1
    for disturbance_file in pklfiles:
        with open(randdist_path+disturbance_file, 'rb') as f:
            obj = dill.load(f)
            dist.append(obj['dist'])
            dist_name.append("random_"+str(rand_num))
            rand_num = rand_num + 1
                
    return dist, dist_name
    
def get_signalpower(data):
    """ calculate signalpower of 1 - valve_state """

    t = np.transpose(data)[0] # timestamps
    dt = np.diff(t) # delta t
    y = 1 - np.transpose(data)[1] # 1 - valve_state

    # calculate signal energy
    E = 0
    for delta_t, signal in zip(dt, y[:-1]):
        E += delta_t*(signal)**2

    # calculate signal power
    P = E /(t[-1] - t[0])
    
    return P