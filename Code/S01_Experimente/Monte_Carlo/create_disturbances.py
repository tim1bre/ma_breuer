import numpy as np
from Code.S02_Allgemein.disturbance_generator import dist_generator, chirp_generator, sin_var_amp_generator, random_walk_dist, noisy_cos, moving_mean
import dill
import os
from os import listdir
from os.path import isfile, join

def create_disturbances(t_dist1_0, t_dist2_0, duration_dist, dt):
    """ creates disturbances, returns dist1, dist2 and names """

    # end of individual disturbances
    t_dist1_e = t_dist1_0 + duration_dist
    t_dist2_e = t_dist2_0 + duration_dist

    # create first disturbances
    dist1 = []
    dist1_name = []

    print('t_period geändert')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_e, dt, tperiod=5, amp=0.5, offset =0.5))
    dist1_name.append('cosinus_1')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_e, dt, tperiod=8, amp=0.5, offset =0.5))
    dist1_name.append('cosinus_2')
    dist1.append(dist_generator('cos', t_dist1_0, t_dist1_e, dt, tperiod=10, amp=0.5, offset =0.5))
    dist1_name.append('cosinus_3')
    
    dist1.append(noisy_cos(t_dist1_0, t_dist1_e, dt, amp=0.5, offset=0.5, tperiod=5, noise=0.025))
    dist1_name.append('noisycos_1')
    dist1.append(noisy_cos(t_dist1_0, t_dist1_e, dt, amp=0.5, offset=0.5, tperiod=8, noise=0.025))
    dist1_name.append('noisycos_2')
    dist1.append(noisy_cos(t_dist1_0, t_dist1_e, dt, amp=0.5, offset=0.5, tperiod=10, noise=0.025))
    dist1_name.append('noisycos_3')

    dist1.append(dist_generator('sq sin', t_dist1_0, t_dist1_e, dt, tperiod=5, amp=0.4, offset =0.6))
    dist1_name.append('sq_cos_1')
    dist1.append(dist_generator('sq sin', t_dist1_0, t_dist1_e, dt, tperiod=8, amp=0.4, offset =0.6))
    dist1_name.append('sq_cos_2')

    dist1.append(chirp_generator(t_dist1_0, t_dist1_e, dt, amp=0.5, offset=0.5, f0=0.1, t1=t_dist1_e, f1=1))
    dist1_name.append('chirp_1')
    dist1.append(chirp_generator(t_dist1_0, t_dist1_e, dt, amp=0.5, offset=0.5, f0=0.05, t1=t_dist1_e, f1=0.75))
    dist1_name.append('chirp_2')
    dist1.append(chirp_generator(t_dist1_0, t_dist1_e, dt, amp=0.5, offset=0.5, f0=0.075, t1=t_dist1_e, f1=1))
    dist1_name.append('chirp_3')
    
    dist1.append(sin_var_amp_generator(t_dist1_0, t_dist1_e, dt, amp0=0.4, ampe=0.5, off0=0.6, offe=0.5, tperiod=5))
    dist1_name.append('ampcos_1')
    dist1.append(sin_var_amp_generator(t_dist1_0, t_dist1_e, dt, 0.2, 0.5, 0.8, 0.5, 10))
    dist1_name.append('ampcos_2')
    dist1.append(sin_var_amp_generator(t_dist1_0, t_dist1_e, dt, 0.3, 0.5, 0.7, 0.5, 5))
    dist1_name.append('ampcos_3')

    slopes = np.linspace(0.2, 0.5, 3, endpoint=True)
    for ind, slope in enumerate(slopes):
        dist1.append(dist_generator('triag', t_dist1_0, t_dist1_e, dt, y0=1, slope=-slope/10))
        dist1_name.append('triag_'+str(ind+1) )
    
    # create second disturbances

    dist2 = []
    dist2_name = []

    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_e, dt, tperiod=5, amp=0.5, offset =0.5))
    dist2_name.append('cosinus_1')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_e, dt, tperiod=8, amp=0.5, offset =0.5))
    dist2_name.append('cosinus_2')
    dist2.append(dist_generator('cos', t_dist2_0, t_dist2_e, dt, tperiod=10, amp=0.5, offset =0.5))
    dist2_name.append('cosinus_3')

    dist2.append(noisy_cos(t_dist2_0, t_dist2_e, dt, amp=0.5, offset=0.5, tperiod=5, noise=0.025))
    dist2_name.append('noisycos_1')
    dist2.append(noisy_cos(t_dist2_0, t_dist2_e, dt, amp=0.5, offset=0.5, tperiod=8, noise=0.025))
    dist2_name.append('noisycos_2')
    dist2.append(noisy_cos(t_dist2_0, t_dist2_e, dt, amp=0.5, offset=0.5, tperiod=10, noise=0.025))
    dist2_name.append('noisycos_3')

    dist2.append(dist_generator('sq sin', t_dist2_0, t_dist2_e, dt, tperiod=5, amp=0.4, offset =0.6))
    dist2_name.append('sq_cos_1')
    dist2.append(dist_generator('sq sin', t_dist2_0, t_dist2_e, dt, tperiod=8, amp=0.4, offset =0.6))
    dist2_name.append('sq_cos_2')

    dist2.append(chirp_generator(t_dist2_0, t_dist2_e, dt, amp=0.5, offset=0.5, f0=0.1, t1=t_dist2_e, f1=1))
    dist2_name.append('chirp_1')
    dist2.append(chirp_generator(t_dist2_0, t_dist2_e, dt, amp=0.5, offset=0.5, f0=0.05, t1=t_dist2_e, f1=0.75))
    dist2_name.append('chirp_2')
    dist2.append(chirp_generator(t_dist2_0, t_dist2_e, dt, amp=0.5, offset=0.5, f0=0.075, t1=t_dist2_e, f1=1))
    dist2_name.append('chirp_3')

    dist2.append(sin_var_amp_generator(t_dist2_0, t_dist2_e, dt, amp0=0.4, ampe=0.5, off0=0.6, offe=0.5, tperiod=5))
    dist2_name.append('ampcos_1')
    dist2.append(sin_var_amp_generator(t_dist2_0, t_dist2_e, dt, 0.2, 0.5, 0.8, 0.5, 10))
    dist2_name.append('ampcos_2')
    dist2.append(sin_var_amp_generator(t_dist2_0, t_dist2_e, dt, 0.3, 0.5, 0.7, 0.5, 5))
    dist2_name.append('ampcos_3')

    slopes = np.linspace(0.2, 0.5, 3, endpoint=True)
    for ind, slope in enumerate(slopes):
        dist2.append(dist_generator('triag', t_dist2_0, t_dist2_e, dt, y0=1, slope=-slope/10))
        dist2_name.append('triag_'+str(ind+1) )

    # add preselected random disturbances

    mypath =  os.path.dirname(os.path.realpath(__file__))
    f = listdir(mypath)
    onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and 'random_dist' in f and '.pkl' in f)]
    
    for disturbance_file in onlyfiles[0:6]:
        with open(mypath+'/'+disturbance_file, 'rb') as f:
            obj = dill.load(f)
            dist1_name.append(obj['dist_name'])
            dist1.append(obj['dist1'])
            dist2_name.append(obj['dist_name'])
            dist2.append(obj['dist2'])
    
    return dist1, dist1_name, dist2, dist2_name
    

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