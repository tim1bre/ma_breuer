import numpy as np
from Code.S02_Allgemein.disturbance_generator import dist_generator, chirp_generator, sin_var_amp_generator, random_walk_dist_long, noisy_cos, moving_mean
import dill
import matplotlib.pyplot as plt

def create_random_disturbances(t_dist1_0, duration_dist, dt):
    """ create random disturbances, set keep = 1 in debugger to save them to later use """

    path = r"C:\Users\timbr\OneDrive\Masterarbeit\Repo\ma_breuer\Code\S01_Experimente\Paper\\"

    t_dist1_e = t_dist1_0 + duration_dist
        
    more_disturbances = 1
    i = 1
    keep = 1

    while more_disturbances == 1:
        
        dist_name = 'random_'+str(i)
        dist1 = random_walk_dist_long(t_dist1_0, t_dist1_e, dt, 0.01)

        t = [i[0] for i in dist1]
        y = [i[1] for i in dist1]

        plt.plot(t, y)
        plt.show()

        if keep == 1:

            dist = {}
            dist['dist_name'] = dist_name
            dist['dist'] = dist1
            
            filename = path+'random_dist'+str(i)+'.pkl'
            with open(filename, 'wb') as f:
                dill.dump(dist, f)

            i += 1

dt = 0.05
duration_dist = 400
t_dist1_0 = 100 # start first disturbance

create_random_disturbances(t_dist1_0, duration_dist, dt)