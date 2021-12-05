import numpy as np
from scipy import signal

class setpoint_algo:

    def det_new_setpoint(self, data, tank):
        """ move setpoint based on past data, exclude begin of experiment """

        if tank.t < self.t0_get_thres: # exclude begin of experiment
            tank.dist_setpoint = tank.pi['0'].y_s0
        else: 
            if self.timespan == 'full':
                i_start = np.abs(np.subtract(data['t'], self.t0_get_thres)).argmin()
            else:
                t_start = tank.t - self.timespan
                if t_start > self.t0_get_thres:
                    i_start = np.abs(np.subtract(data['t'], t_start)).argmin()
                else:
                    i_start = np.abs(np.subtract(data['t'], self.t0_get_thres)).argmin()

            y_min = [np.min(data['y'+str(i)+'_filt'][i_start:]) for i in range(0, tank.tank_count)]

            for i in range(0, tank.tank_count):
                tank.dist_setpoint = 2 * tank.pi[str(i)].y_s0 - np.min(y_min)
                if tank.dist_setpoint < tank.pi[str(i)].y_s0:
                    tank.dist_setpoint = tank.pi[str(i)].y_s0