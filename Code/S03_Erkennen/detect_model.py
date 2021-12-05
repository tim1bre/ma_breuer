import numpy as np
from scipy import signal

class detect_model:
    t0_get_thres = 70 # start time for interval to determine model threshold
    te_get_thres = 95 # end time for interval to determine model threshold

    def detect_dist(self, data, tank):
        """ detection dist based on a parallel simulation """

        self.det_pdist(data, tank)
        self.det_new_setpoint(data, tank)

    def det_pdist(self, data, tank):
        """ calculate "probability" of disturbance """

        maf_num = 35 # number of samples for moving average filter
        y_mes = [0] * tank.tank_count
        y_mod = [0] * tank.tank_count

        if len(data['y0']) < maf_num:
            maf_num = len(data['y0'])

        for i in range(0, tank.tank_count):
            y_mes[i] = data['y'+str(i)+'_filt'] 
            y_mod[i] = data['y'+str(i)+'_mod']

        # calculate difference between "real" system and parallel simulation
        mod_err = np.average(np.subtract(y_mes[0][-maf_num:], y_mod[0][-maf_num:]))
        tank.mod_error.append(mod_err)

        for i in range(0, tank.tank_count):
            if tank.t < self.te_get_thres-0.2:
                tank.pdist[i] = 0
            elif tank.t <= self.te_get_thres:
                tank.pdist[i] = 0

                # calculate threshold for model error
                max_error = np.max(np.abs(tank.mod_error[data['t'].index(self.t0_get_thres):]))
                std_error = np.std(tank.mod_error[data['t'].index(self.t0_get_thres):])

                self.model_thres = max_error + std_error
                
            elif abs(mod_err) > self.model_thres:
                tank.pdist[i] = 1
            else:
                tank.pdist[i] -= 0.8*tank.pi[str(i)].dt/10

                if tank.pdist[i] < 0:
                    tank.pdist[i] = 0

        

            
            