import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class detect_dtw_mean:
    """ disturbance detection using dytnamic time warping """
    
    def get_ref(self, data):
        """ get reference from undisturbed signal and determine thresholds"""

        # determine threshold for dtw
        self.ref = data['y0_filt'][self.i_start_ref:self.i_start_ref+self.dtw_bin_size:self.down_sample]
        self.ref = self.ref - np.mean(self.ref) # remove mean
        compare_list = [data['y0_filt'][self.i_start_ref+i*self.dtw_bin_size:self.i_start_ref+(i+1)*self.dtw_bin_size:self.down_sample] for i in range(1,3) ]
        compare_list = [i - np.mean(i) for i in compare_list] # remove mean
        distance = [fastdtw(x, self.ref, dist=euclidean)[0] for x in compare_list]
        self.dtw_thres = 4*np.max(distance)

    def detect_dist(self, data, tank):
        """ method used in simulation, calls helper methods to determine pdist and setpoint"""

        self.det_pdist(data, tank)
        if tank.pdist[0] > 0:
            self.det_new_setpoint(data, tank)

    def det_pdist(self, data, tank):
        """ determine probability of disturbance"""
        
        # if first call, calculate bin sizes
        if self.first_call==0:
            self.dtw_bin_size = int(self.dtw_bin_seconds/tank.dt)
            self.i_start_ref = int(self.t_start_ref/tank.dt)
            self.first_call = 1

        if tank.t >= self.te_get_thres:

            # get reference and threshold
            if self.ref_on == 0:
                self.get_ref(data)
                self.ref_on = 1
            
            # calculate distance from signal to reference using dtw
            x = data['y0_filt'][-self.dtw_bin_size::self.down_sample]
            x = x - np.mean(x) # remove mean

            distance, _ = fastdtw(x, self.ref, dist=euclidean)
            
            # check if distance is greater than threshold
            if distance >= self.dtw_thres:
                tank.pdist[0] = 1
            else:
                tank.pdist[0] -= 1/(18/0.01)

                if tank.pdist[0] <= 0:
                    tank.pdist[0] = 0

        # start of experiment, use of methods starts at t = 95s
        else:
            distance = 0
            tank.pdist[0] = 0
        
        # save data, for debugging
        self.distance_list.append(distance)