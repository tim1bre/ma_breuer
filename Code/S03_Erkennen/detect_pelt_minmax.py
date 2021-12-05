import numpy as np
import ruptures as rpt # PELT implementation
import matplotlib.pyplot as plt
from Code.S03_Erkennen.setpoint_algo import setpoint_algo

class detect_pelt_minmax(setpoint_algo):
    """ disturbance detection using pruned exact linear time (PELT) method """

    def __init__(self, pen=5):

        self.pen = pen # penalty
        self.name = 'pelt_minmax'
        self.downsample = 5 # downsample of data
        self.call_num = 0 # number of calls
        self.dn_calls = 9 # number of elements between calls
        self.window_size =  100 # use last 200 data points, truncate data for faster runtime 

        self.flag_threshold = 0
        self.start_method = 95
        self.t0_get_thres = 70 # start for setpoint_algo
        self.timespan = 'full' # window size for setpoint algo

        self.t_end_readjust = 0 # expected timespan during which readjustment to original setpoint takes place
        self.delta_list = []
        
        # save data for animation
        self.anim_on = 0
        if self.anim_on == 1:
            print('collect data for animation!')
            self.anim_points = []
            self.anim_points_time = []
            self.anim_changepoints = []
            self.anim_cp_idx = []
            self.anim_timespamp = []
            self.anim_t_adjust = []

    def get_thres(self, data):

        i0 = data['t'].index(min(data['t'], key=lambda x:abs(x-self.t0_get_thres)))
        delta_y = max(data['y0_filt'][i0:]) - min(data['y0_filt'][i0:])

        self.delta_y_thres = 3.5 * delta_y
    
    def detect_dist(self, data, tank):
        """ method used in simulation, calculate pdist and setpoint"""

        if tank.t >= self.start_method:

            if self.flag_threshold == 0:
                
                self.get_thres(data)
                self.flag_threshold = 1

            # call method only every 10th iteration
            if self.call_num == self.dn_calls:

                # changepoint detection using PELT from ruptures package
                points = np.array(data['y0_filt'][-self.window_size*self.downsample::self.downsample]) # data used
                model="rbf"
                algo = rpt.Pelt(model=model).fit(points)
                result = algo.predict(pen=self.pen) # pen = penalty value

                t_changepoints = [data['t'][-self.window_size*self.downsample] + i*tank.dt*self.downsample for i in result] # convert index in time (approx)
                t_changepoints = [i for i in t_changepoints if (i >= self.start_method) and (i >=self.t_end_readjust)]

                # save data for animation
                if self.anim_on == 1:
                    self.anim_points.append(points)
                    self.anim_points_time.append(np.array(data['t'][-self.window_size*self.downsample::self.downsample]))
                    self.anim_cp_idx.append(result)
                    self.anim_changepoints.append(t_changepoints)
                    self.anim_timespamp.append(tank.t)
                    self.anim_t_adjust.append(self.t_end_readjust)

                if len(t_changepoints) > 1:

                    # index last relevant changepoint

                    idx_cp = data['t'].index(min(data['t'], key=lambda x:abs(x-t_changepoints[-2])))
                    delta_y = max(data['y0_filt'][idx_cp:]) - min(data['y0_filt'][idx_cp:])
                    self.delta_list.append(delta_y)

                    if delta_y > self.delta_y_thres:
                        tank.pdist[0] = 1
                        self.det_new_setpoint(data, tank) # calculate new setpoint
                    else:
                        tank.pdist[0] -= 0.1 # decrease disturbance probability

                        if tank.pdist[0] <= 0:
                            tank.pdist[0] = 0

                            # block detection for 15 seconds
                            self.t_end_readjust = tank.t + 10

                elif (tank.t < self.t_end_readjust) & (data['y0_filt'][-1] < tank.pi['0'].y_s - 0.1):
                    print('blockieren aufheben')
                    self.t_end_readjust = 0
                    
                self.call_num = 0
            else:
                self.call_num += 1
        else:

            # assume no disturbance during first part of experiment
            tank.pdist[0] = 0

    def plot_result(self, points, result):
        """ plot changepoints and data """
        
        rpt.display(points, result, figsize=(10, 6))
        plt.show()  

