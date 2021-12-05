import numpy as np

class detect_gradient:
    name = 'gradient algo'
    
    def __init__(self, factor_y_min, t_start_learning=70, sample_count = 55, pdist_dec = 0.075):
        self.dy_threshold = None
        self.factor_y_min = factor_y_min
        self.t_start_learning = t_start_learning
        self.sample_count = sample_count
        self.pdist_dec = pdist_dec
    
    def detect_dist(self, data, tank):

        self.y_min = [tank.pi[str(i)].y_s0 * self.factor_y_min for i in range(0, tank.tank_count)] # minimum value, lower than than --> asume disturbance #TODO: could be moved in init
        tank.dy = [0] * tank.tank_count # init list for gradient

        for i in range(0, tank.tank_count):

            if data['y' + str(i) +'_filt'][-1] < self.y_min[i] and tank.t > self.t_start_learning:
                self.learn_threshold_grad(tank, data, i)

            tank.dy[i] = self.calculate_grad(data['y' + str(i) + '_filt'], tank, i)

        tank.dy = tuple(tank.dy)

    def calculate_grad(self, x, tank, i):
        """ approx gradient of x, difference of mean of two bins """
        
        if len(x) > 2*self.sample_count:
            y_bin1 = x[-self.sample_count:]
            y_bin2 = x[-2*self.sample_count:-self.sample_count]

            y_grad = (np.mean(y_bin1)-np.mean(y_bin2))*tank.dt

        else:
            y_grad = 0

        if self.dy_threshold is None:
            tank.pdist[i] = 0

        elif (y_grad < self.dy_threshold) or (x[-1]<self.y_min[i]):
            tank.pdist[i] = 1
        else:
            tank.pdist[i] -= tank.dt*self.pdist_dec

            if tank.pdist[i] < 0:
                tank.pdist[i] = 0

        return y_grad

    def learn_threshold_grad(self, tank, data, i):
        """ learn threshold for gradient: gradient more negative --> disturbance """

        t_start = tank.t - 20
        i_start = np.abs(data['t'] - t_start).argmin()

        if self.dy_threshold is None:
            self.dy_threshold = np.min(data['dy'+str(i)][i_start:]) # look for largest negative gradient
            tank.dist_setpoint = 2 * tank.pi[str(i)].y_s0 - np.min(data['y'+str(i)][i_start:])

        elif np.min(data['dy'+str(i)][i_start:]) > self.dy_threshold: # overwrite if there is a gradient that indicates a disturbance faster
            self.dy_threshold = np.min(data['dy'+str(i)][i_start:])

        if tank.dist_setpoint < 2 * tank.pi[str(i)].y_s0 - np.min(data['y'+str(i)][i_start:]):
            tank.dist_setpoint = 2 * tank.pi[str(i)].y_s0 - np.min(data['y'+str(i)][i_start:])

        if tank.dist_setpoint < tank.pi[str(i)].y_s0:
            tank.dist_setpoint = tank.pi[str(i)].y_s0

        