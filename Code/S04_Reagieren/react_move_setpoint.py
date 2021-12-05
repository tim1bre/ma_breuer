from Code.S02_Allgemein.find_minima import find_minima
import numpy as np

class react_move_setpoint():
    
    def __init__(self, dist_dec, limit_reserve = 1):

        self.dist_dec = dist_dec
        self.limit_reserve = limit_reserve
        self.upperlim_reserve = 0.001
        self.lowerlim_reserve = 0.0005
        self.dist_start = 0
        self.t_lastmin = 0
        self.new_dist = 1

        if self.limit_reserve == 0:
            self.name = 'move setpoint'
        elif self.limit_reserve == 1:
            self.name = 'move setpoint lim'

    def react_dist(self, tank, u):

        if tank.t > 15:

            for i in range(0, tank.tank_count):

                if (tank.pdist[i] > 0) or (tank.pdist_forecast==1):
                    tank.pi[str(i)].y_s = tank.dist_setpoint - tank.pi[str(i)].correct # move setpoint to create reserve

                elif tank.pdist[i] == 0:

                    tank.pi[str(i)].y_s -= tank.pi[str(i)].y_s0 * self.dist_dec
                            
                    if tank.pi[str(i)].y_s < tank.pi[str(i)].y_s0:
                        tank.pi[str(i)].y_s = tank.pi[str(i)].y_s0
                    
        if self.limit_reserve == 0:
            tank.pi['0'].correct = 0

    def adjust_setpoint(self, data, tank):

        if self.limit_reserve == 1:

            for i in range(tank.tank_count):

                if tank.pdist[i] > 0:

                    # get start point of local disturbance and determine minima of y0_filt
                    if self.new_dist == 1:
                        #self.dist_start = [i for i,v in enumerate(data['pdist0'][self.dist_start:]) if v == 0][-1] + self.dist_start
                        self.dist_start = int(tank.t/tank.dt)
                        self.new_dist = 0
                    minima = find_minima(data['y' + str(i) + '_filt'][self.dist_start::])
                
                    # check that it is a new minimum
                    if len(minima) > self.min_count:

                        # check if setpoint is too high
                        if minima[self.min_count] > (tank.pi[str(i)].y_s0 + self.upperlim_reserve):
                            tank.pi[str(i)].correct = minima[self.min_count] - (tank.pi[str(i)].y_s0 + self.upperlim_reserve) + tank.pi[str(i)].correct
                        # check if setpoint is too low
                        elif minima[self.min_count] < (tank.pi[str(i)].y_s0 - self.lowerlim_reserve):

                            # "anti-windup" for correct term: if since last minimum u was in average above 95% of u_max, adjusting the setpoint won't help
                            if(np.mean(data['u0'][int(self.t_lastmin/tank.dt):]) < tank.pi['0'].u_max*0.95):
                                tank.pi[str(i)].correct = minima[self.min_count] - (tank.pi[str(i)].y_s0 + self.upperlim_reserve) + tank.pi[str(i)].correct
                        
                        self.t_lastmin = tank.t
                        self.min_count += 1

                else:
                    tank.pi[str(i)].correct = 0
                    self.min_count = 1 # count of minima within local disturbance
                    self.new_dist = 1