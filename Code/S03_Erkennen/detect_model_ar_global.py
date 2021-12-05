""" global ist mit in arima! --> kann entfernt werden """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from Code.S03_Erkennen.detect_model import detect_model

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

class detect_model_ar(detect_model):

    def __init__(self, global_forecast=1):  
        self.forecast_based = 25 # number of measurements the model is built on
        self.forecast_future = 20 # number of measurements the model predicts
        self.model_update = 60 # update arima model (local) after n measurements
        self.i = 0
        self.global_forecast = global_forecast

        if global_forecast == 1:
            self.name = 'model ar l+g'
        else:
            self.name = 'model ar l'

        self.pred_result_local = [[0],[0]]

        self.downsample_local = 5
        self.downsample_global = 100

        self.dist_list = [] # list to save start of individual disturbances
        self.first_call = 0 # first call of detect_dist
        
        self.init_global_model = 0

        self.t_start_next_dist = None

        self.start_liveplot = 100 # start liveplot
        self.count_start_global = 4 # after how many dist start to predict future disturbances
        self.write_data = 1 # write predict results for plot

        self.debug_print = 0 # 0 = no print, 1 = local, 2 = global

    def detect_dist(self, data, tank):

        if (self.first_call == 0): #& (tank.t > 5):
            #tank.dt =  np.round(data['t'][-1] - data['t'][-2], 2)
            self.first_call = 1
            data['arma_y'] = []
            data['t_arma_y'] = []
            data['arma_p'] = []
            data['t_arma_p'] = []
            data['t_arma_p'].append(0) #tank.t)
            data['pdist_forecast'] = []
            data['pdist_forecast'].append(0)
        
        # detection model-based on a parallel simulation
        self.det_pdist(data, tank)

        for i in range(0, tank.tank_count):

            # add disturbance to list to use it for global forecast
            if (tank.pdist[i] >= 0.9) & (tank.t > self.te_get_thres):

                if tank.pdist[i] >= 1:
                    if len(self.dist_list)==0:
                        self.dist_list.append(tank.t)

                        if self.debug_print == 2:
                            print('dist_list: ' + str(self.dist_list))
                    elif tank.t-self.dist_list[-1] > 50:
                        self.dist_list.append(tank.t)

                        if self.debug_print == 2:
                            print('dist_list: ' + str(self.dist_list))

                    # init use of global modell
                    if (len(self.dist_list) == self.count_start_global) & (self.init_global_model==0) & (self.global_forecast==1):
                        lags = int( np.mean(np.diff(self.dist_list))/(tank.dt*self.downsample_global) )
                        self.enter_end = tank.t + lags - 6
                        self.init_global_model = 1

                # get new setpoint based on arima model
                if self.i == 0:

                    y_movingav = pd.DataFrame(data['y'+str(i)+'_filt']).rolling(5).mean()

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")

                        try:
                            self.model_local = ARIMA(y_movingav[::self.downsample_local][-self.forecast_based:], order=(3,0,0))
                            model_fit_local = self.model_local.fit(disp=0)
                            self.pred_result_local = model_fit_local.forecast(self.forecast_future)
                            self.t_pred_local = [data['t'][-1] + self.downsample_local*tank.dt*i for i in range(0, len(self.pred_result_local[0]))]

                            if self.write_data == 1:
                                data['arma_y'].append(self.pred_result_local)
                                data['t_arma_y'].append(self.t_pred_local)

                            if tank.dist_setpoint == None:    
                                tank.dist_setpoint = 2* tank.pi['0'].y_s0 - self.pred_result_local[0][-1]
                            else:
                                tank.dist_setpoint = max(0, tank.dist_setpoint, 2* tank.pi['0'].y_s0 - self.pred_result_local[0][-1])
                            
                            if self.debug_print == 1:
                                print('arma fit succeeded at ' +str(tank.t))

                        except:
                            if self.debug_print == 1:
                                print('arma fit failed at ' +str(tank.t))
                            
                            if tank.dist_setpoint == None:
                                tank.dist_setpoint = tank.pi['0'].y_s0
                                self.i = -1 # new model in next interation

                # check model qualility
                else:
                    try:
                        error = self.pred_result_local[0][self.i-1] - data['y0'][-1]
                        if (error > 0.1) | (error < -0.5) :
                            if self.debug_print == 1:
                                print('new model necessary!')
                            self.i = -1 # new model in next interation
                    except:
                        pass

                self.i += 1
                if self.i >= self.model_update:
                    self.i = 0

                if (tank.liveplot == 1) & (tank.t > self.start_liveplot):
                    self.update_liveplot(tank, data)

            # global model to forecast pdist and therefore future disturbances
            elif ((len(self.dist_list)) >= self.count_start_global) & (self.global_forecast==1):

                if (tank.t >= self.enter_end) & (tank.pdist[0] == 0):

                    lags = int( np.mean(np.diff(self.dist_list))/(tank.dt*self.downsample_global) )

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        self.model_global = sm.tsa.statespace.SARIMAX(data['pdist0'][data['t'].index(80)::self.downsample_global],order=(3,0,0),seasonal_order=(3,0,0,lags), enforce_stationarity=False, enforce_invertibility=False, simple_differencing=False)
                        model_fit_global = self.model_global.fit(disp=0)
                        self.pred_result_global = model_fit_global.forecast(self.forecast_future)

                    self.t_pred_global = [data['t'][-1] + self.downsample_global*tank.dt*i for i in range(0, len(self.pred_result_global))]
                        
                    if self.write_data == 1:
                        data['arma_p'].append(self.pred_result_global)
                        data['t_arma_p'].append(self.t_pred_global)

                    try:
                        self.t_start_next_dist = self.t_pred_global[np.argwhere(self.pred_result_global>0.6)[0][0]]
                        self.t_end_next_dist = self.t_start_next_dist + 5
                        self.enter_end = self.t_start_next_dist + lags - 10
                    except:

                        if self.debug_print == 1:
                                print('sarimax fit failed at ' +str(tank.t))

                        self.enter_end = tank.t + 5
                
                    print('start next dist: '+str( self.t_start_next_dist))
                    if self.debug_print == 1:
                        print('sarimax fit succeeded at ' +str(tank.t))

                    if (tank.liveplot == 1) & (tank.t > self.start_liveplot):
                        self.update_liveplot(tank, data)

        if (self.t_start_next_dist != None) & (self.global_forecast==1):
            if (tank.t >= self.t_start_next_dist - 12.5) & (tank.t <= self.t_end_next_dist):
                tank.pdist_forecast = 1

                if self.debug_print == 2:
                    print('pdist_forecast at :' + str(tank.t))
                data['pdist_forecast'].append(1)
            else:
                tank.pdist_forecast = 0
                data['pdist_forecast'].append(0)

    def update_liveplot(self, tank, data):
        
        if (len(self.pred_resul_global[0])>5) & (len(self.pred_result_local[0])>5):

            tank.axs1[0][0].clear()
            tank.axs1[1][0].clear()
            tank.axs1[2][0].clear()
            tank.axs1[0][0].set(ylabel='Ventil')
            tank.axs1[0][0].plot(data['t'][-450:], data['dist0'][-450:], 'r-')
            tank.axs1[1][0].set(ylabel='Füllstand')
            tank.axs1[1][0].plot(data['t'][-450:], data['y0'][-450:], self.t_pred_local, self.pred_result_local[0])
            tank.axs1[2][0].set(ylabel='Sollwert')
            tank.axs1[2][0].plot(data['t'][-450:], data['pi0'][-450:])

            tank.axs1[0][1].clear()
            tank.axs1[1][1].clear()
            tank.axs1[2][1].clear()
            tank.axs1[0][1].set(ylabel='Ventil')
            tank.axs1[0][1].plot(data['t'], data['dist0'], 'r-')
            tank.axs1[1][1].set(ylabel='Störung')
            tank.axs1[1][1].plot(data['t'][:-1], data['pdist0'], self.t_pred_global, self.pred_result_global)
            tank.axs1[2][1].set(ylabel='Sollwert')
            tank.axs1[2][1].plot(data['t'][:-1], data['pi0'])

            plt.pause(0.0001)

        elif len(self.pred_result_local[0])>5:

            tank.axs1[0][0].clear()
            tank.axs1[1][0].clear()
            tank.axs1[2][0].clear()
            tank.axs1[0][0].set(ylabel='Ventil')
            tank.axs1[0][0].plot(data['t'][-450:], data['dist0'][-450:], 'r-')
            tank.axs1[1][0].set(ylabel='Füllstand')
            tank.axs1[1][0].plot(data['t'][-450:], data['y0'][-450:], self.t_pred_local, self.pred_result_local[0])
            tank.axs1[2][0].set(ylabel='Sollwert')
            tank.axs1[2][0].plot(data['t'][-450:], data['pi0'][-450:])

            tank.axs1[0][1].clear()
            tank.axs1[1][1].clear()
            tank.axs1[2][1].clear()
            tank.axs1[0][1].set(ylabel='Ventil')
            tank.axs1[0][1].plot(data['t'], data['dist0'], 'r-')
            tank.axs1[1][1].set(ylabel='Füllstand')
            tank.axs1[1][1].plot(data['t'], data['y0'], self.t_pred_local, self.pred_result_local[0])
            tank.axs1[2][1].set(ylabel='Sollwert')
            tank.axs1[2][1].plot(data['t'][:-1], data['pi0'])

            plt.pause(0.0001)

        