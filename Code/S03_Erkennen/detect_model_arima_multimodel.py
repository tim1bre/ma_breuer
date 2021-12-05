import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from Code.S03_Erkennen.detect_model import detect_model

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from fastdtw import fastdtw

class detect_model_arima_multimodel(detect_model):

    def __init__(self): #, global_forecast=1):  
        self.forecast_based = 25 # number of measurements the model is built on
        self.forecast_future = 10 # number of measurements the model predicts
        self.name = 'multimodel arima'
        self.pred_result_local = [[0],[0]]
        self.downsample_local = 5
        self.first_call = 0 # first call of detect_dist
        self.write_data = 0 # write predict results for plot
        self.debug_print = 0 # 0 = no print, 1 = local, 2 = global
        self.saved_models = {}
        self.saved_models['data'] = []
        self.saved_models['drop'] = []
        self.saved_models_count = 0
        self.enter_saved = 0
        self.model_update = 1
        self.overwrite_setpoint = 1 # 
        self.dist_new = 1
        self.dist_list = []

    def detect_dist(self, data, tank):

        # init data storage
        if (self.first_call == 0):
            self.first_call = 1
            data['arma_y'] = []
            data['t_arma_y'] = []
            data['arma_p'] = []
            data['t_arma_p'] = []
            data['t_arma_p'].append(0)
            data['pdist_forecast'] = []
            data['pdist_forecast'].append(0)
        
        # detection model-based on a parallel simulation
        self.det_pdist(data, tank)

        for i in range(0, tank.tank_count):

            if (tank.pdist[i] >= 1) & (tank.t > self.te_get_thres):
                """disturbance: get new setpoint"""

                if self.dist_new == 1:
                    #self.dist_count += 1
                    self.dist_new = 0
                    self.dist_list.append(tank.t)

                # make local prediction within a disturbance to get new setpoint based on arima model

                # use last 1.25s of signal 
                y_movingav = [np.mean(data['y'+str(i)+'_filt'][j:j+6]) for j in range(len(data['y'+str(i)+'_filt'])-5) ]
                mod_data = y_movingav[::self.downsample_local][-self.forecast_based:] # last 1.25 secs
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    # check whether update is needed
                    if (self.model_update == 1) or ((self.model_update == 0) and (mod_data[-1] < self.model_lowlim)):

                        try:
                            # differentiate twice to achieve stationarity
                            diff1 = np.diff(mod_data)
                            diff2 = np.diff(diff1)

                            model_fit_local = ARIMA(diff2, order=(2,2,1)).fit(disp=0)
                            forecast_diff2 = model_fit_local.forecast(self.forecast_future)[0]

                            # reverse differentiating
                            forecast_diff1 = np.cumsum(forecast_diff2) + diff1[-1]
                            forecast = np.cumsum(forecast_diff1) + mod_data[-1]
                            
                            self.pred_result_local = list(forecast)
                            self.model_update = 0 # no update in next timestep is needed

                            # lower limit that y needs to fall below to trigger fit of new model and forecast
                            self.model_lowlim = min(self.model_lowlim, forecast[0]+0.25*(forecast[-1]-forecast[0])) 

                            if self.write_data == 1:
                                self.t_pred_local = [data['t'][-1] + self.downsample_local*tank.dt*i for i in range(0, len(self.pred_result_local))]
                                data['arma_y'].append(self.pred_result_local)
                                data['t_arma_y'].append(self.t_pred_local)

                            # determine new setpoint based on forecast

                            # no dist_setpoint, use last value of forecast if it shows downward trend
                            if tank.dist_setpoint == None:    
                                tank.dist_setpoint = 2* tank.pi['0'].y_s0 - self.pred_result_local[-1]

                                if tank.dist_setpoint < tank.pi['0'].y_s0:
                                    tank.dist_setpoint = tank.pi['0'].y_s0

                            # only overwrite dist_setpoint if y is predicted to fall even more
                            else:
                                tank.dist_setpoint = max(0, tank.dist_setpoint, 2* tank.pi['0'].y_s0 - self.pred_result_local[-1])
                            
                            if self.debug_print == 1:
                                print('arima fit succeeded at ' +str(tank.t))

                        except Exception as e:
                            if self.debug_print == 1:
                                print('arima fit failed at ' +str(tank.t))
                                print(e)
                            
                            if tank.dist_setpoint == None:
                                tank.dist_setpoint = tank.pi['0'].y_s0
                           
            elif tank.pdist[i] == 0:
                """ no disturbance, save recent disturbance if not done yet """

                self.model_update = 1
                self.enter_saved = 0
                self.dist_new = 1
                self.model_lowlim = tank.pi[str(i)].y_s0 # set lowerlimit to initial value after disturbance
            
                # get necessary reserve and save it for future disturbances
                if len(self.dist_list) > self.saved_models_count:

                    # determine index of start and end of disturbance
                    y_movingav = pd.DataFrame(data['y'+str(i)+'_filt']).rolling(5).mean()
                    t_start = self.dist_list[-1]
                    i_start = np.abs(data['t'] - t_start).argmin() - 200
                    t_end = tank.t
                    i_end = np.abs(data['t'] - t_end).argmin()
                    y_movingav_list = [i[0] for i in np.array(y_movingav[i_start:i_end])]
                    ind_lmin = find_peaks([-1*i for i in y_movingav_list], prominence=0.15)[0]
                    ind_lmax = find_peaks([i for i in y_movingav_list], prominence=0.15)[0]
                    lmin = [y_movingav_list[i] for i in ind_lmin]
                    lmax = [y_movingav_list[i] for i in ind_lmax]
                    
                    if len(lmin)==len(lmax):
                        mean_drop = np.median(np.subtract(lmax[:-1], lmin[1:])) 
                    elif len(lmin)-1==len(lmax):
                        mean_drop = np.median(np.subtract(lmax[:-1], lmin[1:-1])) 
                        print('lmin größer als lmax')
                    elif len(lmin)==len(lmax)-1:
                        mean_drop = np.median(np.subtract(lmax[:-1], lmin)) 
                        print('lmax größer als lmin')
                    else:
                        print("error")

                    self.saved_models['data'].append(y_movingav_list)
                    self.saved_models['drop'].append(mean_drop)

                    print("count: " +str(self.saved_models_count))
                    print('drop: ' + str(mean_drop))

                    self.saved_models_count += 1
                
            if (tank.pdist[i] >= 0.2) & (self.saved_models_count >= 2):
                """ disturbance: use information on past disturbances """

                self.use_saved_models(data, tank, i)

    def use_saved_models(self, data, tank, i):

        compare_bin = 200 # time steps since entered disturbance
        self.overwrite_setpoint = 1

        if self.enter_saved == 200:
            compare_bin = 400
            self.overwrite_setpoint = 1
        elif self.enter_saved == 250:
            compare_bin = 450
        elif self.enter_saved == 300:
            compare_bin = 500
        elif self.enter_saved == 350:
            compare_bin = 550
        elif self.enter_saved == 400:
            compare_bin = 600
        elif self.enter_saved == 450:
            compare_bin = 650
        elif self.enter_saved == 500:
            compare_bin = 700
        
        if compare_bin >= 200:
        
            if (self.enter_saved%50 == 0):

                y_movingav = pd.DataFrame(data['y'+str(i)+'_filt']).rolling(5).mean()

                compare_dist_list = []
                for ref_data in self.saved_models['data']:
                    distance, _ = fastdtw(y_movingav[-compare_bin:], ref_data[:compare_bin], dist=euclidean)
                    compare_dist_list.append(distance)

                self.best_model_index = compare_dist_list.index(np.min(compare_dist_list))
                new_dist_setpoint = tank.pi['0'].y_s0 + self.saved_models['drop'][self.best_model_index] #+ 0.1
                if self.enter_saved == 200:
                    print('------------------')
                if self.enter_saved%50 == 0:
                    print('compare bin: ' + str(compare_bin))
                    print('self.best_model_index: ' + str(self.best_model_index))

                if (self.enter_saved > 1300) & (np.any(y_movingav[-300:] < 18)):
                    self.overwrite_setpoint = 0
                    #print('turn off overwrite')
                if (self.enter_saved > 2300) & (np.any(y_movingav[-300:] < 19.5)):
                    self.overwrite_setpoint = 0
                    #print('turn off overwrite')
                
            else:
                new_dist_setpoint = tank.pi['0'].y_s0 + self.saved_models['drop'][self.best_model_index]
        
            if self.overwrite_setpoint == 1:
                tank.dist_setpoint = new_dist_setpoint

        self.enter_saved += 1

        