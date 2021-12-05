import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy import signal

from Code.S03_Erkennen.detect_model import detect_model

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

from Code.S02_Allgemein.filter import fft

class detect_model_arima(detect_model):
    """ model-based detection, disturbance setpoint based on arima forecast"""

    def __init__(self, global_forecast=1, downsample_local=1):  
        self.forecast_based = 15 # seconds number of measurements the model is built on
        self.forecast_future = 1 # number of measurements the model predicts
        self.global_forecast = global_forecast
        if global_forecast == 1:
            self.name = 'model sarima'
        else:
            self.name = 'model arima'
        self.pred_result_local = [[0],[0]]
        self.downsample_local = downsample_local # downsample so that Hessian Matrix doesn't become singular
        self.downsample_global = 500 # downsample so that Hessian Matrix doesn't become singular
        self.dist_list = [] # list to save start of individual disturbances
        self.first_call = 0 # first call of detect_dist
        self.init_global_model = 0
        self.t_start_next_dist = None # time when next disturbance is predicted to start
        self.count_start_global = 4 # after how many dist start to predict future disturbances
        self.write_data =  0 # save prediction results in list
        if self.write_data == 1:
            print("write data active, slower performance")
        self.debug_print = 0 # 0 = no print, 1 = local, 2 = global

    def detect_dist(self, data, tank):

        # init lists for prediction data
        if (self.first_call == 0):
            self.first_call = 1
            data['arma_y'] = []
            data['t_arma_y'] = []
            data['arma_p'] = []
            data['t_arma_p'] = []
            data['pdist_forecast'] = []
            self.num, self.den = signal.butter(N=4, Wn=1.5, btype='low', analog=False, fs=1/tank.dt)
            # no prediction at the beginning to make plot easier to undestand
            data['t_arma_p'].append(0) # 0
            data['pdist_forecast'].append(0)
            tank.dist_setpoint = tank.pi['0'].y_s0
        
        # detection of disturbance
        self.det_pdist(data, tank)

        for i in range(0, tank.tank_count):

            # enter loop after model has settled down
            if (tank.pdist[i] >= 0.9) & (tank.t > self.te_get_thres):

                # get start time of disturbance
                if (tank.pdist[i] >= 1) & (self.global_forecast == 1):
                    if len(self.dist_list) == 0:
                        self.dist_list.append(tank.t) # add start of first disturbance

                        if self.debug_print == 2:
                            print('dist_list: ' + str(self.dist_list))

                    elif tank.t - self.dist_list[-1] > 50: # add start of new disturbance, min dist between disturbances is 50s
                        self.dist_list.append(tank.t)

                        if self.debug_print == 2:
                            print('dist_list: ' + str(self.dist_list))

                    # init use of global modell if enough disturbances have ocurred
                    if (len(self.dist_list) == self.count_start_global) & (self.init_global_model==0) & (self.global_forecast==1):
                        lags = int( np.mean(np.diff(self.dist_list))/(tank.dt*self.downsample_global) )
                        self.enter_end = tank.t + lags - 6
                        self.init_global_model = 1 # global model is initialized

                # make local prediction within a disturbance to get new setpoint based on arima model

                #y_movingav = [np.mean(data['y'+str(i)+'_filt'][j:j+11]) for j in range(int((tank.t-self.forecast_based)/tank.dt), len(data['y0_filt'])-10)]
                #mod_data = y_movingav[::self.downsample_local]

                y_filt = [signal.filtfilt(self.num, self.den, data['y0'])][-1]
                mod_data = y_filt[-int((self.forecast_based)/tank.dt)::self.downsample_local]

                
                
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    # check whether update is needed
                    if (self.model_update == 1) or ((self.model_update == 0) and (data['y0_filt'][-1] <= self.model_lowlim)):

                        try:
                            # differentiate to achieve stationarity
                            diff_count = 0
                            start_values = []
                            while(mod_data[0]-mod_data[-1] > 2*tank.noise_sigma):
                                start_values.append(mod_data[0])
                                mod_data = np.diff(mod_data)
                                diff_count = diff_count + 1
                            
                            model_fit_local = ARIMA(mod_data, order=(2, diff_count, 1)).fit(disp=0) # arguments: (p-->AR, d, q-->MA)
                            forecast = model_fit_local.forecast(int(self.forecast_future/tank.dt))[0]

                            # reverse differentiating
                            for i in range(0,diff_count):
                                forecast = np.cumsum(forecast) + start_values[-i]
                                                        
                            self.pred_result_local = list(forecast)
                            self.model_update = 0 # no update in next timestep is needed

                            # lower limit that y needs to fall below to trigger fit of new model and forecast
                            self.model_lowlim = min(self.model_lowlim, data['y0_filt'][-1]+0.1*(forecast[-1]-data['y0_filt'][-1])) 

                            if self.write_data == 1:
                                self.t_pred_local = [data['t'][-1] + self.downsample_local*tank.dt*i for i in range(0, len(self.pred_result_local))]
                                data['arma_y'].append(self.pred_result_local)
                                data['t_arma_y'].append(self.t_pred_local)

                            # determine new setpoint, use forecast or actual value (conservative choice)
                            tank.dist_setpoint = max(2* tank.pi['0'].y_s0 - self.pred_result_local[-1], 2* tank.pi['0'].y_s0 - data['y0_filt'][-1], tank.pi['0'].y_s0, tank.dist_setpoint)
                            
                            if self.debug_print == 1:
                                print('arima fit succeeded at ' +str(tank.t))

                        except Exception as e:
                            if self.debug_print == 1:
                                print('arima fit failed at ' +str(tank.t))
                                print(e)
                            
                            if tank.dist_setpoint == None:
                                tank.dist_setpoint = tank.pi['0'].y_s0

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

                    except Exception as e:
                        
                        if self.debug_print == 1:
                            print('sarimax fit failed at ' +str(tank.t))
                            print(e)

                        self.enter_end = tank.t + 5
                
                    print('start next dist: '+str( self.t_start_next_dist))
                    if self.debug_print == 1:
                        print('sarimax fit succeeded at ' +str(tank.t))

                    #if (tank.liveplot == 1) & (tank.t > self.start_liveplot):
                    #    self.update_liveplot(tank, data)
            
            elif(tank.pdist[i] == 0):

                # reset before the start of a new disturbance
                self.model_update = 1
                self.model_lowlim = tank.pi[str(i)].y_s0
        
        if (self.t_start_next_dist != None) & (self.global_forecast==1):
            if (tank.t >= self.t_start_next_dist - 12.5) & (tank.t <= self.t_end_next_dist):
                tank.pdist_forecast = 1

                if self.debug_print == 2:
                    print('pdist_forecast at :' + str(tank.t))
                data['pdist_forecast'].append(1)
            else:
                tank.pdist_forecast = 0
                data['pdist_forecast'].append(0)

def analyse_model(diff, diff2):
    # analyse different models with respect to quality and time

    import pandas as pd
    import time
                        
    aic = []
    bic = []
    mod_p = []
    mod_q = []
    mod_d = []
    mod_dt = []

    d = [1, 2]
    p = [0, 1, 2, 3]
    q = [0, 1, 2, 3]

    for dd in d:
        for qq in q:
            for pp in p:
                try:
                    t0 = time.time()
                    if dd==1:
                        fit = ARIMA(diff, order=(pp,dd, qq)).fit(disp=0)
                    elif dd==2:
                        fit = ARIMA(diff2, order=(pp,dd, qq)).fit(disp=0)
                    dt = time.time() - t0

                    aic.append(fit.aic)
                    bic.append(fit.bic)
                    mod_p.append(pp)
                    mod_q.append(qq)
                    mod_d.append(dd)
                    mod_dt.append(dt)

                except Exception as e:
                    print(e)
                    aic.append(1000)
                    bic.append(1000)
                    mod_p.append(pp)
                    mod_q.append(qq)
                    mod_d.append(dd)
                    mod_dt.append(dt)

    data = {'dt': mod_dt, 'd': mod_d, 'p': mod_p, 'd': mod_d, 'q': mod_q, 'aic': aic, 'bic': bic}
    df = pd.DataFrame(data)    