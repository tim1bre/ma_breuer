import numpy as np
from Code.S03_Erkennen.detect_dtw_mean import detect_dtw_mean
from Code.S03_Erkennen.setpoint_algo import setpoint_algo

class detect_dtw_mean_algo(detect_dtw_mean, setpoint_algo):
    
    def __init__(self, dtw_bin_seconds = 3, down_sample = 2):
        self.name = 'dtw mean algo'
        self.timespan = 'full'
        
        self.t_start_ref = 70 # start of reference signal
        self.dtw_bin_seconds = dtw_bin_seconds # bin size for reference signal in seconds
        self.down_sample = down_sample

        self.t0_get_thres = 70 # start time for interval to determine model threshold
        self.te_get_thres = 100 # end time for interval to determine model threshold
        
        self.distance_list = []
        self.first_call = 0
        self.ref_on = 0 