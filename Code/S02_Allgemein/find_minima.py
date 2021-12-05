from scipy.signal import find_peaks
import numpy as np

def find_minima(y_movingav_list):
    """ find minimas in a signal """
    
    ind_lmin = find_peaks([-1*i for i in y_movingav_list], prominence=0.005)[0]
    lmin = [y_movingav_list[i] for i in ind_lmin]

    return lmin