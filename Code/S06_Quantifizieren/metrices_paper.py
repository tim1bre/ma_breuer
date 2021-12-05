import numpy as np
import matplotlib.pyplot as plt

def res_triang(tanksys, t_start):
    """ resilance triangle """

    i_start = t_start/ tanksys.dt
    error = np.subtract(tanksys.pi['0'].y_s0, tanksys.data['y0_filt'])
    error[error<0] = 0

    tanksys.metrices['res_triang'] = sum(error[int(i_start):])

def res_triang_noise(tanksys, t_start):
    """ resilance triangle with lower threshold to avoid influence of noise"""

    i_start = t_start/ tanksys.dt
    error = np.subtract(tanksys.pi['0'].y_s0 - 2*tanksys.noise_sigma, tanksys.data['y0_filt'])
    error[error<0] = 0

    tanksys.metrices['res_triang_noise'] = sum(error[int(i_start):])

def mttr(tanksys, t_dist_0=100, t_dist_e=500):
    """ mean time to recovery """

    y_lim = tanksys.pi['0'].y_s0 - 2*tanksys.noise_sigma # height that needs to be surpassed to count as recovery
    try:
        idx_rec = np.argwhere((np.array(tanksys.data['y0_filt']) <= y_lim) & (np.array(tanksys.data['t']) > t_dist_0) & (np.array(tanksys.data['t']) < t_dist_e))
        mttr = tanksys.data['t'][idx_rec[-1][0]] - t_dist_0 + tanksys.dt
    except:
        mttr = np.nan
        
    tanksys.metrices['mttr'] = mttr

def max_loss(tanksys, t_dist_0=100, t_dist_e=500):
    """ maximal loss due to disturbance """

    loss = np.subtract(tanksys.pi['0'].y_s0, tanksys.data['y0_filt'])
    max_loss = np.max(loss[int(t_dist_0/tanksys.dt):int(t_dist_e/tanksys.dt)])

    tanksys.metrices['max_loss'] = max_loss

def ttd(tanksys, t_dist_0=100, t_dist_e=500):
    """ time to detect disturbance """

    idx_ttd = np.argwhere((np.array(tanksys.data['pdist0']) > 0) & (np.array(tanksys.data['t']) > t_dist_0) & (np.array(tanksys.data['t']) < t_dist_e))
    if len(idx_ttd) > 0:
        ttd = tanksys.data['t'][idx_ttd[0][0]] - t_dist_0
    else:
        ttd = np.nan

    tanksys.metrices['ttd'] = ttd

def false_pos(tanksys, t_dist0=100, t_diste=500, t_start=100):

    i_start = int(t_start/ tanksys.dt)
    i_dist0 = int(t_dist0/ tanksys.dt)
    i_diste = int(t_diste/ tanksys.dt) 

    i_detect = [i[0]+1 for i in np.argwhere(np.diff(tanksys.data['pdist0']) > 0.5)]
    fp = len(np.argwhere((np.array(i_detect) > i_start) & ((np.array(i_detect) < i_dist0) | (np.array(i_detect) > i_diste))))
     
    tanksys.metrices['fp'] = fp
    
def false_neg(tanksys, t_dist0=100, t_diste=500):

    i_dist0 = int(t_dist0/ tanksys.dt)
    i_diste = int(t_diste/ tanksys.dt) 

    i_pos = [i[0]+1 for i in np.argwhere(np.array(tanksys.data['pdist0']) > 0)]
    
    if len(np.argwhere((np.array(i_pos) > i_dist0) & (np.array(i_pos) < i_diste))) == 0:
        fn = 1
    else:
        fn = 0

    tanksys.metrices['fn'] = fn

def disq(tanksys, t_dist0):

    i_dist0 = int(t_dist0/ tanksys.dt)
    y_lim = tanksys.pi['0'].y_s0 + 2*tanksys.noise_sigma # height that should not be surpassed
    
    if (tanksys.data['y0_filt'][i_dist0-1] <= y_lim) & (tanksys.data['pdist0'][i_dist0-1] == 0):
        disq = 0
    else:
        disq = 1
     
    tanksys.metrices['disq'] = disq