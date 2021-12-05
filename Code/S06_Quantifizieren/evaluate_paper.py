import numpy as np
from Code.S06_Quantifizieren.metrices_paper import res_triang, res_triang_noise, mttr, max_loss, ttd, false_pos, false_neg, disq

def evaluate_metrices_paper(tanksys, t_start=100):

    tanksys.metrices = {}
    t_dist_0 = 100
    t_dist_e = 500

    res_triang(tanksys, t_start)
    res_triang_noise(tanksys, t_start)
    mttr(tanksys, t_dist_0, t_dist_e)
    max_loss(tanksys, t_dist_0, t_dist_e)
    ttd(tanksys, t_dist_0, t_dist_e)
    false_pos(tanksys, t_dist_0, t_dist_e)
    false_neg(tanksys, t_dist_0, t_dist_e)
    disq(tanksys, t_dist_0)
    tanksys.metrices['sim_time'] = tanksys.sim_time

def metrices_to_hdf5(tanksys):

    evaluate_metrices_paper(tanksys)
    # attributes
    dict_attr = {}
    dict_attr['dist_id'] = tanksys.dist_id
    dict_attr['dist_name'] = tanksys.dist_name
    dict_attr['noise_sigma'] = tanksys.noise_sigma
    dict_attr['deadtime'] = tanksys.deadtime
    dict_attr['detect_mode'] = tanksys.detect.name
    dict_attr['react_mode'] = tanksys.react.name
    dict_attr['A'] = tanksys.A
    dict_attr['q'] = tanksys.q
    if 'model' in tanksys.detect.name:
        dict_attr['model_qual'] = tanksys.detect.ref_sys.model_qual
    else:
        dict_attr['model_qual'] = np.nan

    # metrices
    dict_metr = {}
    for col in tanksys.metrices.keys():
        dict_metr[col] = tanksys.metrices[col]

    # raw data
    data = ['t', 'y0', 'y0_filt', 'pdist0', 'pi0', 'u0']
    dict_data = {}
    for col in data:
        dict_data[col] = tanksys.data[col]

    return [dict_attr, dict_data, dict_metr]

