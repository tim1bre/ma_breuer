import matplotlib.pyplot as plt
#from Code.S06_Quantifizieren.evaluate_all_metrices import evaluate_all
from Code.S02_Allgemein.save_object import load_object

def plot_only_sim(data, t0=100):
    
    fig1, axs1 = plt.subplots(5, 1, sharex=True)
    fig1.suptitle('Simulationsergebnisse Eintank')

    axs1[0].set(ylabel='Ventil')
    axs1[0].plot(data['t'], data['dist0'], 'r-')
    axs1[1].set(ylabel='Füllstand')
    axs1[1].plot(data['t'], data['y0'], 'r-')
    axs1[1].plot( [data['t'][0], data['t'][-1]], [0.2, 0.2], color="black")
    axs1[2].set(ylabel='Pdist')
    axs1[2].plot(data['t'], data['pdist0'], 'r-')
    axs1[3].set(ylabel='Sollwert')
    axs1[3].plot(data['t'], data['pi0'], 'r-')
    axs1[4].set(ylabel='Regler')
    axs1[4].plot(data['t'], data['u0'], 'r-')
    axs1[0].set_xlim(left=t0)

    plt.show()

def plot_only_sim_filename(mypath, filename):

    tanksys = load_object(mypath + '\\' + str(filename))
    data = tanksys.data
    
    fig1, axs1 = plt.subplots(4, 1, sharex=True)
    if 'model' in tanksys.detect.name:
        fig1.suptitle('noise: ' + str(tanksys.noise_sigma) + ', deadtime: ' + str(tanksys.deadtime) + ', mode: ' + str(tanksys.detect.name) + ' (qual: ' +  str(tanksys.detect.ref_sys.model_qual) + ')\n dist: ' + str(tanksys.dist_name))
    else:
        fig1.suptitle('noise: ' + str(tanksys.noise_sigma) + ', deadtime: ' + str(tanksys.deadtime) + ', mode: ' + str(tanksys.detect.name) + ')\n dist: ' + str(tanksys.dist_name))


    axs1[0].set(ylabel='Ventil')
    axs1[0].plot(data['t'], data['dist0'], 'r-')
    axs1[1].set(ylabel='Füllstand')
    axs1[1].plot(data['t'], data['y0'], 'r-')
    axs1[1].plot( [data['t'][0], data['t'][-1]], [20, 20], color="black")
    axs1[2].set(ylabel='Pdist')
    axs1[2].plot(data['t'], data['pdist0'], 'r-')
    axs1[3].set(ylabel='Sollwert')
    axs1[3].plot(data['t'], data['pi0'], 'r-')

    plt.show()

"""def plot_results_full(tanksys):

    if tanksys.tank_count > 1:
        print('nicht für Zweitank implementiert')

    # evaluate resulst
    data = evaluate_all(tanksys.data, 95)

    keys = ['false_pos_count', 'false_neg_count', 'time_to_detect', 'time_to_recover', 'yexceed+5%', 'resloss_sum', 'max_loss']
    for i in keys:
        print(str(i) + ': ' + str(data[i]))

    fig1, axs1 = plt.subplots(6, 1, sharex=True)

    if 'model' in tanksys.detect.name:
        fig1.suptitle('noise: ' + str(tanksys.noise_sigma) + ', deadtime: ' + str(tanksys.deadtime) + ', mode: ' + str(tanksys.detect.name) + ' (qual: ' +  str(tanksys.detect.ref_sys.model_qual) + ')\n dist: ' + str(tanksys.dist_name))
    else:
        fig1.suptitle('noise: ' + str(tanksys.noise_sigma) + ', deadtime: ' + str(tanksys.deadtime) + ', mode: ' + str(tanksys.detect.name) + ')\n dist: ' + str(tanksys.dist_name))

    axs1[0].set(ylabel='Ventil')
    axs1[0].plot(data['t'], data['dist0'], 'r-')
    axs1[1].set(ylabel='Füllstand')
    axs1[1].plot(data['t'], data['y0'], 'r-')
    axs1[1].set_ylim(18, 23)
    axs1[1].plot( [data['t'][0], data['t'][-1]], [20, 20], color="black")
    axs1[2].set(ylabel='Pdist')
    axs1[2].plot(data['t'], data['pdist0'], 'r-')
    axs1[3].set(ylabel='Sollwert')
    axs1[3].plot(data['t'], data['pi0'], 'r-')
    axs1[4].set(ylabel='Stellgröße')
    axs1[4].plot(data['t'], data['u0'], 'r-')
    axs1[5].set(ylabel='Res. Loss')
    axs1[5].plot(data['t'], data['resloss_akku'], 'r-')

    plt.show()"""
