import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from IPython import display

from Code.S02_Allgemein.save_object import load_object

# read data
path = r"C:\Users\timbr\OneDrive\Masterarbeit\Daten\2020_10_14_Daten_fuer_Kolloquium\\"
filename = 'implentierte_methoden_PELTmitKl2020_10_14__12_05_41__8278.pkl'
tanksys1 = load_object(path + filename)

# select data for animation
data1 = {}
data1['dist'] = tanksys1.data['pdist0']
data1['dist_t'] = tanksys1.data['t']
data1['data'] = tanksys1.detect.anim_points
data1['t_data'] = tanksys1.detect.anim_points_time
data1['cp'] = tanksys1.detect.anim_changepoints
data1['t_adjust'] = tanksys1.detect.anim_t_adjust
data1['y0_filt'] = tanksys1.data['y0_filt']
data1['t'] = tanksys1.detect.anim_timespamp
t = tanksys1.detect.anim_timespamp

n = len(t)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=45, metadata=dict(artist='Tim Breuer'), bitrate=1800)

# create figure
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 4.43))


def animate(i=t, data1=data1):
    
    #fig.suptitle('ZEIT: '+str(round(t[i],1)) +' s')
    fig.suptitle(str(round(t[i],1)))
    
    # get index corrsponding to current timestep for supplementary animation data
    m = t[i]
    ind = data1['t'].index(m)

    # get indexes of simulation data
    idx0 = data1['dist_t'].index(data1['t_data'][ind][0])
    idxe = data1['dist_t'].index(data1['t_data'][ind][-1])
    
    # disturbance probability
    animate.line1 = ax[0].clear()
    ax[0].set_ylim([-0.1, 1.1])
    #animate.line1 = ax[0].set(ylabel='Störungerkennung')
    animate.line1 = ax[0].plot(data1['dist_t'][idx0:idxe], data1['dist'][idx0:idxe], 'k-')
    
    # water height
    animate.line2 = ax[1].clear()
    ax[1].set_ylim([0.985, 1.03])
    #animate.line2 = ax[1].set(ylabel='Füllstand')
    animate.line2, = ax[1].plot(data1['t_data'][ind], np.divide(data1['data'][ind], 20), 'k-')

    # draw lines for changepoints
    for i in data1['cp'][ind][:-1]:

        color = '#004e73'
        ax[1].axvspan(i-0.125, i+0.125, facecolor=color, alpha=0.5)

    # draw observation window
    if len(data1['cp'][ind]) > 1:
        ax[1].axvspan(data1['cp'][ind][-2], data1['t'][ind], facecolor='silver', alpha=0.5)

        idx_cp = np.where(data1['t_data'][ind] == min(data1['t_data'][ind], key=lambda x:abs(x-data1['cp'][ind][-2])))[0][0]
        
        y_min = min(data1['data'][ind][idx_cp:])
        t_y_min = data1['t_data'][ind][np.where(data1['data'][ind] == y_min)[0][0]]

        y_max = max(data1['data'][ind][idx_cp:])
        t_y_max = data1['t_data'][ind][np.where(data1['data'][ind] == y_max)[0][0]]

        ax[1].plot(t_y_min, y_min/20, 'x', color='#004e73', markersize=9.5)
        ax[1].plot(t_y_max, y_max/20, 'x', color='#004e73', markersize=9.5)

    plot_num = 2
    for i in range(0,plot_num):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].yaxis.set_ticks_position('left')
        ax[i].xaxis.set_ticks_position('bottom')
        ax[i].yaxis.set_tick_params(direction="out", width=1)
        ax[i].xaxis.set_tick_params(direction="out", width=1)

        for tick in ax[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(11)
        for tick in ax[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(11)

    plt.yticks(fontname = "Verdana")
    plt.xticks(fontname = "Verdana")
       
anim = animation.FuncAnimation(fig, animate, repeat=False, blit=False, frames=n, interval=1)

anim.save('PELT_mit_Klassifizierung.mp4', writer=writer)

plt.show()