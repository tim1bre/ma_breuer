import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from Code.S01_Experimente.Rexer.detect_streettype import detect_streettype

#from functools import partial
import changefinder
import ruptures as rpt
from scipy import signal

"""def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n"""

# read data
path = r"C:\Users\timbr\OneDrive\Masterarbeit\Daten\2020_08_25_Rexer\\"
df70 = pd.read_csv(path+'kreisstrasse_70kmh.csv')
df100 = pd.read_csv(path+'bundesstrasse_100kmh.csv')
df130 = pd.read_csv(path+'autobahn_130kmh.csv')

data_list = [df70, df100, df130]
data_names = [70, 100, 130]

ref_size_seconds = 5 # size of reference for DTW
downsample = 20

# extract reference signal for each group

ref_max1 = ref_size_seconds
ref_min2 = 19 - ref_size_seconds

ref1 = [ list(i.loc[i['TIME'] < ref_max1,'ACCELERATION in m/s^2']) for i in data_list]

ref2 = [ list(i.loc[(i['TIME'] > ref_min2) & (i['TIME'] < ref_min2 + ref_size_seconds),'ACCELERATION in m/s^2']) for i in data_list]
ref = ref1+ref2


this_detect = detect_streettype(ref, data_names+data_names, downsample)

# create test signal

batch_size_range = [5, 10]

signal = []
groundtrouth = []

for i in range(0, 8):

    idx = random.randint(0,2)
    df_choice = data_list[idx]
    name = data_names[idx]

    dt = random.uniform(5, 10)
    t0 = random.uniform(ref_max1, ref_min2-7)
    te = t0 + dt

    if te > ref_min2:
        te = ref_min2
    
    selection = list(df_choice.loc[(df_choice['TIME'] >= t0) & (df_choice['TIME'] <= te),'ACCELERATION in m/s^2'])
    signal += selection
    groundtrouth  += [name] * len(selection)

changepoints = this_detect.detect_cp_offline(signal)
print('changepoints detected')

classification, result_list = this_detect.classify(signal)
print("classification terminated")

# plot signal, groundtrouth and changepoints

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,4))
plt.subplots_adjust(hspace = 0.4)

t_groundtrouth = [i*0.001 for i in range(0, len(groundtrouth))]
ax[0].plot(t_groundtrouth, groundtrouth, color ='black', linestyle=':', linewidth=2)

t_result_list = [i*0.001 for i in range(0, len(result_list))]
ax[0].plot(t_result_list, result_list, color='gray')

t_signal = [i*0.001 for i in range(0, len(signal))]
ax[1].plot(t_signal, signal, color = 'black')

for i,j in zip(changepoints, classification):
    ax[1].axvspan(i*0.001-0.08, i*0.001+0.08, color='gray', alpha=0.3)

for i in range(0,2):
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

plt.sca(ax[0])
plt.yticks([70, 100, 130], ['Kreis', 'Bund', 'Auto'])

plt.yticks(fontname = "Verdana")
plt.xticks(fontname = "Verdana")
plt.savefig('Rexer3.svg')
plt.show()