import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import changefinder
import ruptures as rpt
from scipy import signal

# Daten einlesen
path = r"C:\Users\timbr\OneDrive\Masterarbeit\Daten\2020_08_25_Rexer\\"
df70 = pd.read_csv(path+'kreisstrasse_70kmh.csv')
df100 = pd.read_csv(path+'bundesstrasse_100kmh.csv')
df130 = pd.read_csv(path+'autobahn_130kmh.csv')

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

"""
 Verläufe plotten
df70.plot('TIME', 'ACCELERATION in m/s^2')
df100.plot('TIME', 'ACCELERATION in m/s^2')
df130.plot('TIME', 'ACCELERATION in m/s^2')
plt.show()

df70.plot('TIME', 'VELOCITY in m/s')
df100.plot('TIME', 'VELOCITY in m/s')
df130.plot('TIME', 'VELOCITY in m/s')
plt.show()

df70.plot('TIME', 'DISPLACEMENT in m')
df100.plot('TIME', 'DISPLACEMENT in m')
df130.plot('TIME', 'DISPLACEMENT in m')
plt.show()"""

# extract reference signal for each group

ref70 = df70.loc[df70['TIME'] < 1, 'ACCELERATION in m/s^2']
ref70 = [i for i in ref70]
ref100 = df100.loc[df100['TIME'] < 1, 'ACCELERATION in m/s^2']
ref100 = [i for i in ref100]
ref130 = df130.loc[df130['TIME'] < 1, 'ACCELERATION in m/s^2']
ref130 = [i for i in ref130]

# create test signal

sig1 = df70.loc[(df70['TIME']>5) & (df70['TIME']<10), 'ACCELERATION in m/s^2']
sig1 = [i for i in sig1]
sig2 = df100.loc[(df100['TIME']>5) & (df100['TIME']<10), 'ACCELERATION in m/s^2']
sig2 = [i for i in sig2]
sig3 = df130.loc[(df130['TIME']>5) & (df130['TIME']<10), 'ACCELERATION in m/s^2']
sig3 = [i for i in sig3]

sig = sig1 + sig2 + sig3 + sig1 + sig2 + sig1 + sig2 + sig3 + sig1+ sig1 + sig2

points = np.array(sig[::15])

# RUPTURES PACKAGE

# Changepoint detection with the Pelt search method
model="rbf"
algo = rpt.Pelt(model=model, min_size=300).fit(points)
result = algo.predict(pen=1.5)
rpt.display(points, result, figsize=(10, 6))
plt.title('Change Point Detection: Pelt Search Method')
plt.show()

# Changepoint detection with filtered data

points = moving_average(sig, 15)[::15]

model="rbf"
algo = rpt.Pelt(model=model, min_size=300).fit(points)
result = algo.predict(pen=1)
rpt.display(points, result, figsize=(10, 6))
plt.title('Change Point Detection: Pelt Search Method')
plt.show()
    
# CHANGEFINDER PACKAGE
f, (ax1, ax2) = plt.subplots(2, 1)
f.subplots_adjust(hspace=0.4)
ax1.plot(points)
ax1.set_title("data point")

# Initiate changefinder function
cf = changefinder.ChangeFinder(smooth=40)
scores = [cf.update(p) for p in points]
ax2.plot(scores)
ax2.set_title("anomaly score")
plt.show()