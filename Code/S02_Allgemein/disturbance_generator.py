import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
import random

def dist_generator(dist, t0, te, dt, tperiod = None, y0 = None, ye = None, slope= None, amp = None, offset = None):

    if dist == 'sin':
        t = np.linspace(t0, te, int((te-t0)/dt+1))
        y = -amp*np.sin((t-t0)*2*np.pi/tperiod) + offset

        data = [[t[i], y[i]] for i in range(0, len(t))]

    elif dist == 'cos':
        t = np.linspace(t0, te, int((te-t0)/dt+1))
        y = amp*np.cos((t-t0)*2*np.pi/tperiod) + offset

        data = [[t[i], y[i]] for i in range(0, len(t))]

    elif dist == 'sq sin':
        t = np.linspace(t0, te, int((te-t0)/dt+1))
        sinewave = np.sin((t-t0)*2*np.pi/tperiod)*(-1)

        # if excactly 0 negative sign
        sinewave[sinewave == 0] =-1

        y = np.sign(sinewave)*amp + offset

        data = [[t[i], y[i]] for i in range(0, len(t))]

    # keine negative Ventilöffung
    for i in data:
        if i[1] < 0:
            i[1] = 0

    #plt.plot(np.transpose(data)[0], np.transpose(data)[1])
    #plt.show()

    data[-1][1] = 1
    data[0][1] = 1

    return data

def triangle_generator(t0, te, dt, tperiod = None, amp = None):
    
    t = np.linspace(t0, te, int((te-t0)/dt+1))
    t_down = np.linspace(t0, t0+tperiod/2, int((tperiod/2)/dt+1))
    slope =  amp/(tperiod/2)

    y_down = [1-(i-t0)*slope for i in t_down]
    y_up = y_down[::-1]
    y = (y_down + y_up[1:-1]) * int((te-t0)/tperiod) + [1]

    data = [[t[i], y[i]] for i in range(0, len(t))]
    #plt.plot(np.transpose(data)[0], np.transpose(data)[1])
    #plt.show()
    return data

def chirp_generator(t0, te, dt, amp, offset, f0, t1, f1):

    t = np.linspace(t0, te-dt, int((te-t0)/dt))
    y_chirp = chirp(t-t0, f0, t1, f1)

    y = amp*y_chirp + offset

    # cut of after last value that is close to 1 in order to avoid jumps
    max_index = np.argwhere(y > 0.999)[-1][0]

    data = [[t[i], y[i]] for i in range(0, max_index+1)]

    
    data.append([te, 1])

    #plt.plot(np.transpose(data)[0], np.transpose(data)[1])
    #plt.show()
    
    data[0][1] = 1

    return data

def sin_var_amp_generator(t0, te, dt, amp0, ampe, off0, offe, tperiod):

    t = np.linspace(t0, te, int((te-t0)/dt+1))
    amp = np.linspace(amp0, ampe, int((te-t0)/dt+1))
    off = np.linspace(off0, offe, int((te-t0)/dt+1))
    y = np.add(np.multiply(amp, np.cos((t-t0)*2*np.pi/tperiod)), off)

    y[y>1] = 1
    y[y<0] = 0

    data = [[t[i], y[i]] for i in range(0, len(t))]

    data[-1][1] = 1
    data[0][1] = 1

    #plt.plot(np.transpose(data)[0], np.transpose(data)[1])
    #plt.show()
    return data

def random_walk_dist(t0, te, dt, step_size):
    t = np.linspace(t0, te, int((te-t0)/dt+1))
    y = [1]

    for i in range(len(t)-2):

        if (len(y) < 300) & (y[-1]>0.75):
            factor = 1.075
        elif (len(y) < 600) & (y[-1]>0.45):
            factor = 1.05
        elif (i>(len(t)-600)) & (y[-1]<0.45):
            factor = 0.975
        elif (i>(len(t)-300)) & (y[-1]<0.75):
            factor = 0.9
        else:
            factor = 1
        
        
        y_next = (y[-1]+random.uniform(-step_size*factor, step_size/factor))

        if y_next > 1:
            y_next = 1
        elif y_next < 0:
            y_next = 0
        
        y.append(y_next)
    y.append(1)

    data = [[t[i], y[i]] for i in range(0, len(t))]

    #plt.plot(np.transpose(data)[0], np.transpose(data)[1])
    #plt.show()

    data[-1][1] = 1
    data[0][1] = 1
    return data

def random_walk_dist_long(t0, te, dt, step_size):
    t = np.linspace(t0, te, int((te-t0)/dt+1))
    y = [1]

    for i in range(len(t)-2):

        if len(y) < 15:
            factor = 1.075
        elif i > (len(t)-50):
            factor = 0.85
        elif i > (len(t)-100):
            factor = 0.9
        elif i > (len(t)-250):
            factor = 0.925
        elif i > (len(t)-350):
            factor = 0.975
        elif (y[-1]>0.85):
            factor = 1.075
        elif y[-1] > 0.75:
            factor = 1.025
        elif y[-1] < 0.6:
            factor = 0.975
        elif y[-1] < 0.5:
            factor = 0.95            
        else:
            factor = 1
        
        
        y_next = (y[-1]+random.uniform(-step_size*factor, step_size/factor))

        if y_next > 1:
            y_next = 1
        elif y_next < 0:
            y_next = 0
        
        y.append(y_next)
    y.append(1)

    data = [[t[i], y[i]] for i in range(0, len(t))]

    #plt.plot(np.transpose(data)[0], np.transpose(data)[1])
    #plt.show()

    data[-1][1] = 1
    data[0][1] = 1
    return data

def noisy_cos(t0, te, dt, amp, offset, tperiod, noise):

    t = np.linspace(t0, te, int((te-t0)/dt+1))
    y = amp*np.cos((t-t0)*2*np.pi/tperiod) + offset

    y = [i + random.uniform(-1*noise, noise) for i in y]

    y = [0 if i<0 else i for i in y]
    y = [1 if i>1 else i for i in y]
    
    data = [[t[i], y[i]] for i in range(0, len(t))]

    plt.plot(np.transpose(data)[0], np.transpose(data)[1])
    plt.show()
    data[-1][1] = 1
    data[0][1] = 1

    return data

def moving_mean(t0, te, dt, N):

    t = np.linspace(t0, te, int((te-t0)/dt+1))
    noise = [random.random() for i in range(0, len(t)+N)]

    y = []
    for i in range(0, len(noise)-N):
        y.append(np.average(noise[i:i+N]))

    data = [[t[i], y[i]] for i in range(0, len(t))]

    data[-1][1] = 1
    data[0][1] = 1
    
    return data