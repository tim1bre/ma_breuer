import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def fft(data):
    sp = np.fft.fft(data['y0'])
    freqs = np.fft.fftfreq(len(sp))
    frate = 1/0.05
    freq_in_rad_s = abs(freqs * frate * 2 * np.pi)
    plt.plot(freq_in_rad_s, sp)
    plt.show()

def low_pass(data):
    num, den = signal.butter(4, 3, 'low', analog=False, fs=100)
    y = signal.filtfilt(num, den, data)

    return y
