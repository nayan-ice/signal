import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz
import scipy.signal as sig

# Filter specifications
passband_edge = 2  # kHz
stopband_edge = 5  # kHz
fs = 20  # kHz
filter_length = 21

# Calculate filter parameters
nyquist = 0.5 * fs
passband_frequency = passband_edge / nyquist
stopband_frequency = stopband_edge / nyquist

# Design the filter using firwin with a Hanning window
taps = firwin(filter_length, stopband_frequency, window='hann')
w,h_freq=sig.freqz(taps,fs=fs)
z,p,k=sig.tf2zpk(taps,1)
# Frequency response of the filter
frequency_response = freqz(taps, worN=8000)

# Plot the frequency response
plt.figure(1)
plt.plot(0.5 * fs * frequency_response[0] / np.pi, np.abs(frequency_response[1]), 'b-', label='Filter response')
plt.title('FIR Filter Frequency Response')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Gain')

plt.figure(2)
plt.plot(w,np.unwrap(np.angle(h_freq)))


plt.figure(3)
plt.scatter(np.real(z),np.imag(z),marker='o',edgecolors='r')
plt.scatter(np.real(p),np.imag(p),marker='x',color='g')

plt.grid()
plt.show()
