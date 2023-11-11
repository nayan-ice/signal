import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

# Filter specifications
M = 32  # Filter length
fs = 1.0  # Sampling frequency

# Passband edge frequencies
fp1 = 0.2
fp2 = 0.35

# Stopband edge frequencies
fs1 = 0.1
fs2 = 0.425

# Calculate filter parameters
nyquist = 0.5 * fs
passband_edges = [fp1, fp2]
stopband_edges = [fs1, fs2]

# Design the bandpass filter using firwin
taps = firwin(M, passband_edges, fs=fs, pass_zero=False, window='hamming')

# Frequency response of the filter
frequency_response = freqz(taps, worN=8000, fs=fs)

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.plot(0.5 * fs * frequency_response[0] / np.pi, np.abs(frequency_response[1]), 'b-', label='Filter response')
plt.title('Bandpass Filter Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid()
plt.show()
