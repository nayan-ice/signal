import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

# Filter specifications
M = 61  # Filter length
fs = 1.0  # Sampling frequency

# Passband edge frequency
fp = 0.1

# Calculate filter parameters
nyquist = 0.5 * fs
cutoff_frequency = fp / nyquist

# Design the lowpass filter using firwin
taps = firwin(M, cutoff_frequency, fs=fs, window='hamming')

# Frequency response of the filter
frequency_response = freqz(taps, worN=8000, fs=fs)

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.plot(0.5 * fs * frequency_response[0] / np.pi, np.abs(frequency_response[1]), 'b-', label='Filter response')
plt.title('Lowpass Filter Frequency Response')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid()
plt.show()
