import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# Filter specifications
fs = 10000  # sampling rate
N = 67  # order of filter
fc = 1500  # passband edge frequency
transition_width = 500  # transition width
window = 'blackman'  # window function

# Design the filter using the specified parameters
b = sig.firwin(N + 1, fc, fs=fs, window=window, pass_zero='lowpass', width=transition_width)

# Frequency response
w, h_freq = sig.freqz(b, fs=fs)

# Poles and Zeros
z, p, k = sig.tf2zpk(b, 1)

# Plotting
plt.figure(figsize=(10, 12))

# Magnitude Response
plt.subplot(3, 1, 1)
plt.plot(w, np.abs(h_freq))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Magnitude Response')

# Phase Response
plt.subplot(3, 1, 2)
plt.plot(w, np.unwrap(np.angle(h_freq)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.title('Phase Response')

# Pole-Zero Plot
plt.subplot(3, 1, 3)
plt.scatter(np.real(z), np.imag(z), marker='o', edgecolors='b', label='Zeros')
plt.scatter(np.real(p), np.imag(p), marker='x', color='b', label='Poles')
plt.legend()
plt.title('Pole-Zero Plot')
plt.xlabel('Real')
plt.ylabel('Imaginary')

plt.tight_layout()
plt.show()
