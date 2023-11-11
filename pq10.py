import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

# Step 1: Create the signal
fs = 100  # Sampling rate
t = np.arange(0, 1, 1/fs)  # Time vector
s = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 15 * t) + np.sin(2 * np.pi * 30 * t)

# Plot the original signal
plt.figure(figsize=(10, 4))
plt.plot(t, s)
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# Step 2: Design an IIR filter
def butter_bandstop_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y

# Step 3: Apply the IIR filter to suppress frequencies of 5 Hz and 30 Hz
filtered_signal = butter_bandstop_filter(s, 5, 30, fs)

# Plot the filtered signal
plt.figure(figsize=(10, 4))
plt.plot(t, filtered_signal)
plt.title('Filtered Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
