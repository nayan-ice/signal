import numpy as np
import matplotlib.pyplot as plt

# Generate the sinusoidal signal
f = 10 # signal frequency in Hz
fs = 50 # sampling frequency in Hz
t = np.arange(0, 1, 1/fs)
x = np.sin(2*np.pi*f*t)

# Plot the original signal
plt.subplot(3,1,1)
plt.plot(t, x)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Signal')

# Sample the signal
Ts = 1/fs  # Sampling interval (in seconds)
n = np.arange(0, 1, Ts)
xn = np.sin(2*np.pi*f*n)

# Plot the sampled signal
plt.subplot(3,1,2)
plt.stem(n, xn)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sampled Signal')

# Reconstruct the analog signal using ideal reconstruction
xr = np.zeros_like(t)  # Initialize the reconstructed signal
for i in range(len(n)):
    xr += xn[i] * np.sinc((t - i*Ts) / Ts)

# Plot the reconstructed signal
plt.subplot(3,1,3)
plt.plot(t, xr)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Reconstructed Signal')

plt.tight_layout()
plt.show()