import numpy as np
import matplotlib.pyplot as plt

# System characteristics
a = 0.9  # constant in the sequence

# Frequency range
omega = np.linspace(0, 2 * np.pi, 1000, endpoint=False)

# Frequency response calculation
H = np.sum((a**np.arange(0, 1000)) * np.exp(-1j * np.outer(omega, np.arange(0, 1000))), axis=1)

# Plot the magnitude response
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(omega, np.abs(H))
plt.title('Magnitude Response')
plt.xlabel('Frequency (radians/sample)')
plt.ylabel('Magnitude')

# Plot the real part of the frequency response
plt.subplot(2, 2, 2)
plt.plot(omega, np.real(H))
plt.title('Real Part of Frequency Response')
plt.xlabel('Frequency (radians/sample)')
plt.ylabel('Real Part')

# Plot the imaginary part of the frequency response
plt.subplot(2, 2, 3)
plt.plot(omega, np.imag(H))
plt.title('Imaginary Part of Frequency Response')
plt.xlabel('Frequency (radians/sample)')
plt.ylabel('Imaginary Part')

# Plot the phase response
plt.subplot(2, 2, 4)
plt.plot(omega, np.angle(H))
plt.title('Phase Response')
plt.xlabel('Frequency (radians/sample)')
plt.ylabel('Phase (radians)')

plt.tight_layout()
plt.show()
