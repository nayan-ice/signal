import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tf2zpk

# Coefficients of the numerator and denominator
numerator_coeffs = [1, 0, 1]
denominator_coeffs = [2, 1, -0.5, 0.25]

# Get the zeros, poles, and gain
zeros, poles, gain = tf2zpk(numerator_coeffs, denominator_coeffs)

# Plot the pole-zero diagram
plt.figure(figsize=(8, 8))
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='b', label='Zeros')
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Poles',s=100)
plt.title('Pole-Zero Diagram')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()