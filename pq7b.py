import numpy as np
import matplotlib.pyplot as plt

# System transfer function coefficients
numerator = [10,  8, 4]
denominator = [20, 18, 8, 2]

# Calculate poles and zeros
zeros = np.roots(numerator)
poles = np.roots(denominator)

# Plot poles and zeros using scatter
plt.figure(figsize=(8, 8))
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='b', label='Zeros')
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Poles')
plt.title('Pole-Zero Map')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Display poles and zeros
print('Zeros:', zeros)
print('Poles:', poles)