# signal
# Q1 -- Find the spectrum of the following signal using FFT algorithom
![image](https://github.com/nayan-ice/signal/assets/149757661/60a7a2c5-b24f-4c4c-a514-871d2b264fd2)

```ruby
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np
fs = 200
k = np.arange(0,1,1/fs)
f = 0.25+2*np.sin(2*np.pi*5*k)+np.sin(2*np.pi*12.5*k)+1.5*np.sin(2*np.pi*20*k)+0.5*np.sin(2*np.pi*35*k)
# w = np.linspace(0, 2*np.pi, 1024, endpoint=False)
w = np.fft.fftfreq(len(f), d=1/fs)
F = np.fft.fft(f)

plt.figure(figsize=(10, 6))
# Original Signal
plt.subplot(3, 1, 1)
plt.plot(k, f)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Original Signal')

# Magnitude Spectrum
plt.subplot(3, 1, 2)
plt.plot(w, np.abs(F))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Magnitude Spectrum')
# Magnitude Spectrum
plt.subplot(3, 1,3)
plt.plot(w, np.unwrap(np.angle(F)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Magnitude Spectrum')

plt.tight_layout()
plt.show()
```
# Q2 ![image](https://github.com/nayan-ice/signal/assets/149757661/332eb4a4-e561-4870-9fbe-7953b1f80ef6)
```ruby
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
```
# Q3 ![image](https://github.com/nayan-ice/signal/assets/149757661/1305146e-bd0b-4f3a-91e7-61ddffc3665d)

```ruby
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
```

```ruby
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, lfilter

# Filter specifications
passband_edge = 1.5  # kHz
transition_width = 0.5  # kHz
fs = 10  # kHz
filter_length = 67

# Calculate filter parameters
nyquist = 0.5 * fs
cutoff_frequency = passband_edge / nyquist
width = transition_width / nyquist

# Design the filter using firwin with a Blackman window
taps = firwin(filter_length, cutoff_frequency, window='blackman', width=width)

# Frequency response of the filter
frequency_response = freqz(taps, worN=8000)

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.plot(0.5 * fs * frequency_response[0] / np.pi, np.abs(frequency_response[1]), 'b-', label='Filter response')
plt.title('Lowpass Filter Frequency Response')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Gain')
plt.grid()
plt.show()

# Calculate phase response of the filter
phase_response = np.angle(frequency_response[1])

# Plot the phase response
plt.figure(figsize=(10, 6))
plt.plot(0.5 * fs * frequency_response[0] / np.pi, phase_response, 'r-', label='Phase response')
plt.title('Lowpass Filter Phase Response')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Phase [radians]')
plt.grid()
plt.show()

# Plot the pole-zero diagram
zeros, poles, _ = lfilter(taps, 1.0, [1])
plt.figure(figsize=(8, 8))
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='b', label='Zeros')
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Poles')
plt.title('Pole-Zero Diagram')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.show()



```


# Q4 ![image](https://github.com/nayan-ice/signal/assets/149757661/5c8e3262-50be-4209-9a41-1b8b2dbcd008)
```ruby
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

```

# Q5 ![image](https://github.com/nayan-ice/signal/assets/149757661/d9026e02-f1a6-42d9-903f-2c0d8db2977a)
# 5(a)
```ruby
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


```
# 5(b)
```ruby
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tf2zpk

# Coefficients of the numerator and denominator
numerator_coeffs = [1, 1, 3/2,1/2]
denominator_coeffs = [1,3/2,1/2]

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

```

# Q9 ![image](https://github.com/nayan-ice/signal/assets/149757661/04c3bc0f-eb03-4ed8-9e1e-37410832ab00)
```ruby
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

```
# Q7 ![image](https://github.com/nayan-ice/signal/assets/149757661/a7b48bbd-dcac-4e51-bd9f-cdff513184fb)
# 7(a)
```ruby
import numpy as np
import matplotlib.pyplot as plt

# System transfer function coefficients
numerator = [1, 0, 0, 1]
denominator = [1, 2, 1, 0]

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

```
# 7(b)
```ruby
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
```
# Q8 ![image](https://github.com/nayan-ice/signal/assets/149757661/73a7a377-4573-4cda-9cdc-60206ff17e52)

```ruby
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


```
# Q9 ![image](https://github.com/nayan-ice/signal/assets/149757661/ebf7024e-8358-42b0-8ad3-944c1e89e581)
```ruby
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


```

# Q10 ![image](https://github.com/nayan-ice/signal/assets/149757661/d4b83d49-d34e-464e-8d40-689e90817a60)

```ruby
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


```



# 7(b) plot zero and poles different different charts
```ruby
import numpy as np
import matplotlib.pyplot as plt

# System transfer function coefficients
numerator = [10, 8, 4]
denominator = [20, 18, 8, 2]

# Calculate poles and zeros
zeros = np.roots(numerator)
poles = np.roots(denominator)

# Plot zeros
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='b')
plt.title('Zeros')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Plot poles
plt.subplot(1, 2, 2)
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r')
plt.title('Poles')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

# Display poles and zeros
print('Zeros:', zeros)
print('Poles:', poles)


```
# 7a   plot zero and poles different different charts
```ruby
import numpy as np
import matplotlib.pyplot as plt

# System transfer function coefficients
numerator = [1, 0, 0, 1]
denominator = [1, 2, 1, 0]

# Calculate poles and zeros
zeros = np.roots(numerator)
poles = np.roots(denominator)

# Plot Zeros
plt.figure(figsize=(8, 4))
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='b', label='Zeros')
plt.title('Zeros Plot')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Plot Poles
plt.figure(figsize=(8, 4))
plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Poles')
plt.title('Poles Plot')
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

```





