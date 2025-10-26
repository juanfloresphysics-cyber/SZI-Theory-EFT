"""
SZI Theory: Gravitational Wave Detection Simulation
Repo: https://github.com/juanfloresphysics-cyber/SZI-Theory-EFT
Author: Juan Flores (with Grok assistance)
Date: October 26, 2025

Simulates GW inspiral chirp for binary BHs (M1=M2=30 M_⊙, f0=35 Hz to f1=250 Hz, t=10 s) with SZI modulation δφ = α ∑ λ_n sin(2π total_SZI t / P).
Adds Gaussian noise (SNR~20), detects via matched filter correlation.
Dependencies: numpy, scipy, matplotlib
Run: python sz_i_gw_detection_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.integrate import cumtrapz

# SZI Sequence Generation
def generate_matter_base12(num_terms=24):
    matter = [1, 2, 3, 4, 5, 7, 8, 12, 14]
    while len(matter) < num_terms:
        last = matter[-1]
        matter.append(last * 2)
    return np.array(matter)

def generate_log_sequence(num_terms=24):
    logs = [0.960, 0.480, 0.240, 0.140, 0.120, 0.080, 0.070, 0.050, 0.040, 0.030, 0.020]
    while len(logs) < num_terms:
        logs.append(logs[-1] / 2)
    return np.array(logs[:num_terms])

def generate_primes(num_terms=24):
    primes = []
    num = 2
    while len(primes) < num_terms:
        is_prime = all(num % p != 0 for p in primes)
        if is_prime:
            primes.append(num)
        num += 1
    return np.array(primes)

# Parameters
num_terms = 10  # For modulation
matter = generate_matter_base12(num_terms)
logs = generate_log_sequence(num_terms)
primes = generate_primes(num_terms)
total_szi = matter + logs + primes  # Simplified total for sin modulation

# GW Simulation Parameters
fs = 4096  # Sampling rate Hz (LIGO-like)
t = np.linspace(0, 10, int(10 * fs))  # 10s signal
f0 = 35  # Initial frequency Hz
f1 = 250  # Merger frequency Hz
P = 5.0  # Merger period s
alpha = 0.05  # Deviation factor rad

# GR Baseline Phase: φ_GR(t) ≈ 2π ∫ f(t) dt, f(t) ~ (t_c - t)^{-3/8}
t_c = 10.0  # Coalescence time
f_gr = f0 * ((t_c - t) / (t_c - t[0])) ** (-3/8)
phi_gr = 2 * np.pi * cumtrapz(f_gr, t, initial=0)

# Amplitude h ~ f^{2/3}
h_gr = (f_gr / f0) ** (2/3) * np.cos(phi_gr)

# SZI Modulation: δφ = alpha * sum λ_n sin(2π total_SZI t / P)
mod_term = alpha * np.sum(logs[:num_terms]) * np.sin(2 * np.pi * np.mean(total_szi) * t / P)  # Simplified sum
phi_szi = phi_gr + mod_term
h_szi = (f_gr / f0) ** (2/3) * np.cos(phi_szi)

# Add Gaussian Noise (SNR ~20)
noise_std = np.std(h_gr) / 20  # Adjust for SNR
noise = np.random.normal(0, noise_std, len(t))
h_noisy = h_szi + noise

# Matched Filter Detection: Correlation with clean template h_gr
corr = correlate(h_noisy, h_gr, mode='full')
lags = np.arange(-len(t) + 1, len(t))
snr_peak = np.max(corr) / np.sqrt(np.sum(h_gr**2))

# Output
print(f"SNR Peak: {snr_peak:.2f} (detection threshold ~8)")
print(f"Phase Deviation δφ max: {np.max(mod_term):.3f} rad (~{np.max(mod_term)*180/np.pi:.1f} deg)")

# Plot: Noisy Signal vs Clean, Correlation Peak
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, h_gr, 'r--', label='Clean GR Template')
plt.plot(t, h_noisy, 'b-', label='Noisy SZI Signal')
plt.xlabel('Time (s)')
plt.ylabel('Strain h')
plt.title('SZI GW Detection: Noisy Chirp vs Template')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(lags / fs, corr, 'g-')
plt.axvline(0, color='k', linestyle='--', label='Lag 0 (Peak)')
plt.xlabel('Lag (s)')
plt.ylabel('Correlation')
plt.title(f'Matched Filter: SNR Peak = {snr_peak:.2f}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('sz_i_gw_detection_simulation.png')
plt.show()
print("Plot saved as 'sz_i_gw_detection_simulation.png'.")
