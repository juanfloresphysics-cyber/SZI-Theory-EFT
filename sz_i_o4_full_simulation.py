"""
SZI Theory: O4 Gravitational Wave Full Data Simulation
Repo: https://github.com/juanfloresphysics-cyber/SZI-Theory-EFT
Author: Juan Flores (with Grok assistance)
Date: October 27, 2025

Simulates 10 O4-like GW events (GWTC-4 proxy, M1=M2=30 M_⊙, f0=35 to f1=250 Hz, t=10 s, fs=4096 Hz) with SZI modulation δφ = 0.05 sin(2π mean(total_SZI) t / P).
Adds Gaussian noise (SNR~18.5 avg), matched filter detection.
Dependencies: numpy, scipy, matplotlib
Run: python sz_i_o4_full_simulation.py
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
total_szi = matter + logs + primes

# O4 GW Parameters (10 events proxy)
num_events = 10
fs = 4096  # Hz
t_duration = 10  # s
t = np.linspace(0, t_duration, int(t_duration * fs))
f0 = 35  # Initial Hz
f1 = 250  # Merger Hz
P = 5.0  # Period s
alpha = 0.05  # Deviation rad
t_c = t_duration  # Coalescence time

# Generate GR baseline for one event (template)
f_gr = f0 * ((t_c - t) / (t_c - t[0])) ** (-3/8)
phi_gr = 2 * np.pi * cumtrapz(f_gr, t, initial=0)
h_gr = (f_gr / f0) ** (2/3) * np.cos(phi_gr)

# SZI Modulation for events
mod_term = alpha * np.sum(logs[:num_terms]) * np.sin(2 * np.pi * np.mean(total_szi) * t / P)
phi_szi = phi_gr + mod_term
h_szi = (f_gr / f0) ** (2/3) * np.cos(phi_szi)

# Add Gaussian Noise for 10 events (SNR ~18.5 avg)
snr_events = np.random.normal(18.5, 1.0, num_events)  # Vary SNR
noise_std = np.std(h_gr) / snr_events.mean()  # Base std
h_events = []
lags_list = []
snr_peaks = []
delta_phis = []

for i in range(num_events):
    noise = np.random.normal(0, noise_std, len(t))
    h_noisy = h_szi + noise
    corr = correlate(h_noisy, h_gr, mode='full')
    lags = np.arange(-len(t) + 1, len(t))
    snr_peak = np.max(corr) / np.sqrt(np.sum(h_gr**2))
    delta_phi_max = np.max(mod_term)  # Per event proxy
    h_events.append(h_noisy)
    lags_list.append(lags)
    snr_peaks.append(snr_peak)
    delta_phis.append(delta_phi_max)

# Output
print("O4 Full Proxy Metrics (10 Events):")
for i in range(num_events):
    print(f"Event {i+1}: SNR = {snr_peaks[i]:.2f}, δφ max = {delta_phis[i]:.3f} rad")

print(f"Average SNR: {np.mean(snr_peaks):.2f}, Average Lag: {np.mean(delta_phis):.3f} rad")

# Plot: Sample Event 1 Signal vs Template, Correlation
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, h_gr, 'r--', label='GR Template')
plt.plot(t, h_events[0], 'b-', label='Noisy SZI Event 1')
plt.xlabel('Time (s)')
plt.ylabel('Strain h')
plt.title('SZI O4 Full: Sample Event 1 Noisy Chirp vs Template')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
corr = correlate(h_events[0], h_gr, mode='full')
plt.plot(lags_list[0] / fs, corr, 'g-')
plt.axvline(0, color='k', linestyle='--', label='Lag 0 Peak')
plt.xlabel('Lag (s)')
plt.ylabel('Correlation')
plt.title(f'Matched Filter: SNR Peak = {snr_peaks[0]:.2f}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('sz_i_o4_full_sample_event.png')
plt.show()
print("Plot saved as 'sz_i_o4_full_sample_event.png'.")
