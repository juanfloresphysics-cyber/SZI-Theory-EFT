"""
SZI Theory: Hawking Radiation Simulation
Repo: https://github.com/juanfloresphysics-cyber/SZI-Theory-EFT
Author: Juan Flores (with Grok assistance)
Date: October 26, 2025

Simulates Hawking temperature T_H and spectrum P(ω) for BHs with masses M_n = φ_n M_⊙ modulated by SZI sequences.
Dependencies: numpy, matplotlib, astropy
Run: python sz_i_hawking_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import G, c, hbar, k_B, M_sun

def generate_matter_base12(num_terms=24):
    """SZI matter sequence base 12: 1,2,3,4,5,7,8,12,14,24... (duplication after 14)."""
    matter = [1, 2, 3, 4, 5, 7, 8, 12, 14]
    while len(matter) < num_terms:
        last = matter[-1]
        matter.append(last * 2)
    return np.array(matter)

def generate_log_sequence(num_terms=24):
    """Logarithmic sequence: 0.960,0.480,0.240... halving to 0."""
    logs = [0.960, 0.480, 0.240, 0.140, 0.120, 0.080, 0.070, 0.050, 0.040, 0.030, 0.020]
    while len(logs) < num_terms:
        logs.append(logs[-1] / 2)
    return np.array(logs[:num_terms])

def generate_primes(num_terms=24):
    """First num_terms primes."""
    primes = []
    num = 2
    while len(primes) < num_terms:
        is_prime = all(num % p != 0 for p in primes)
        if is_prime:
            primes.append(num)
        num += 1
    return np.array(primes)

# Parameters
num_terms = 10  # First 10 for demo
matter = generate_matter_base12(num_terms)
logs = generate_log_sequence(num_terms)
primes = generate_primes(num_terms)

# Masses M_n = matter * M_sun.value * (1 + logs) * (1 + primes / sum(primes))
sum_primes = np.sum(primes)
M_n = matter * M_sun.value * (1 + logs) * (1 + primes / sum_primes)

# Hawking Temperature T_H = (hbar.value * c.value**3 / (8 * np.pi * G.value * M_n * k_B.value)) * (1 + logs)
T_H = (hbar.value * c.value**3 / (8 * np.pi * G.value * M_n * k_B.value)) * (1 + logs)

# Spectrum P(ω) for n=1 (blackbody, ω scaled to T_H)
n = 1
omega = np.linspace(0.1, 10, 100) * T_H[n-1]
P_omega = (omega**3) / (np.exp(omega / T_H[n-1]) - 1)

# Output first 5 T_H
print("First 5 T_H (K):", T_H[:5])

# Plot: T_H vs n (left), Spectrum n=1 (right)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_terms+1), T_H, 'b-')
plt.yscale('log')
plt.xlabel('n (SZI Index)')
plt.ylabel('T_H (K)')
plt.title('SZI-Modulated Hawking Temperature')

plt.subplot(1, 2, 2)
plt.plot(omega / T_H[n-1], P_omega, 'r-')
plt.yscale('log')
plt.xlabel('ω / T_H')
plt.ylabel('P(ω)')
plt.title('Hawking Spectrum (n=1)')

plt.tight_layout()
plt.savefig('sz_i_hawking_simulation.png')
plt.show()
print("Plot saved as 'sz_i_hawking_simulation.png'.")
