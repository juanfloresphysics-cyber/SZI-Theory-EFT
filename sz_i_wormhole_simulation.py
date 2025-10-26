"""
SZI Theory: Wormhole Simulation (Morris-Thorne Embedding)
Repo: https://github.com/juanfloresphysics-cyber/SZI-Theory-EFT
Author: Juan Flores (with Grok assistance)
Date: October 26, 2025

Simulates Morris-Thorne wormhole metric ds² = -dt² + dl² + (b² + l²) (dθ² + sin²θ dφ²),
embedded in 3D Flamm paraboloid: x = b arccosh(r/b), y=0, z=l.
Modulated by SZI sequences for duality zero-infinity.
Dependencies: numpy, matplotlib
Run: python sz_i_wormhole_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt

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
num_terms = 10  # For modulation
matter = generate_matter_base12(num_terms)
logs = generate_log_sequence(num_terms)
primes = generate_primes(num_terms)
total_szi = matter + logs + primes  # Total for modulation

# Wormhole Parameters
b = 1.0  # Throat radius (SZI unit)
l = np.linspace(-5, 5, 100)  # Proper distance along throat
r = np.sqrt(b**2 + l**2)  # Radial coordinate
x = b * np.arccosh(r / b)  # Embedding x (Flamm paraboloid)
y = np.zeros_like(l)  # y=0 for 2D section
z = l  # z = l

# Area of throat: 4π b²
throat_area = 4 * np.pi * b**2
print(f"Throat Area (π-modulated): {throat_area:.2f} SZI units")

# Modulation by SZI: Scale b by mean(total_szi) for duality
mod_factor = np.mean(total_szi) / 100  # Normalized modulation
x_mod = x * mod_factor  # SZI-dampened embedding

# Output first 5 x values
print("First 5 x (embedding):", x[:5])

# Plot: Wormhole Embedding x(z) symmetric, modulated
plt.figure(figsize=(8, 6))
plt.plot(z, x, 'b-', label='GR Baseline')
plt.plot(z, x_mod, 'r--', label='SZI Modulated')
plt.plot(z, -x, 'b-', alpha=0.5)  # Symmetric -x
plt.plot(z, -x_mod, 'r--', alpha=0.5)
plt.xlabel('z (Proper Distance)')
plt.ylabel('x (Embedding Coordinate)')
plt.title('SZI Wormhole: Morris-Thorne Embedding (π Throat)')
plt.legend()
plt.grid(True)
plt.savefig('sz_i_wormhole_simulation.png')
plt.show()
print("Plot saved as 'sz_i_wormhole_simulation.png'.")
