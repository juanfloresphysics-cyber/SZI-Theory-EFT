"""
SZI Theory: Loop Quantum Gravity (LQG) Simulation
Repo: https://github.com/juanfloresphysics-cyber/SZI-Theory-EFT
Author: Juan Flores (with Grok assistance)
Date: October 26, 2025

Simulates LQG spin network in 14D with SZI modulation: area A = 8πγ l_P² ∑ j(j+1), volume V = (l_P³ / 6√2) ∑ [j(j+1)(j+2)]^(1/2),
modulated by SZI sequences (matter base 12 to ∞, logs to 0). Evolves under Hamiltonian constraint with QuTiP.
Dependencies: numpy, matplotlib, qutip
Run: python sz_i_lqg_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# Constants (LQG)
gamma = 0.2375  # Barbero-Immirzi parameter
l_P = 1.616e-35  # Planck length m
num_terms = 10  # Number of spin nodes/modes
j_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])  # Spin labels j

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

# Generate SZI sequences for modulation
matter = generate_matter_base12(num_terms)
logs = generate_log_sequence(num_terms)
primes = generate_primes(num_terms)
total_szi = matter + logs + primes
mod_factor = np.mean(total_szi) / 100  # Normalized modulation ~0.1

# LQG Operators: Area A = 8πγ l_P² ∑ j(j+1) * mod_SZI
area = 8 * np.pi * gamma * l_P**2 * np.sum(j_values * (j_values + 1)) * mod_factor

# Volume V = (l_P³ / 6√2) ∑ [j(j+1)(j+2)]^(1/2) * mod_SZI
volume_term = np.sqrt(j_values * (j_values + 1) * (j_values + 2))
volume = (l_P**3 / (6 * np.sqrt(2))) * np.sum(volume_term) * mod_factor

# Hamiltonian Constraint (simplified): H = H_geo + g_SZ ∑ λ_n j_x j_z (QuTiP for 2-spin example)
g_SZ = 0.1  # Coupling
lambda_n = np.mean(logs)  # Mean log modulation
j = 1.0  # Spin j=1 for demo
J_x = qt.jmat(j, '+')  # Simplified j_x for single j
J_z = qt.jmat(j, 'z')
H_geo = g_SZ * lambda_n * (J_x * J_z + J_z * J_x)  # Toy Hamiltonian

t_list = np.linspace(0, 10, 50)  # Planck time scaled
psi0 = qt.spin_coherent(j, theta=0, phi=0)  # Initial coherent state

result = qt.mesolve(H_geo, psi0, t_list, [], [qt.jmat(j, 'z')])  # Expectation <J_z>

entropy = [qt.entropy_vn(rho) for rho in result.states]  # Von Neumann entropy

# Output
print(f"LQG Area (modulated): {area:.2e} m²")
print(f"LQG Volume (modulated): {volume:.2e} m³")
print(f"Max Entropy: {max(entropy):.3f}")

# Plot: Area/Volume (bar) & Entropy evolution (line)
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(range(1, len(j_values)+1), j_values*(j_values+1) * mod_factor, color='gold', label='Area Term')
ax1.set_xlabel('Spin Node n')
ax1.set_ylabel('Area Contribution (scaled)', color='gold')
ax1.tick_params(axis='y', labelcolor='gold')

ax2 = ax1.twinx()
ax2.plot(t_list, entropy, 'b-', label='Entropy')
ax2.set_ylabel('Von Neumann Entropy', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('SZI LQG: Spin Network Area & Entropy Evolution')
fig.tight_layout()
plt.savefig('sz_i_lqg_simulation.png')
plt.show()
print("Plot saved as 'sz_i_lqg_simulation.png'.")
