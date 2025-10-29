"""
SZI Theory: ER=EPR Quantum Wormhole Traversal in 2 Temporal Dimensions
Repo: https://github.com/juanfloresphysics-cyber/SZI-Theory-EFT
Author: Juan Flores (with Grok assistance)
Date: October 27, 2025

Simulates entangled qubits (|00> + |11>) traversing ER-like wormhole in 2 temporal dimensions:
H_t1 (causal): σ_x1 σ_z2, H_t2 (quantum): σ_z1 σ_x2, g=1.0 modulated by λ_mod~0.5.
Computes fidelity and von Neumann entropy over t=0 to π (scaled time).
Dependencies: numpy, qutip, matplotlib
Run: python sz_i_er_epr_2_temporal_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

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
total_szi = matter + logs + primes
lambda_mod = np.mean(logs)  # Mean log modulation ~0.5

g = 1.0  # Coupling
t_list = np.linspace(0, np.pi, 50)  # Scaled traversal time

# Qubits: σ_x1 = sigmax() ⊗ qeye(2), σ_z2 = qeye(2) ⊗ sigmaz()
sx1 = qt.tensor(qt.sigmax(), qt.qeye(2))
sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())
sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
sx2 = qt.tensor(qt.qeye(2), qt.sigmax())

# Hamiltonians: H_t1 (causal time): g * λ_mod * sx1 * sz2
H_t1 = g * lambda_mod * (sx1 * sz2)

# H_t2 (quantum time): g * λ_mod * sz1 * sx2
H_t2 = g * lambda_mod * (sz1 * sx2)

# Combined H for dual temporal evolution (alternating or sum; sum for simplicity)
H_combined = H_t1 + H_t2

# Initial Bell state |00> + |11> / √2
psi0 = (qt.basis(4, 0) + qt.basis(4, 3)).unit()

# Time evolution for t1 (causal)
result_t1 = qt.mesolve(H_t1, psi0, t_list, [], [qt.qeye(4)])

# Time evolution for t2 (quantum)
result_t2 = qt.mesolve(H_t2, psi0, t_list, [], [qt.qeye(4)])

# Combined evolution
result_combined = qt.mesolve(H_combined, psi0, t_list, [], [qt.qeye(4)])

# Fidelity to initial state for each
fidelity_t1 = [abs((psi0.overlap(psi)).data[0,0])**2 for psi in result_t1.states]
fidelity_t2 = [abs((psi0.overlap(psi)).data[0,0])**2 for psi in result_t2.states]
fidelity_combined = [abs((psi0.overlap(psi)).data[0,0])**2 for psi in result_combined.states]

# Von Neumann entropy for each (full system)
entropy_t1 = [qt.entropy_vn(psi) for psi in result_t1.states]
entropy_t2 = [qt.entropy_vn(psi) for psi in result_t2.states]
entropy_combined = [qt.entropy_vn(psi) for psi in result_combined.states]

# Output
print("Final Fidelity t1 (Causal): {:.3f}".format(fidelity_t1[-1]))
print("Final Fidelity t2 (Quantum): {:.3f}".format(fidelity_t2[-1]))
print("Final Fidelity Combined: {:.3f}".format(fidelity_combined[-1]))
print("Max Entropy t1: {:.3f}".format(max(entropy_t1)))
print("Max Entropy t2: {:.3f}".format(max(entropy_t2)))
print("Max Entropy Combined: {:.3f}".format(max(entropy_combined)))

# Plot: Fidelity & Entropy vs t
plt.figure(figsize=(10, 6))
plt.plot(t_list, fidelity_t1, 'b-', label='Fidelity t1 (Causal)')
plt.plot(t_list, fidelity_t2, 'r-', label='Fidelity t2 (Quantum)')
plt.plot(t_list, fidelity_combined, 'g-', label='Fidelity Combined')
plt.xlabel('Time t (scaled)')
plt.ylabel('Fidelity')
plt.title('SZI ER=EPR in 2 Temporal Dimensions: Fidelity Evolution')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(t_list, entropy_t1, 'b-', label='Entropy t1 (Causal)')
plt.plot(t_list, entropy_t2, 'r-', label='Entropy t2 (Quantum)')
plt.plot(t_list, entropy_combined, 'g-', label='Entropy Combined')
plt.xlabel('Time t (scaled)')
plt.ylabel('Von Neumann Entropy')
plt.title('SZI ER=EPR in 2 Temporal Dimensions: Entropy Evolution')
plt.legend()
plt.grid(True)

plt.savefig('sz_i_er_epr_2_temporal_simulation.png')
plt.show()
print("Plot saved as 'sz_i_er_epr_2_temporal_simulation.png'.")
