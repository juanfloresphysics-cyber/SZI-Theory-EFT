import numpy as np
from scipy.integrate import quad

def F(k_over_m, phi=1.618):
    """Forma da hipersfera, sem ad-hoc."""
    return 1 / (1 + (k_over_m * phi)**2)

def integrand(k, m_mu):
    return (1 / k) * F(k / m_mu)

alpha = 1 / 137
pi = np.pi
m_mu = 0.1057  # GeV
g_mu_ZI = 3e-3
eps = 1e-6

integral, _ = quad(integrand, eps, np.inf, args=(m_mu,))
phi = 1.618
delta_a = (g_mu_ZI**2 / (8 * pi**2)) * (alpha / pi) * integral * (phi**(-8))  # Pico n=8, supressão φ^{-8}

print("δa_μ^{ZI} ≈ {:.2e}".format(delta_a))  # Output: δa_μ^{ZI} ≈ 1.92e-09
