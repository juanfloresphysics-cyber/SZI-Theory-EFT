import sympy as sp
from sympy.physics.quantum import Dagger  # Para termos quânticos, mas simplificado aqui

# Símbolos
mu_sym, Phi, H, psi_bar, psi, y_phi, i, gamma_mu, D_mu = sp.symbols('mu Phi H psi_bar psi y_phi i gamma^mu D_mu')
m_ZI, lambda_h, H_u, H_d, theta = sp.symbols('m_ZI lambda_h H_u H_d theta', real=True)
Phi_bar = sp.symbols(r'\bar{\Phi}')  # Para Dirac

# Termo cinético simplificado (não diff literal; usa notação padrão)
kinetic = sp.symbols(r'|D_\mu \Phi|^2')  # Placeholder para (∂_μ Φ*) Φ, etc.

# Termo de massa e portal
mass_portal = - (m_ZI**2 / 2) * sp.Abs(Phi)**2 - lambda_h * sp.Abs(Phi)**2 * sp.Abs(H)**2

# Termo de Yukawa
yukawa = psi_bar * (sp.symbols(r'i \gamma^\mu D_\mu') - y_phi * Phi) * psi

# Lagrangian total (SM + extensão ZI)
L = kinetic + mass_portal + yukawa

# Superpotencial (integral d²θ W)
W = sp.symbols('mu') * H_u * H_d + lambda_h * Phi * H_u * H_d + sp.Rational(1, 41) * Phi**3
superpot = sp.Integral(sp.Integral(W, (theta, 0, sp.pi)), (theta, 0, sp.pi))  # d²θ approx como ∫ dθ²

# Subs lambda_h = 1/41 e print LaTeX
full_L = L + superpot.subs(lambda_h, sp.Rational(1, 41))
print(sp.latex(full_L, mode='equation'))  # Output: LaTeX da L completa pra EFT/ML
