"""
Symbolic check of the inversion  Delta(epsilon) -> Delta(E)
for the gap of Hutchinson-Hirsch-Marsiglio (arXiv:1801.08988), Eq. (37):

    Delta(eps) = D0 - Ds*eps/(4t)        E^2 = (eps-mu)^2 + Delta(eps)^2
"""
import sympy as sp

eps, E, mu = sp.symbols('epsilon E mu', real=True)
D0 = sp.Symbol('Delta0', real=True)
b = sp.Symbol('b', real=True)           # b = Delta_s/(4t)
DF = sp.Symbol('Delta_F', real=True)    # Delta_F = D0 - b*mu = Delta(eps = mu)

# ---- Steps 1-2 : E^2 as a quadratic polynomial in eps ------------------------
Delta_eps = D0 - b*eps
E2 = (eps - mu)**2 + Delta_eps**2
quad = sp.Poly(sp.expand(E2 - E**2), eps)
print("coefficients of  E^2(eps) - E^2 = 0  (descending powers of eps):")
for c in quad.all_coeffs():
    print("   ", sp.factor(c))

# ---- Step 3 : roots -------------------------------------------------------------
roots = sp.solve(sp.Eq(E2, E**2), eps)
print("\nroots eps_pm(E) (sympy):")
for r in roots:
    print("   ", sp.simplify(r))

# ---- Step 4 : claimed closed forms ------------------------------------------------
D = sp.sqrt((1 + b**2)*E**2 - DF**2)
xi_p = (b*DF + D)/(1 + b**2)           # xi = eps - mu, upper root
xi_m = (b*DF - D)/(1 + b**2)           # lower root
Dp = (DF - b*D)/(1 + b**2)             # Delta on the upper-root branch
Dm = (DF + b*D)/(1 + b**2)             # Delta on the lower-root branch

# D0 = DF + b*mu  (definition of Delta_F)
sub = {D0: DF + b*mu}
print("\nidentity checks (should print 0):")
print("  xi_+^2 + Delta_+^2 - E^2 =", sp.simplify(sp.expand(xi_p**2 + Dp**2 - E**2)))
print("  xi_-^2 + Delta_-^2 - E^2 =", sp.simplify(sp.expand(xi_m**2 + Dm**2 - E**2)))
print("  Delta_+ - Delta(eps=mu+xi_+) =", sp.simplify(Dp - Delta_eps.subs(eps, mu + xi_p).subs(sub)))
print("  Delta_- - Delta(eps=mu+xi_-) =", sp.simplify(Dm - Delta_eps.subs(eps, mu + xi_m).subs(sub)))

# ---- minimum excitation energy, Eqs. (39)-(40) of the paper -----------------------
eps_star = sp.solve(sp.diff(E2, eps), eps)[0]
E2min = sp.simplify(E2.subs(eps, eps_star))
print("\nlocation of the minimum  eps* =", sp.simplify(eps_star))
print("  difference to paper Eq.(39) (mu + D0*b)/(1+b^2):", sp.simplify(eps_star - (mu + D0*b)/(1 + b**2)))
print("E_min^2 =", sp.factor(E2min))
print("  difference to paper Eq.(40) squared (D0 - mu*b)^2/(1+b^2):", sp.simplify(E2min - (D0 - mu*b)**2/(1 + b**2)))

# ---- Jacobian d eps / dE on each branch ---------------------------------------------
print("\nJacobian:")
print("  d xi_+/dE =", sp.simplify(sp.diff(xi_p, E)))
print("  d xi_-/dE =", sp.simplify(sp.diff(xi_m, E)))

# ---- complete-the-square form ------------------------------------------------------
xi = sp.Symbol('xi', real=True)
xi_star = b*DF/(1 + b**2)
Emin2 = DF**2/(1 + b**2)
lhs = xi**2 + (DF - b*xi)**2
rhs = Emin2 + (1 + b**2)*(xi - xi_star)**2
print("\ncomplete-the-square identity  xi^2 + (DF - b xi)^2 - [Emin^2 + (1+b^2)(xi - xi*)^2] =",
      sp.simplify(sp.expand(lhs - rhs)))

# ---- large-E behaviour -----------------------------------------------------------
print("\nlarge-E expansion of Delta_+(E):", sp.series(Dp, E, sp.oo, 2))
