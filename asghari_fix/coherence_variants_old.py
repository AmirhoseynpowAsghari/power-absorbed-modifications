"""The repo's (master) integrand with different sign choices in the coherence bracket   (python3 coherence_variants_old.py)

   W ∝ ∫ dE g(E) g(E+hw) [ 1 + X/(E' E'') + s*eta*Delta^2/(E' E'') ] [f(E+hw) - f(E)]     (E = band energy, f = normal-state Fermi function: the repo's framework)

   X = |xi||xi'|  or  xi xi'  (signed);   s = +1 (repo / paper's final formula)  or  s = -1 (what M = B(uu' - eta vv') gives)
   every variant is normalised by ITS OWN Delta -> 0 limit (as in the paper: W_N = W(Delta = 0)).
"""
import os, sys
import numpy as np
from scipy.integrate import trapezoid

ROOT = "/home/user/asghari_master/project/ultrasonic"
cwd0 = os.getcwd()
os.chdir(ROOT); sys.path.insert(0, ROOT)
import parameters_ua as PUA
from kgrid_ua import build_dispersions_ua
from dos_ua import compute_dos
from physics_ua import fermi_vec, Delta_T, find_mu
from sc_loader import load_sc_params, get_sc_params
from absorption import compute_ratio
os.chdir(cwd0)

VSO, eta = 0.4082, +1.0
E_grid = np.linspace(PUA.E_grid_min, PUA.E_grid_max, PUA.NE)
hw = PUA.hbar * PUA.omega
sc, source = load_sc_params(json_path=ROOT + "/../sc_params/results/tc_vs_vso.json")   # absolute path: stage-1 results of branch master
assert "fallback" not in str(source).lower() and "json" in str(source).lower(), source
Tc, Delta0 = get_sc_params(VSO, sc)
ep, em = build_dispersions_ua(VSO)
gE = compute_dos(E_grid, ep, em)
Ep_grid = E_grid + hw
gEp = np.interp(Ep_grid, E_grid, gE, left=0.0, right=0.0)
mask = np.abs(E_grid) > 1e-6


def integrand(T, mu, Delta, xi_mode, s):
    xi, xip = E_grid - mu, Ep_grid - mu
    E, Ep = np.sqrt(xi ** 2 + Delta ** 2), np.sqrt(xip ** 2 + Delta ** 2)
    E, Ep = np.where(E > 1e-6, E, 1e-6), np.where(Ep > 1e-6, Ep, 1e-6)
    X = np.abs(xi) * np.abs(xip) if xi_mode == "abs" else xi * xip
    br = 1.0 + X / (E * Ep) + (s * eta * Delta ** 2 / (E * Ep) if Delta > 1e-10 else 0.0)
    return gE * gEp * br * (fermi_vec(Ep_grid, T, mu) - fermi_vec(E_grid, T, mu)), br


variants = [("repo / paper:  |xi||xi'|, +eta D^2", "abs", +1), ("signed xi xi', +eta D^2", "signed", +1),
            ("|xi||xi'|, -eta D^2", "abs", -1), ("signed xi xi', -eta D^2   (correct bracket)", "signed", -1)]
print(f"V_SO = {VSO}, eta = +1, hbar*omega = {hw}, Tc = {Tc:.4f}, Delta_eff(0) = {Delta0:.4f}   (repo framework: band-energy grid, normal-state f)\n")
print("columns: W(variant) / W(Delta=0 of the SAME variant)   [and in brackets: / W_N of the paper (bracket = 2 for every pair)]")
print(f"{'T/Tc':>5} {'Delta':>7} | " + " | ".join(f"{v[0][:30]:>30}" for v in variants) + " | repo ratio")
rows = []
for frac in (0.1, 0.3, 0.5, 0.7, 0.8):
    T = frac * Tc
    Delta = Delta_T(T, Tc, Delta0)
    mu = find_mu(T, Delta, VSO)
    out, out2 = [], []
    I2, _ = integrand(T, mu, 0.0, "abs", +1)                             # the paper's W_N: bracket = 2 for every pair
    for name, xm, s in variants:
        I, br = integrand(T, mu, Delta, xm, s)
        I0, br0 = integrand(T, mu, 0.0, xm, s)
        out.append(abs(trapezoid(I[mask], E_grid[mask]) / trapezoid(I0[mask], E_grid[mask])))
        out2.append(abs(trapezoid(I[mask], E_grid[mask]) / trapezoid(I2[mask], E_grid[mask])))
    ref = compute_ratio(T, mu, Delta, E_grid, gE)                         # the repo's own function
    rows.append((frac, out, ref, out2))
    print(f"{frac:5.2f} {Delta:7.4f} | " + " | ".join(f"{o:8.4f} [{o2:7.4f}]".rjust(30) for o, o2 in zip(out, out2)) + f" | {ref:.5f}")
print(f"\nvariant 1 reproduces compute_ratio: max |difference| = {max(abs(r[1][0] - r[2]) for r in rows):.1e}")

# the bound on the bracket
worst = {}
for name, xm, s in variants:
    lo, hi = 9.0, -9.0
    for frac in (0.1, 0.5, 0.8):
        T = frac * Tc; Delta = Delta_T(T, Tc, Delta0); mu = find_mu(T, Delta, VSO)
        _, br = integrand(T, mu, Delta, xm, s)
        lo, hi = min(lo, br.min()), max(hi, br.max())
    worst[name] = (lo, hi)
print("\nrange of the bracket 1 + (...)/(E E') over the whole energy grid (the normal-state value is 2):")
for k, (lo, hi) in worst.items():
    print(f"   {k:46s}: [{lo:+.4f}, {hi:.4f}]")
