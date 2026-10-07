"""Does the phase s*exp(-i theta_k) of Delta_ks = s e^{-i theta_k} Dbar_ks obstruct the energy-space treatment?   (python3 check_phase_energy_space.py, ~10 s)

  1. spectrum:   E_ks from the COMPLEX gap Delta_ks versus E(eps_ks) built from the real Dbar(eps) -> identical
  2. matrix element: exact spin-space BdG |M|^2 (gap matrix built in SPIN space, complex phases and all) versus
        (a) |<ks|k's'>|^2 * (1/2)[1 + xi xi'/EE' - Dbar Dbar'/EE']                   real SIGNED Dbar            <- what the energy-space code uses
        (b) the same with |Dbar| |Dbar'|                                              sign dropped
        (c) the same with Re(Delta_ks Delta*_k's') = ss' Dbar Dbar' cos(theta-theta')  phase of the gap kept, helicity-vertex phase dropped
"""
import numpy as np
import helicity_matrix_element as hme
from solve_gap import Model
from qp_absorption import sc_state_vs_T

VSO = 0.4082
m = Model(U=95.0, dt=4.5, n=1.885, VSO=VSO, Nk=1200)
Tc, Ts, D0, Ds, mu, muN = sc_state_vs_T(m, [0.7])
i = 1
hme.P.update(VSO=VSO, D0=float(D0[i]), Ds=float(Ds[i]), mu=float(mu[i]))
P = hme.P
print(f"state: V_SO = {VSO}, T/Tc = 0.7:  Delta0 = {P['D0']:.5f}, Delta_s = {P['Ds']:.5f}, mu = {P['mu']:.5f}\n")

# ---- 1. spectrum -------------------------------------------------------------------------------------------------------
N = 120
k = (np.arange(N) + 0.5) * 2 * np.pi / N - np.pi
kx, ky = np.meshgrid(k, k, indexing="ij")
R = np.hypot(np.sin(kx), np.sin(ky))
e_theta = (np.sin(ky) - 1j * np.sin(kx)) / R                       # exp(i theta_k)
worst, phases = 0.0, []
for s in (+1, -1):
    eps = -2 * (np.cos(kx) + np.cos(ky)) - 2 * s * VSO * R
    Dbar = P["D0"] - P["Ds"] * eps / 4
    Delta = s * np.conj(e_theta) * Dbar                              # complex gap  s e^{-i theta} Dbar
    E_complex = np.sqrt((eps - P["mu"]) ** 2 + np.abs(Delta) ** 2)   # paper Eq. (26)
    E_of_eps = np.hypot(eps - P["mu"], P["D0"] - P["Ds"] * eps / 4)  # a function of eps only
    worst = max(worst, np.abs(E_complex - E_of_eps).max())
    phases.append(np.angle(Delta))
phases = np.concatenate([p.ravel() for p in phases])
print(f"1. spectrum: the complex gap Delta_ks has phases spread over [{phases.min():+.2f}, {phases.max():+.2f}] rad, yet\n"
      f"   max |E_ks(complex Delta) - E(eps_ks)| over {2 * N * N} states = {worst:.1e}   (E depends on (k,s) only through eps_ks)\n")

# ---- 2. matrix element -------------------------------------------------------------------------------------------------
rng = np.random.default_rng(5)
rows = []
for _ in range(3000):
    kk, kp = hme.random_k(rng), hme.random_k(rng)
    s, sp = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
    exact = abs(hme.M_numeric(kk, s, kp, sp, P, hme.vertex_scalar)) ** 2          # exact BdG, B = identity in spin space (eta = +1)
    u, v, E, xi, Db = hme.uv(kk, s, P)
    up, vp, Ep, xip, Dbp = hme.uv(kp, sp, P)
    ov = abs(np.vdot(hme.spinor(kk, s), hme.spinor(kp, sp))) ** 2
    base = 1 + xi * xip / (E * Ep)
    a = ov * 0.5 * (base - Db * Dbp / (E * Ep))
    b = ov * 0.5 * (base - abs(Db) * abs(Dbp) / (E * Ep))
    Dc, Dcp = s * np.conj(hme.phase(kk)) * Db, sp * np.conj(hme.phase(kp)) * Dbp
    c = ov * 0.5 * (base - np.real(Dc * np.conj(Dcp)) / (E * Ep))
    rows.append((exact, a, b, c, Db * Dbp < 0))
rows = np.array(rows, float)
sel = rows[:, 0] > 1e-4 * rows[:, 0].max()
print(f"2. |M|^2 for {int(sel.sum())} random pairs (k,s) -> (k',s') with non-negligible weight (exact spin-space BdG, scalar vertex):")
for name, col in (("(a) signed real Dbar        ", 1), ("(b) |Dbar|, sign dropped    ", 2), ("(c) complex phase kept       ", 3)):
    rel = np.abs(rows[sel, col] - rows[sel, 0]) / rows[sel, 0]
    print(f"   {name}: max relative error {rel.max():.1e}   median {np.median(rel):.1e}")
neg = sel & (rows[:, 4] > 0)
rel_b = np.abs(rows[neg, 2] - rows[neg, 0]) / rows[neg, 0]
print(f"   share of pairs with Dbar*Dbar' < 0 (across the node): {neg.sum() / sel.sum():.2f};  error of (b) on those pairs: median {np.median(rel_b):.2f}")
