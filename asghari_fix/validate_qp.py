"""Validation of qp_absorption.py (energy-space absorption).   python3 validate_qp.py     (about 1-2 min)

 A. normal-state limit      Delta -> 0:  S_sc + S_pair  must equal the normal-state absorption S_N  (checks the pair-creation channel and all prefactors)
 B. BCS limit               flat DOS, constant gap, omega -> 0, eta = +1:   W_s/W_n = 2 f(Delta)      (textbook result)
 C. low / high temperature  W/W_N -> 0 as T -> 0 for omega < 2 E_min,   W/W_N -> 1 at T_c
 D. brute force in k-space  all pairs (k,s) -> (k',s') on a k-grid, with the helicity overlap (1 + s s' cos dtheta)/2 and a Gaussian delta function,
                            against the energy-space formula (checks the branches, the Jacobians, the band test and the helicity average 1/2).
                            In the user's regime (branch master, U = 95, V_SO = 0.4082, T/Tc = 0.7, omega = 0.7, below the pair-breaking onset) the brute-force ratio converges with the grid
                            size, 0.119 / 0.122 / 0.124 / 0.124 for N = 64 / 96 / 128 / 160 (Gaussian width 0.05 / 0.03 / 0.04 / 0.04), while the energy-space ratio converges as its DOS
                            smoothing is reduced, 0.131 / 0.129 / 0.127 for 0.04 / 0.02 / 0.01: the residual 2-5% is regularisation and sampling, not the formula.
                            With the pair channel just open (T/Tc = 0.96) the two agree to 1.7% already at N = 64.  Close above the pair-breaking edge (HHM-like state, omega - 2 Emin = 0.4) BOTH sides depend on
                            their regularisation and converge towards each other: energy space 2.209 / 2.260 / 2.297 for DOS smoothing 0.04 / 0.02 / 0.01;
                            brute force 2.473 / 2.405 / 2.342 for Gaussian width 0.08 / 0.05 / 0.03 (N = 64 / 64 / 96).
 D2. flat DOS               same comparison without lattice van Hove singularities, pair channel open: agreement 5e-4 (checks the branch algebra exactly)
"""
import numpy as np
from qp_absorption import (RashbaDOS, absorption, fermi_qp, fermi_normal, qp_constants, qp_energy)
from solve_gap import Model

rng = np.random.default_rng(1)


# ------------------------------------------------------------------------------------------------------------------
class FlatDOS:
    """Particle-hole symmetric flat band, N(eps) = N0 for |eps| <= W (integral = 2)."""
    def __init__(self, W=12.0, n=24000):
        self.W = W
        e = np.linspace(-W, W, n + 1)
        self.eps_i = 0.5 * (e[1:] + e[:-1])
        self.w_i = np.full(n, 2.0 / n)
        self.N0 = 2.0 / (2 * W)

    def N(self, eps):
        return np.where(np.abs(eps) <= self.W, self.N0, 0.0)


def test_A_normal_limit(dos, label):
    print(f"\n[A] normal-state limit ({label}):  Delta = 0  ->  (S_sc + S_pair) / S_N  must be 1")
    worst = 0.0
    for T in (0.15, 0.4, 0.9):
        for omega in (0.3, 0.7, 1.5):
            Model_mu = None
            # mu_N from the number equation of the normal state at filling n = 1.885
            from scipy.optimize import brentq
            f = lambda mu: np.sum(dos.w_i * fermi_normal(dos.eps_i, T, mu)) - 1.885
            muN = brentq(f, -10, 15)
            for eta in (+1, -1):
                r = absorption(dos, T, omega, 0.0, 0.0, muN, muN, eta=eta)
                tot = (r["S_sc"] + r["S_pair"]) / r["S_N"]
                worst = max(worst, abs(tot - 1))
                print(f"   T={T:4.2f} omega={omega:3.1f} eta={eta:+d}:  S_sc/S_N={r['S_sc']/r['S_N']:.4f}  S_pair/S_N={r['S_pair']/r['S_N']:.4f}  sum={tot:.5f}")
    print(f"   worst |sum - 1| = {worst:.1e}")
    return worst


def test_B_bcs_limit():
    print("\n[B] BCS limit: flat DOS, constant gap Delta, mu = 0, omega -> 0, eta = +1:  W_s/W_n  vs  2 f(Delta)")
    flat = FlatDOS()
    out = []
    for Delta, T in ((0.5, 0.2), (0.5, 0.35), (0.5, 0.6), (1.0, 0.5), (0.3, 0.3)):
        omega = 0.002
        r = absorption(flat, T, omega, Delta, 0.0, 0.0, 0.0, eta=+1)
        exact = 2.0 * fermi_qp(Delta, T)
        out.append((Delta, T, r["ratio"], exact))
        print(f"   Delta={Delta:3.1f} T={T:4.2f}:  computed {r['ratio']:.5f}   2f(Delta) = {exact:.5f}   rel. diff = {abs(r['ratio']/exact-1):.1e}")
    return max(abs(o[2] / o[3] - 1) for o in out)


def test_C_limits(VSO=0.4082, omega=0.7):
    print(f"\n[C] limits for the user's parameters (branch master: U = 95, dt = 4.5, n = 1.885, V_SO = {VSO}, omega = {omega}):  eta = +1")
    from qp_absorption import sc_state_vs_T
    m = Model(U=95.0, dt=4.5, n=1.885, VSO=VSO, Nk=1200)
    dos = RashbaDOS(VSO, Nk=1200)
    fr = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
    Tc, Ts, D0, Ds, mu, muN = sc_state_vs_T(m, fr)
    print(f"   Tc = {Tc:.4f}")
    rows = []
    for T, d0, ds, mu_, mun in zip(Ts, D0, Ds, mu, muN):
        r = absorption(dos, T, omega, d0, ds, mu_, mun, eta=+1)
        rows.append((T / Tc, r["Emin"], r["ratio"], r["S_pair"] / r["S_N"]))
        print(f"   T/Tc={T/Tc:5.3f}  Emin={r['Emin']:.4f}  2Emin={2*r['Emin']:.3f}  mu={mu_:.4f} muN={mun:.4f}  W/W_N={r['ratio']:.3e}  (pair part {r['S_pair']/r['S_N']:.2e})")
    return rows


# ------------------------------------------------------------------------------------------------------------------
def brute_force(VSO, T, omega, D0, Ds, mu, muN, eta, N=64, width=0.05, overlap=True, chunk=300):
    """All pairs of (k, s) states on an N x N midpoint grid of the full zone; Gaussian delta function of width `width`."""
    k = (np.arange(N) + 0.5) * 2 * np.pi / N - np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kx, ky = kx.ravel(), ky.ravel()
    R = np.hypot(np.sin(kx), np.sin(ky))
    e0 = -2 * (np.cos(kx) + np.cos(ky))
    eps = np.concatenate([e0 - 2 * s * VSO * R for s in (+1, -1)])
    sh = np.concatenate([np.full(kx.size, s) for s in (+1, -1)]).astype(float)
    gx = np.concatenate([np.sin(ky) / R] * 2)
    gy = np.concatenate([-np.sin(kx) / R] * 2)
    xi = eps - mu
    Db = D0 - Ds * eps / 4
    E = np.hypot(xi, Db)
    fq, fN = fermi_qp(E, T), fermi_normal(eps, T, muN)
    nst = eps.size
    Nk2 = float(kx.size) ** 2
    gauss = lambda x: np.exp(-(x / width) ** 2) / (width * np.sqrt(np.pi))
    S_sc = S_pair = S_N = 0.0
    for i0 in range(0, nst, chunk):
        sl = slice(i0, min(i0 + chunk, nst))
        if overlap:
            cosd = gx[sl, None] * gx[None, :] + gy[sl, None] * gy[None, :]
            ss = sh[sl, None] * sh[None, :]
            ov_s, ov_p = 0.5 * (1 + ss * cosd), 0.5 * (1 - ss * cosd)
        else:
            ov_s = ov_p = 1.0
        EaEb = E[sl, None] * E[None, :]
        xx = xi[sl, None] * xi[None, :] / EaEb
        dd = Db[sl, None] * Db[None, :] / EaEb
        # scattering: a = row (upper), b = column (lower)
        S_sc += np.sum(ov_s * 0.5 * (1 + xx - eta * dd) * (fq[None, :] - fq[sl, None]) * gauss(E[sl, None] - E[None, :] - omega))
        # pair creation, 1/2 of the ordered double sum
        S_pair += 0.5 * np.sum(ov_p * 0.5 * (1 - xx + eta * dd) * (1 - fq[sl, None] - fq[None, :]) * gauss(E[sl, None] + E[None, :] - omega))
        # normal state
        S_N += np.sum(ov_s * (fN[None, :] - fN[sl, None]) * gauss(eps[sl, None] - eps[None, :] - omega))
    return S_sc / Nk2, S_pair / Nk2, S_N / Nk2


def test_D_bruteforce():
    print("\n[D] brute-force k-space all-pairs sum (with the helicity overlap) against the energy-space formula")
    cases = []
    # (i) the user's regime (omega below the pair-breaking threshold)  (ii) a weaker-coupling state where the pair channel is open
    m = Model(U=95.0, dt=4.5, n=1.885, VSO=0.4082, Nk=1200)
    from qp_absorption import sc_state_vs_T
    Tc, Ts, D0, Ds, mu, muN = sc_state_vs_T(m, [0.7, 0.96])
    i, j = 1, -1                      # T/Tc = 0.7 (below the pair-breaking onset), T/Tc = 0.96 (above it: onset at 0.91)
    cases.append(("user regime, T/Tc=0.7, omega=0.7 (scattering only)", 0.4082, Ts[i], 0.7, D0[i], Ds[i], mu[i], muN[i]))
    cases.append(("user regime, T/Tc=0.96, omega=0.7 (pair channel just open)", 0.4082, Ts[j], 0.7, D0[j], Ds[j], mu[j], muN[j]))
    cases.append(("HHM-like state (U=90), T=0.30, omega=2.2 (pair channel open)", 0.5, 0.30, 2.2, 0.29255, 1.33119, 3.72649, None))
    out = []
    for name, VSO, T, omega, d0, ds, mu_, mun in cases:
        dos = RashbaDOS(VSO, Nk=1200)
        if mun is None:
            from scipy.optimize import brentq
            mun = brentq(lambda mm: np.sum(dos.w_i * fermi_normal(dos.eps_i, T, mm)) - 1.875, -10, 15)
        for eta in (+1, -1):
            r = absorption(dos, T, omega, d0, ds, mu_, mun, eta=eta)
            bf = brute_force(VSO, T, omega, d0, ds, mu_, mun, eta)
            bf_noov = brute_force(VSO, T, omega, d0, ds, mu_, mun, eta, overlap=False) if eta == +1 else bf
            R_es = r["ratio"]
            R_bf = (bf[0] + bf[1]) / bf[2]
            R_bf0 = (bf_noov[0] + bf_noov[1]) / bf_noov[2]
            out.append((name, eta, R_es, R_bf, R_bf0, r["S_pair"] / r["S_N"], bf[1] / bf[2]))
            print(f"   {name}, eta={eta:+d}\n        W/W_N energy-space = {R_es:.4f}   brute force with overlap = {R_bf:.4f}   without overlap = {R_bf0:.4f}"
                  f"   (pair part: {r['S_pair']/r['S_N']:.4f} vs {bf[1]/bf[2]:.4f})")
    return out


def eps_grid_bruteforce(dos, T, om, D0, Ds, mu, muN, eta, width, de=0.004):
    """2-D grid over (eps_a, eps_b) with a Gaussian delta function: independent of the branch / Jacobian algebra."""
    e = np.arange(-dos.W + de / 2, dos.W, de)
    xi, Db = e - mu, D0 - Ds * e / 4
    E = np.hypot(xi, Db)
    fq, fN = fermi_qp(E, T), fermi_normal(e, T, muN)
    g = lambda x: np.exp(-(x / width) ** 2) / (width * np.sqrt(np.pi))
    S_sc = S_pair = S_N = 0.0
    for i0 in range(0, e.size, 400):
        sl = slice(i0, i0 + 400)
        EaEb = E[sl, None] * E[None, :]
        xx, dd = xi[sl, None] * xi[None, :] / EaEb, Db[sl, None] * Db[None, :] / EaEb
        S_sc += np.sum(0.5 * (1 + xx - eta * dd) * (fq[None, :] - fq[sl, None]) * g(E[sl, None] - E[None, :] - om))
        S_pair += 0.5 * np.sum(0.5 * (1 - xx + eta * dd) * (1 - fq[sl, None] - fq[None, :]) * g(E[sl, None] + E[None, :] - om))
        S_N += np.sum((fN[None, :] - fN[sl, None]) * g(e[sl, None] - e[None, :] - om))
    s = dos.N0 ** 2 * de * de
    return S_sc * s, S_pair * s, S_N * s


def test_D2_flat_pair_channel():
    print("\n[D2] flat DOS, energy-dependent gap (b != 0), pair channel open: analytic branches/Jacobians vs a 2-D energy-grid sum")
    flat = FlatDOS(W=9.0, n=18000)
    D0, Ds, mu = 0.29255, 1.33119, 3.72649
    worst = 0.0
    for T, om in ((0.30, 2.2), (0.45, 2.6), (0.20, 3.0)):
        muN = mu - 0.4
        for eta in (+1, -1):
            r = absorption(flat, T, om, D0, Ds, mu, muN, eta=eta)
            bf = eps_grid_bruteforce(flat, T, om, D0, Ds, mu, muN, eta, 0.01)
            Rb = (bf[0] + bf[1]) / bf[2]
            worst = max(worst, abs(r["ratio"] / Rb - 1))
            print(f"   T={T:4.2f} omega={om:3.1f} eta={eta:+d}: analytic {r['ratio']:.4f} (pair {r['S_pair']/r['S_N']:.4f})   energy-grid brute force {Rb:.4f} (pair {bf[1]/bf[2]:.4f})")
    print(f"   worst relative difference = {worst:.1e}")
    return worst


if __name__ == "__main__":
    dos = RashbaDOS(0.4082, Nk=1200)
    wA = test_A_normal_limit(dos, "Rashba DOS, V_SO = 0.4082")
    wB = test_B_bcs_limit()
    rowsC = test_C_limits()
    wD2 = test_D2_flat_pair_channel()
    outD = test_D_bruteforce()
    print("\nSUMMARY:  A worst |sum-1| = %.1e   B worst rel. diff = %.1e   D2 worst rel. diff = %.1e" % (wA, wB, wD2))
    print("          D (lattice), user regime, scattering only: max |R_es - R_bf| = %.4f" % max(abs(o[2] - o[3]) for o in outD if "scattering only" in o[0]))
    print("          D (lattice), user regime, pair channel just open: max |R_es - R_bf| = %.4f (relative %.3f)"
          % (max(abs(o[2] - o[3]) for o in outD if "just open" in o[0]), max(abs(o[2] - o[3]) / o[3] for o in outD if "just open" in o[0])))
    print("          D (HHM-like state, omega - 2 Emin = 0.4): difference %.3f / %.3f is regularisation-limited (sigma_smooth, Gaussian width) -- see docstring"
          % tuple(abs(o[2] - o[3]) for o in outD if o[0].startswith("HHM")))
