"""
Energy-space quasiparticle absorption for the Rashba-Hubbard superconductor (dimensionless: t = hbar = k_B = 1).

Replacement for  project/ultrasonic/absorption.py  (and absorption_u.py).

WHAT IS COMPUTED
----------------
Golden rule with a constant matrix element B (the paper's approximation: the sums over k and k' are unrestricted,
i.e. momentum is not conserved), summed over BOTH helicity bands, for the quasiparticle (Bogoliubov) states
a = (k, s) with energy

        E_a = sqrt( xi_a^2 + Dbar_a^2 ),   xi_a = eps_a - mu,   Dbar_a = D0 - Ds * eps_a / (4 t)      (HHM Eqs. 26, 37)

All k-dependence enters through eps_a only, so every k-sum is a one-dimensional integral over the band energy eps
with the normal-state density of states N(eps) (sum_s -> N(eps), integral of N = 2).

 (1) quasiparticle SCATTERING, a = upper state (E_a = E_b + omega), b = lower state:

        S_sc = sum_{a,b} C_ab [f(E_b) - f(E_a)] delta(E_a - E_b - omega),      C_ab = (1/2)[1 + xi_a xi_b/(E_a E_b) - eta Dbar_a Dbar_b/(E_a E_b)]

 (2) quasiparticle PAIR CREATION (only for omega >= 2 E_min), b = the lower-energy member of the pair:

        S_pair = sum_{a,b; E_b < E_a} Cp_ab [1 - f(E_a) - f(E_b)] delta(E_a + E_b - omega),   Cp_ab = (1/2)[1 - xi_a xi_b/(E_a E_b) + eta Dbar_a Dbar_b/(E_a E_b)]

 f(E) = 1/(exp(E/T) + 1)  with E measured from the condensate: NO chemical potential in the quasiparticle occupation.

 eta = +1: time-reversal-even perturbation (ultrasound, case I);  eta = -1: odd (electromagnetic, case II).

 NORMAL STATE (same model, Delta = 0):   S_N = sum_{a,b} [f_N(eps_b) - f_N(eps_a)] delta(eps_a - eps_b - omega),   f_N with mu_N(T) at the same filling.

 RATIO:  W/W_N = (S_sc + S_pair) / S_N        (all prefactors 2*pi*omega*B^2*N_k^2 cancel)

HOW THE DELTA FUNCTION IS DONE (the energy-space step, k-dependence kept)
-------------------------------------------------------------------------
 E(eps)^2 = Emin^2 + (1 + b^2)(eps - eps*)^2,   b = Ds/4t.   Hence for a given E >= Emin there are TWO band energies,
        eps_pm(E) = eps* +- sqrt(E^2 - Emin^2)/sqrt(1 + b^2),     |d eps_pm / dE| = E / ( sqrt(1+b^2) sqrt(E^2 - Emin^2) ),
 both of which must lie inside the band (|eps| <= W, the band test).  The state b is summed over the DOS (a sum over bins);
 the delta function fixes E_a, and the sum over the two roots eps_pm(E_a) of the state a carries the Jacobian above.
 No singularity appears in the integrand because E_a >= Emin + omega (scattering) or E_a >= omega/2 (pair creation).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ----------------------------------------------------------------------------------------------------------------
# normal-state density of states of the two Rashba helicity bands
# ----------------------------------------------------------------------------------------------------------------
class RashbaDOS:
    """eps_ks = -2t(cos kx + cos ky) -+ 2 V_SO sqrt(sin^2 kx + sin^2 ky) on a midpoint k-grid of one quarter of the zone.

    eps_i, w_i : grid of band energies and weights w_i = N(eps_i) d_eps (sum w_i = 2), used to sum over the state b
    N(eps)     : smoothed density of states (integral = 2), exactly zero outside the band |eps| <= W  (the band test)
    """

    def __init__(self, VSO, Nk=1600, nbins=4000, smooth=0.02, t=1.0):
        self.VSO, self.t = VSO, t
        self.W = 4.0 * t * np.sqrt(1.0 + VSO ** 2 / (2.0 * t ** 2))
        edges = np.linspace(-self.W - 1e-9, self.W + 1e-9, nbins + 1)
        k = (np.arange(Nk) + 0.5) * np.pi / Nk
        cnt = np.zeros(nbins)
        sm = np.zeros(nbins)
        for i0 in range(0, Nk, 100):
            kx = k[i0:i0 + 100, None]
            e0 = -2 * t * (np.cos(kx) + np.cos(k[None, :]))
            R = np.sqrt(np.sin(kx) ** 2 + np.sin(k[None, :]) ** 2)
            for s in (+1, -1):
                e = (e0 - 2 * s * VSO * R).ravel()
                idx = np.clip(np.searchsorted(edges, e) - 1, 0, nbins - 1)
                cnt += np.bincount(idx, minlength=nbins)
                sm += np.bincount(idx, weights=e, minlength=nbins)
        de = edges[1] - edges[0]
        self.grid = 0.5 * (edges[1:] + edges[:-1])
        raw = cnt / (Nk ** 2 * de)
        self.dens = gaussian_filter1d(raw, smooth / de, mode="constant")
        self.dens *= 2.0 / (np.sum(self.dens) * de)                      # renormalise: integral exactly 2
        # the SAME smoothed density is used for the state b (sum over grid points) and for the state a (interpolation),
        # so that sum rules such as  Delta -> 0  hold to the accuracy of the quadrature
        keep = np.abs(self.grid) <= self.W
        self.eps_i = self.grid[keep]
        self.w_i = self.dens[keep] * de

    def N(self, eps):
        eps = np.asarray(eps, float)
        out = np.interp(eps, self.grid, self.dens, left=0.0, right=0.0)
        return np.where(np.abs(eps) <= self.W, out, 0.0)                 # band test


# ----------------------------------------------------------------------------------------------------------------
# occupations
# ----------------------------------------------------------------------------------------------------------------
def fermi_qp(E, T):
    """Occupation of a Bogoliubov quasiparticle of energy E >= 0 (measured from the condensate): NO mu."""
    return 1.0 / (np.exp(np.clip(np.asarray(E, float) / T, -700, 700)) + 1.0)


def fermi_normal(eps, T, mu):
    """Normal-state occupation of a band state."""
    return 1.0 / (np.exp(np.clip((np.asarray(eps, float) - mu) / T, -700, 700)) + 1.0)


# ----------------------------------------------------------------------------------------------------------------
# quasiparticle spectrum as a function of the band energy
# ----------------------------------------------------------------------------------------------------------------
def qp_constants(D0, Ds, mu, t=1.0):
    b = Ds / (4.0 * t)
    DF = D0 - b * mu                                       # gap at the Fermi level
    s = np.sqrt(1.0 + b * b)
    return dict(b=b, DF=DF, s=s, Emin=abs(DF) / s, estar=(mu + b * D0) / (1.0 + b * b))


def qp_energy(eps, D0, Ds, mu, t=1.0):
    return np.hypot(eps - mu, D0 - Ds * eps / (4.0 * t))


def branches(E, P):
    """The two band energies eps_pm(E) of the roots of E(eps) = E  and  ell = sqrt(E^2 - Emin^2).   E >= Emin."""
    ell = np.sqrt(np.maximum(E ** 2 - P["Emin"] ** 2, 0.0))
    return P["estar"] + ell / P["s"], P["estar"] - ell / P["s"], ell


# ----------------------------------------------------------------------------------------------------------------
# the absorption integrals
# ----------------------------------------------------------------------------------------------------------------
def absorption(dos, T, omega, D0, Ds, mu, muN, eta=+1, t=1.0, diag=False):
    """Returns dict(S_sc, S_pair, S_N, ratio, Emin).   W/W_N = (S_sc + S_pair)/S_N.
    diag=True adds the internal arrays of the scattering term under the key "internals" (for verification only)."""
    P = qp_constants(D0, Ds, mu, t)
    eb, wb = dos.eps_i, dos.w_i
    xib = eb - mu
    Dbb = D0 - Ds * eb / (4.0 * t)
    Eb = np.maximum(np.hypot(xib, Dbb), 1e-12)

    # ---- (1) scattering: b lower, a upper with E_a = E_b + omega ---------------------------------------------
    Ea = Eb + omega
    fa, fb = fermi_qp(Ea, T), fermi_qp(Eb, T)
    S_sc = 0.0
    internals = {}
    ell_a = np.sqrt(Ea ** 2 - P["Emin"] ** 2)                       # > 0 because Ea >= Emin + omega
    jac = Ea / (P["s"] * ell_a)
    for sgn in (+1, -1):
        ea = P["estar"] + sgn * ell_a / P["s"]
        xia, Dba = ea - mu, D0 - Ds * ea / (4.0 * t)
        internals[sgn] = dict(ea=ea, xia=xia, Dba=Dba)
        C = 0.5 * (1.0 + xib * xia / (Eb * Ea) - eta * Dbb * Dba / (Eb * Ea))
        S_sc += np.sum(wb * dos.N(ea) * jac * C * (fb - fa))

    # ---- (2) pair creation: b is the lower member of the pair (E_b < omega/2), E_a = omega - E_b ----------------
    S_pair = 0.0
    sel = Eb < 0.5 * omega
    if np.any(sel):
        Ebs, xibs, Dbbs, wbs = Eb[sel], xib[sel], Dbb[sel], wb[sel]
        Eas = omega - Ebs
        ell = np.sqrt(np.maximum(Eas ** 2 - P["Emin"] ** 2, 1e-300))
        jac = Eas / (P["s"] * ell)
        occ = 1.0 - fermi_qp(Eas, T) - fermi_qp(Ebs, T)
        for sgn in (+1, -1):
            ea = P["estar"] + sgn * ell / P["s"]
            xia, Dba = ea - mu, D0 - Ds * ea / (4.0 * t)
            Cp = 0.5 * (1.0 - xibs * xia / (Ebs * Eas) + eta * Dbbs * Dba / (Ebs * Eas))
            S_pair += np.sum(wbs * dos.N(ea) * jac * Cp * occ)

    # ---- normal state ------------------------------------------------------------------------------------------
    S_N = np.sum(wb * dos.N(eb + omega) * (fermi_normal(eb, T, muN) - fermi_normal(eb + omega, T, muN)))
    out = dict(S_sc=S_sc, S_pair=S_pair, S_N=S_N, ratio=(S_sc + S_pair) / S_N if S_N > 0 else np.nan, Emin=P["Emin"])
    if diag:
        out["internals"] = dict(eb=eb, Dbb=Dbb, Eb=Eb, Ea=Ea, ell_a=ell_a, branches=internals, P=P)
    return out


def absorption_E(dos, T, omega, D0, Ds, mu, muN, eta=+1, t=1.0, n=40001):
    """The SAME integrals with the QUASIPARTICLE ENERGY E of the lower state b as the integration variable (explicit limits):

        S_sc   = Int_{Emin}^{Emax}  dE  sum_{sb} Nq_sb(E)  sum_{sa} Nq_sa(E + omega)  C   [f(E) - f(E + omega)]
        S_pair = Int_{Emin}^{omega/2} dE sum_{sb} Nq_sb(E)  sum_{sa} Nq_sa(omega - E)  Cp  [1 - f(E) - f(omega - E)]

        Nq_s(E) = N(eps_s(E)) * E / ( sqrt(1+b^2) sqrt(E^2 - Emin^2) ),   eps_s(E) = eps* + s sqrt(E^2 - Emin^2)/sqrt(1+b^2),  s = +-1
        N(eps) = 0 for |eps| > W (band test)  =>  the upper limit is  Emax = max(E(-W), E(+W)), not infinity.

    The inverse-square-root edge of Nq_sb at Emin is removed by E = Emin + x^2 (dE = 2x dx, ell = x sqrt(E + Emin)).
    The state a is not integrated: the delta function fixes E_a, and its two roots eps_pm(E_a) carry the Jacobian inside Nq_sa(E_a)."""
    P = qp_constants(D0, Ds, mu, t)
    s, Em = P["s"], P["Emin"]
    Emax = max(qp_energy(-dos.W, D0, Ds, mu, t), qp_energy(dos.W, D0, Ds, mu, t))

    def Nq(E, sgn):                                   # branch density of states Nq_s(E) and band energy eps_s(E)   (E > Emin)
        ell = np.sqrt(E ** 2 - Em ** 2)
        eps = P["estar"] + sgn * ell / s
        return dos.N(eps) * E / (s * ell), eps

    def Nq_dE(E, sgn):                                # Nq_s(E) * dE/dx   with dE/dx = 2x: regular at x = 0
        eps = P["estar"] + sgn * np.sqrt(np.maximum(E ** 2 - Em ** 2, 0.0)) / s
        return dos.N(eps) * E * 2.0 / (s * np.sqrt(E + Em)), eps

    def trapz_E(Ehi, f):                              # Int_{Emin}^{Ehi} dE ...  via x
        X = np.sqrt(max(Ehi - Em, 0.0))
        x = np.linspace(0.0, X, n)
        return np.trapezoid(f(Em + x ** 2), x) if X > 0 else 0.0

    def scat(E):                                      # E = E_b (lower), E_a = E + omega
        Ea, tot = E + omega, 0.0
        for sb in (+1, -1):
            wb, eb = Nq_dE(E, sb)
            xib, Dbb = eb - mu, D0 - Ds * eb / (4.0 * t)
            for sa in (+1, -1):
                na, ea = Nq(Ea, sa)
                xia, Dba = ea - mu, D0 - Ds * ea / (4.0 * t)
                C = 0.5 * (1.0 + xib * xia / (E * Ea) - eta * Dbb * Dba / (E * Ea))
                tot = tot + wb * na * C
        return tot * (fermi_qp(E, T) - fermi_qp(Ea, T))

    def pair(E):                                      # E = E_b (lower, < omega/2), E_a = omega - E
        Ea, tot = omega - E, 0.0
        for sb in (+1, -1):
            wb, eb = Nq_dE(E, sb)
            xib, Dbb = eb - mu, D0 - Ds * eb / (4.0 * t)
            for sa in (+1, -1):
                na, ea = Nq(Ea, sa)
                xia, Dba = ea - mu, D0 - Ds * ea / (4.0 * t)
                Cp = 0.5 * (1.0 - xib * xia / (E * Ea) + eta * Dbb * Dba / (E * Ea))
                tot = tot + wb * na * Cp
        return tot * (1.0 - fermi_qp(E, T) - fermi_qp(Ea, T))

    S_sc = trapz_E(Emax, scat)
    S_pair = trapz_E(0.5 * omega, pair) if 0.5 * omega > Em else 0.0
    S_N = np.sum(dos.w_i * dos.N(dos.eps_i + omega) * (fermi_normal(dos.eps_i, T, muN) - fermi_normal(dos.eps_i + omega, T, muN)))
    return dict(S_sc=S_sc, S_pair=S_pair, S_N=S_N, ratio=(S_sc + S_pair) / S_N if S_N > 0 else np.nan, Emin=Em, Emax=Emax)


# ----------------------------------------------------------------------------------------------------------------
# temperature dependence of the self-consistent state (Delta0(T), Delta_s(T), mu(T)) -- uses solve_gap.Model
# ----------------------------------------------------------------------------------------------------------------
def sc_state_vs_T(model, fracs, Tlow=0.01, verbose=False):
    """Walk up from T = Tlow to Tc along the self-consistent branch.  Returns Tc and arrays (T, D0, Ds, mu, muN).
    fracs: increasing T/Tc values in (0, 1)."""
    out = model.solve_low_T(Tlow)
    if out is None:
        return None
    Tc, x, ok, res = out
    xs = [x]
    Ts = [Tlow]
    cur = x.copy()
    for f in fracs:
        T = f * Tc
        if T <= Tlow:
            continue
        xn, ok, res = model.solve_at(T, cur)
        # accept on the residual, not on the solver's success flag (it reports 'no progress' when the guess is already exact)
        if res > 1e-8 or np.hypot(xn[0], xn[1]) < 1e-6:
            if verbose:
                print(f"  gap solver stopped at T/Tc={f:.3f} (ok={ok}, res={res:.1e})")
            break
        cur = xn
        xs.append(xn)
        Ts.append(T)
    xs = np.array(xs)
    Ts = np.array(Ts)
    muN = np.array([model.mu_normal(T) for T in Ts])
    return Tc, Ts, xs[:, 0], xs[:, 1], xs[:, 2], muN
