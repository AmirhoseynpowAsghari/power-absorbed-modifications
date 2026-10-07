"""
Self-consistent mean-field solution of Hutchinson-Hirsch-Marsiglio, arXiv:1801.08988
(Rashba + correlated hopping, square lattice, t = 1).

Because  Delta_bar_ks = D0 - Ds*eps_ks/(4t)  (Eq. 37)  depends on (k,s) only through eps_ks,
the three coupled equations (32),(33),(38) can be written with the single average

        <F> = (1/N) sum_{k,s} F(eps_ks)          (<1> = 2, two helicity bands)

        Delta_bar(eps) = D0 - Ds*eps/(4t)
        E   = sqrt((eps-mu)^2 + Delta_bar^2)
        K   = tanh(E/2T)/(2E)                    [= (1-2f(E))/(2E), the 'g(E)' of Eqs. 32-34]

        D0 = -1/2 [ U <K Dbar> - 2 (dt/t) <eps K Dbar> ]        (Eq. 32:  8 dt s_k + 4 (dt/t) VSO s sqrt(..) = -2 (dt/t) eps_ks)
        Ds = -4 dt <K Dbar>                                      (Eq. 33)
        n  = 1 - 1/2 < (eps-mu)/E * tanh(E/2T) >                 (Eq. 38)

The k-sum is done once and stored as a fine energy histogram (counts and mean energy per bin),
so every later evaluation is a 1-D sum over a few thousand bins.
"""
import numpy as np
from scipy.optimize import brentq, root

t = 1.0


def rashba_dos_histogram(VSO, Nk=1600, nbins=6000):
    """Histogram of eps_ks over both helicity bands.
    Returns (eps_i, w_i) with  sum_i w_i = 2  and  sum_i w_i eps_i = <eps>  exact for each bin."""
    k = (np.arange(Nk) + 0.5) * np.pi / Nk              # midpoint grid in [0, pi] (symmetry of the BZ)
    ky = k[None, :]
    lo, hi = -4.0 * np.sqrt(1 + VSO ** 2 / 2) - 1e-9, 4.0 + 2 * VSO * np.sqrt(2) + 1e-9
    edges = np.linspace(lo, hi, nbins + 1)
    cnt = np.zeros(nbins)
    sm = np.zeros(nbins)
    for i0 in range(0, Nk, 100):                        # chunks of kx to limit memory
        kx = k[i0:i0 + 100, None]
        e0 = -2 * t * (np.cos(kx) + np.cos(ky))
        R = np.sqrt(np.sin(kx) ** 2 + np.sin(ky) ** 2)
        for s in (+1, -1):
            e = (e0 - 2 * s * VSO * R).ravel()
            idx = np.clip(np.searchsorted(edges, e) - 1, 0, nbins - 1)
            cnt += np.bincount(idx, minlength=nbins)
            sm += np.bincount(idx, weights=e, minlength=nbins)
    good = cnt > 0
    eps_i = sm[good] / cnt[good]
    w_i = cnt[good] / Nk ** 2
    return eps_i, w_i


class Model:
    def __init__(self, U, dt, n, VSO, Nk=1600, nbins=6000):
        self.U, self.dt, self.n, self.VSO = U, dt, n, VSO
        self.eps, self.w = rashba_dos_histogram(VSO, Nk, nbins)

    def avg(self, F):
        return np.dot(self.w, F)

    # ---- normal state ---------------------------------------------------------
    def mu_normal(self, T):
        f = lambda mu: 1 - 0.5 * self.avg(np.tanh((self.eps - mu) / (2 * T))) - self.n
        return brentq(f, -8, 8)

    def linear_matrix(self, T):
        """2x2 matrix acting on (D0, c = Ds/4t) of the linearised gap equations."""
        U, dt = self.U, self.dt
        mu = self.mu_normal(T)
        xi = self.eps - mu
        xi = np.where(np.abs(xi) < 1e-12, 1e-12, xi)
        K = np.tanh(xi / (2 * T)) / (2 * xi)
        K0, K1, K2 = self.avg(K), self.avg(self.eps * K), self.avg(self.eps ** 2 * K)
        M = np.array([[1 + 0.5 * U * K0 - dt / t * K1, -0.5 * U * K1 + dt / t * K2],
                      [4 * dt * K0, 4 * t - 4 * dt * K1]])
        return M, mu

    def Tc(self):
        det = lambda T: np.linalg.det(self.linear_matrix(T)[0])
        Ts = np.linspace(0.02, 1.6, 200)
        d = np.array([det(T) for T in Ts])
        idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        if len(idx) == 0:
            return None
        i = idx[-1]
        Tc = brentq(det, Ts[i], Ts[i + 1], xtol=1e-10)
        M, mu = self.linear_matrix(Tc)
        w, v = np.linalg.eig(M)
        vec = v[:, np.argmin(np.abs(w))].real
        return Tc, mu, np.array([vec[0], 4 * t * vec[1]])      # (D0, Ds) direction

    # ---- superconducting state ------------------------------------------------
    def residual(self, x, T):
        D0, Ds, mu = x
        eps = self.eps
        Db = D0 - Ds * eps / (4 * t)
        xi = eps - mu
        E = np.sqrt(xi ** 2 + Db ** 2)
        K = np.tanh(E / (2 * T)) / (2 * E)
        KD = K * Db
        r0 = D0 + 0.5 * (self.U * self.avg(KD) - 2 * self.dt / t * self.avg(eps * KD))
        r1 = Ds + 4 * self.dt * self.avg(KD)
        r2 = self.n - (1 - 0.5 * self.avg(xi / E * np.tanh(E / (2 * T))))
        return np.array([r0, r1, r2])

    def solve_at(self, T, x0):
        sol = root(self.residual, x0, args=(T,), method="hybr", tol=1e-13)
        return sol.x, sol.success, np.max(np.abs(sol.fun))

    def solve_low_T(self, Tlow=0.01):
        out = self.Tc()
        if out is None:
            return None
        Tc, mu_c, vec = out
        vec = vec / np.max(np.abs(vec))
        best = None
        # (a) direct attempt at Tlow from several amplitudes along the linearised null vector
        for amp in (0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0):
            for sgn in (+1, -1):
                x0 = np.array([sgn * amp * vec[0], sgn * amp * vec[1], mu_c])
                x, ok, res = self.solve_at(Tlow, x0)
                if ok and res < 1e-9 and np.hypot(x[0], x[1]) > 1e-3:
                    best = (x, ok, res)
                    break
            if best:
                break
        # (b) otherwise follow the branch down from Tc
        if best is None:
            x = np.array([0.0, 0.0, mu_c])
            Ts = np.linspace(Tc, Tlow, 400)
            amp0 = 0.2
            x = np.array([amp0 * vec[0], amp0 * vec[1], mu_c])
            for T in Ts[2:]:
                x_new, ok, res = self.solve_at(T, x)
                if ok and res < 1e-9 and np.hypot(x_new[0], x_new[1]) > 1e-4:
                    x = x_new
                else:
                    break
            best = (x, ok, res)
        return Tc, best[0], best[1], best[2]


def Emin_of(x):
    D0, Ds, mu = x
    b = Ds / (4 * t)
    return abs(D0 - b * mu) / np.sqrt(1 + b ** 2)


if __name__ == "__main__":
    print("Paper's Fig. 8 (read from the figure): U=90t, dt=4.5t, n=1.875, kBT=0.01t")
    print("   paper:  VSO  0.05  0.1   0.2   0.3   0.4   0.5   0.6")
    print("           kBTc 0.33  0.33  0.35  0.38  0.42  0.47  0.53")
    print("           Emin 0.57  0.58  0.62  0.68  0.75  0.85  1.03")
    print()
    print("   VSO   kBTc   |  D0      Ds      mu     E_min  2Emin/kBTc")
    for VSO in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        m = Model(U=90.0, dt=4.5, n=1.875, VSO=VSO)
        out = m.solve_low_T()
        if out is None:
            print(f"  {VSO:4.2f}   no Tc")
            continue
        Tc, x, ok, res = out
        print(f"  {VSO:4.2f}  {Tc:6.3f}  | {x[0]:7.3f} {x[1]:7.3f} {x[2]:7.3f}  {Emin_of(x):6.3f}  "
              f"{2 * Emin_of(x) / Tc:6.2f}   conv={ok} res={res:.1e}")
