"""Matrix element M(ks|k's') of arXiv:2607.02812 in the helicity basis of the Rashba superconductor
(model of arXiv:1801.08988: Rashba SOC + extended-s singlet gap + triplet gap along the Rashba vector).

ANALYTIC RESULT (derived in the note):
    M(ks|k's') = B(ks|k's') [u_ks u_k's' - eta v_ks v_k's'] ,     B(ks|k's') = <ks| O |k's'>
    |M|^2      = |B|^2 (1/2) [1 + xi xi'/(E E') - eta Dbar Dbar'/(E E')]
    pair creation:  N = X (u_ks v_k's' + eta u_k's' v_ks),  X = <ks| O |T(k's')>
    eta = +1 for a time-reversal-even operator (ultrasound, case I), -1 for a T-odd one (current, case II)

CHECK: exact diagonalisation of the 4x4 spin-space BdG Hamiltonian, whose gap matrix is built in the SPIN basis
(singlet + triplet), i.e. without using any helicity-space argument.

    python3 helicity_matrix_element.py        (about 10 s)   -> prints all checks, writes fig5_helicity_matrix_element.png
"""
import numpy as np

# ---------------------------------------------------------------------------------------------
# parameters (energies in units of the hopping t = 1): my solution of the gap equations, V_SO = 0.5 t,
# U = 90 t, dt = 4.5 t, n = 1.875, k_B T = 0.01 t   (solve_gap.py)
# ---------------------------------------------------------------------------------------------
P = dict(VSO=0.5, D0=0.29255, Ds=1.33119, mu=3.72649)
T_HOP = 1.0

s0 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
isy = 1j * sy                                  # i sigma_y = [[0, 1], [-1, 0]]


def neg(k):
    return (-k[0], -k[1])


# ------------------------------------------------------------------ band structure and spinors
def R_of(k):
    return np.hypot(np.sin(k[0]), np.sin(k[1]))


def eps0(k, t=T_HOP):
    return -2 * t * (np.cos(k[0]) + np.cos(k[1]))


def s_k(k):
    return 0.5 * (np.cos(k[0]) + np.cos(k[1]))


def eps_s(k, s, VSO, t=T_HOP):                 # paper 1801.08988 Eq. (15)
    return eps0(k, t) - 2 * s * VSO * R_of(k)


def phase(k):                                  # e^{i theta(k)}, Eq. (17)
    return (np.sin(k[1]) - 1j * np.sin(k[0])) / R_of(k)


def spinor(k, s):                              # helicity eigenvector, Eq. (16): (1, s e^{i theta}) / sqrt 2
    return np.array([1.0, s * phase(k)]) / np.sqrt(2)


def rho(k, s):                                 # T |ks> = rho_ks |-k s>,  T = i sigma_y K   ->  rho = s e^{-i theta}
    return s * np.conj(phase(k))


def H_so(k, VSO):                              # -2 V_SO (sin ky sx - sin kx sy)
    return -2 * VSO * (np.sin(k[1]) * sx - np.sin(k[0]) * sy)


def h_spin(k, p, t=T_HOP):
    return (eps0(k, t) - p["mu"]) * s0 + H_so(k, p["VSO"])


def Dbar(k, s, p, t=T_HOP):                    # trivial-irrep gap, Eqs. (36)-(37): D0 - Ds eps_ks / 4t  (signed, real)
    return p["D0"] - p["Ds"] * eps_s(k, s, p["VSO"], t) / (4 * t)


def uv(k, s, p, t=T_HOP):
    """u, v (real, v carries the sign of Dbar), E, xi, Dbar for the helicity state (k, s)."""
    xi = eps_s(k, s, p["VSO"], t) - p["mu"]
    Db = Dbar(k, s, p, t)
    E = np.hypot(xi, Db)
    return np.sqrt(0.5 * (1 + xi / E)), np.sign(Db) * np.sqrt(0.5 * (1 - xi / E)), E, xi, Db


# ------------------------------------------------------------------ spin-space BdG (independent of helicity arguments)
def D_spin(k, p, t=T_HOP):
    """Gap matrix in SPIN space: [ (D0 + Ds s_k) + Dxy (g_k . sigma) ] i sigma_y,   g_k = (sin ky, -sin kx, 0),  Dxy = V_SO Ds / 2t."""
    Dxy = p["VSO"] * p["Ds"] / (2 * t)
    g_sigma = np.sin(k[1]) * sx - np.sin(k[0]) * sy
    return ((p["D0"] + p["Ds"] * s_k(k)) * s0 + Dxy * g_sigma) @ isy


def bdg(k, p, t=T_HOP):
    """4x4 BdG matrix in the Nambu basis (c_k up, c_k dn, c^+_{-k up}, c^+_{-k dn})."""
    D = D_spin(k, p, t)
    return np.block([[h_spin(k, p, t), D], [D.conj().T, -h_spin(neg(k), p, t).T]])


def positive_vectors(k, p, t=T_HOP):
    """Positive-energy BdG eigenvectors at k, labelled by helicity (matched to the analytic E_ks)."""
    w, V = np.linalg.eigh(bdg(k, p, t))
    out = {}
    for s in (+1, -1):
        E = uv(k, s, p, t)[2]
        j = int(np.argmin(np.abs(w - E)))
        if abs(w[j] - E) > 1e-9:
            raise RuntimeError("BdG eigenvalue does not match E_ks")
        out[s] = V[:, j]
    return out


# ------------------------------------------------------------------ one-body vertices B(k|k') = spin matrix
def vertex_scalar(k, kp, p=None):
    """Ultrasound (deformation potential): spin independent, T-even (eta = +1).  Momentum transfer q = k - k'."""
    return s0


def make_vertex_current(alpha, p, t=T_HOP):
    """Electromagnetic absorption: A . dH/dk at the mid-point (k + k')/2, A = (cos alpha, sin alpha); T-odd (eta = -1)."""
    def vertex(k, kp, _p=None):
        km = (0.5 * (k[0] + kp[0]), 0.5 * (k[1] + kp[1]))
        Vx = 2 * t * np.sin(km[0]) * s0 + 2 * p["VSO"] * np.cos(km[0]) * sy
        Vy = 2 * t * np.sin(km[1]) * s0 - 2 * p["VSO"] * np.cos(km[1]) * sx
        return np.cos(alpha) * Vx + np.sin(alpha) * Vy
    return vertex


# ------------------------------------------------------------------ the matrix element: analytic and numerical
def M_analytic(k, s, kp, sp, p, vertex, eta, t=T_HOP):
    """Returns (M, B_hel) with  M = B_hel (u u' - eta v v')  and B_hel = <ks|O|k's'>."""
    u, v = uv(k, s, p, t)[:2]
    up, vp = uv(kp, sp, p, t)[:2]
    Bhel = np.vdot(spinor(k, s), vertex(k, kp, p) @ spinor(kp, sp))
    return Bhel * (u * up - eta * v * vp), Bhel


def Nambu_B(Bkk, Bmm):
    """diag(B_{k,k'}, -B_{-k',-k}^T): Nambu form of the number-conserving one-body operator."""
    Z = np.zeros((2, 2), dtype=complex)
    return np.block([[Bkk, Z], [Z, -Bmm.T]])


def M_numeric(k, s, kp, sp, p, vertex, t=T_HOP):
    """Coefficient of gamma^+_{ks} gamma_{k's'} in O = sum B c^+ c, from the exact spin-space BdG eigenvectors."""
    w = positive_vectors(k, p, t)[s]
    wp = positive_vectors(kp, p, t)[sp]
    return np.vdot(w, Nambu_B(vertex(k, kp, p), vertex(neg(kp), neg(k), p)) @ wp)


TAUX = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]]).astype(complex)


def N_analytic(k, s, pp, r, p, vertex, eta, t=T_HOP):
    """Pair-creation amplitude for gamma^+_{ks} gamma^+_{pp r}:  N = X (u_a v_b + eta u_b v_a),  |X| = |<ks|O|-pp r>|."""
    ua, va = uv(k, s, p, t)[:2]
    ub, vb = uv(pp, r, p, t)[:2]
    X = np.vdot(spinor(k, s), vertex(k, neg(pp), p) @ spinor(neg(pp), r))      # modulus = |<a|O|T b>|
    return X * (ua * vb + eta * ub * va), X


def N_numeric(k, s, pp, r, p, vertex, t=T_HOP):
    """Coefficient of gamma^+_a gamma^+_b (a < b) in O, from the exact BdG eigenvectors: (Y_ab - Y_ba)/2."""
    wa = positive_vectors(k, p, t)[s]
    wb = positive_vectors(pp, p, t)[r]
    Yab = np.vdot(wa, Nambu_B(vertex(k, neg(pp), p), vertex(pp, neg(k), p)) @ (TAUX @ np.conj(wb)))
    Yba = np.vdot(wb, Nambu_B(vertex(pp, neg(k), p), vertex(k, neg(pp), p)) @ (TAUX @ np.conj(wa)))
    return 0.5 * (Yab - Yba)


def coherence(xi, Db, E, xip, Dbp, Ep, eta):
    """(u u' - eta v v')^2 = (1/2)[1 + xi xi'/EE' - eta Dbar Dbar'/EE']  (scattering)."""
    return 0.5 * (1 + xi * xip / (E * Ep) - eta * Db * Dbp / (E * Ep))


def coherence_pair(xi, Db, E, xip, Dbp, Ep, eta):
    """(u v' + eta u' v)^2 = (1/2)[1 - xi xi'/EE' + eta Dbar Dbar'/EE']  (pair creation)."""
    return 0.5 * (1 - xi * xip / (E * Ep) + eta * Db * Dbp / (E * Ep))


# ------------------------------------------------------------------ the checks
def random_k(rng, tries=1000):
    while True:
        k = tuple(rng.uniform(-np.pi, np.pi, 2))
        if R_of(k) > 0.08 and abs(uv(k, +1, P)[2] - uv(k, -1, P)[2]) > 2e-3:
            return k


def run_checks(nsamp=1500, seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    res = {}

    # (a) time-reversal phase and spin-space <-> helicity gap
    dT, dG, dE = 0.0, 0.0, 0.0
    for _ in range(200):
        k = random_k(rng)
        for s in (+1, -1):
            dT = max(dT, np.abs(isy @ np.conj(spinor(k, s)) - rho(k, s) * spinor(neg(k), s)).max())
        D = D_spin(k, P)
        G = np.array([[np.vdot(spinor(k, s), D @ np.conj(spinor(neg(k), sp))) for sp in (+1, -1)] for s in (+1, -1)])
        want = np.diag([rho(neg(k), s) * Dbar(k, s, P) for s in (+1, -1)])
        dG = max(dG, np.abs(G - want).max())
        ev = np.sort(np.linalg.eigvalsh(bdg(k, P)))
        Es = sorted([uv(k, +1, P)[2], uv(k, -1, P)[2]])
        dE = max(dE, np.abs(ev - np.array([-Es[1], -Es[0], Es[0], Es[1]])).max())
    res.update(dT=dT, dG=dG, dE=dE)

    # (b) the q = 0 helicity vertices of the current: closed forms against spinor sandwiches
    dV = 0.0
    for _ in range(200):
        k = random_k(rng)
        R = R_of(k)
        for s in (+1, -1):
            for sp in (+1, -1):
                for comp in ("x", "y"):
                    vert = make_vertex_current(0.0 if comp == "x" else np.pi / 2, P)
                    num = np.vdot(spinor(k, s), vert(k, k, P) @ spinor(k, sp))
                    if comp == "x":
                        cf = (2 * T_HOP * np.sin(k[0]) - 2 * s * P["VSO"] * np.sin(k[0]) * np.cos(k[0]) / R) if s == sp \
                            else 2j * P["VSO"] * s * np.cos(k[0]) * np.sin(k[1]) / R
                    else:
                        cf = (2 * T_HOP * np.sin(k[1]) - 2 * s * P["VSO"] * np.sin(k[1]) * np.cos(k[1]) / R) if s == sp \
                            else -2j * P["VSO"] * s * np.cos(k[1]) * np.sin(k[0]) / R
                    dV = max(dV, abs(num - cf))
    res["dV"] = dV

    # (c) overlap form factor <ks|k's'> = (1/2)(1 + s s' e^{i(theta' - theta)})
    dF = 0.0
    for _ in range(200):
        k, kp = random_k(rng), random_k(rng)
        for s in (+1, -1):
            for sp in (+1, -1):
                ov = np.vdot(spinor(k, s), spinor(kp, sp))
                cf = 0.5 * (1 + s * sp * phase(kp) / phase(k))
                dF = max(dF, abs(ov - cf))
                dF = max(dF, abs(abs(ov) ** 2 - 0.5 * (1 + s * sp * np.cos(np.angle(phase(k)) - np.angle(phase(kp))))))
    res["dF"] = dF

    # (d) matrix elements against exact BdG: scattering M and pair creation N, both eta = +1 and eta = -1
    cases = [("scalar (ultrasound, eta=+1)", vertex_scalar, +1)]
    for alpha in (0.0, 0.9):
        cases.append((f"current A at {alpha:.1f} rad (EM, eta=-1)", make_vertex_current(alpha, P), -1))
    pts = {"M": [], "N": []}
    worst = {}
    for name, vert, eta in cases:
        wM = wN = 0.0
        for _ in range(nsamp // len(cases)):
            k, kp = random_k(rng), random_k(rng)
            for s in (+1, -1):
                for sp in (+1, -1):
                    Ma, Bh = M_analytic(k, s, kp, sp, P, vert, eta)
                    Mn = M_numeric(k, s, kp, sp, P, vert)
                    Na, X = N_analytic(k, s, kp, sp, P, vert, eta)
                    Nn = N_numeric(k, s, kp, sp, P, vert)
                    pts["M"].append((abs(Mn) ** 2, abs(Ma) ** 2, eta, s * sp))
                    pts["N"].append((abs(Nn) ** 2, abs(Na) ** 2, eta, s * sp))
                    wM = max(wM, abs(abs(Mn) - abs(Ma)))
                    wN = max(wN, abs(abs(Nn) - abs(Na)))
                    # |M|^2 = |B|^2 * coherence  (the closed form of the paper)
                    u, v, E, xi, Db = uv(k, s, P); up, vp, Ep, xip, Dbp = uv(kp, sp, P)
                    wM = max(wM, abs(abs(Ma) ** 2 - abs(Bh) ** 2 * coherence(xi, Db, E, xip, Dbp, Ep, eta)))
                    _, Xp = N_analytic(k, s, kp, sp, P, vert, eta)
                    wN = max(wN, abs(abs(Na) ** 2 - abs(Xp) ** 2 * coherence_pair(xi, Db, E, xip, Dbp, Ep, eta)))
        worst[name] = (wM, wN)
    # (e) electromagnetic consequences at q = 0 (current vertex): intraband M = A.v_ks, intraband pair creation = 0, interband != 0
    alpha = 0.9
    vert = make_vertex_current(alpha, P)
    dM_intra, nN_intra, N_inter = 0.0, 0.0, []
    for _ in range(200):
        k = random_k(rng)
        R = R_of(k)
        for s in (+1, -1):
            vgrp = np.array([2 * T_HOP * np.sin(k[0]) - 2 * s * P["VSO"] * np.sin(k[0]) * np.cos(k[0]) / R,
                             2 * T_HOP * np.sin(k[1]) - 2 * s * P["VSO"] * np.sin(k[1]) * np.cos(k[1]) / R])
            dM_intra = max(dM_intra, abs(M_numeric(k, s, k, s, P, vert) - (np.cos(alpha) * vgrp[0] + np.sin(alpha) * vgrp[1])))
            nN_intra = max(nN_intra, abs(N_numeric(k, s, neg(k), s, P, vert)), abs(N_analytic(k, s, neg(k), s, P, vert, -1)[0]))
            N_inter.append(abs(N_numeric(k, s, neg(k), -s, P, vert)))
    res.update(dM_intra=dM_intra, nN_intra=nN_intra, N_inter_med=float(np.median(N_inter)), N_inter_min=float(np.min(N_inter)))
    res["worst"] = worst
    res["pts"] = {key: np.array(val) for key, val in pts.items()}
    res["nM"] = len(pts["M"])
    res["scale"] = float(np.max(res["pts"]["M"][:, 1]))
    if verbose:
        print("---- spin-space construction ---------------------------------------------------------------")
        print(f"T phase:  max|i sy conj(phi_s(k)) - rho_ks phi_s(-k)|              = {dT:.1e}")
        print(f"gap:      max|<ks|D_spin|conj(-k s')> - delta_ss' rho_(-k)s Dbar_ks|  = {dG:.1e}   (no interband gap)")
        print(f"spectrum: max|eig(BdG) - (+-E_k+, +-E_k-)|                          = {dE:.1e}")
        print("---- helicity-basis vertices -----------------------------------------------------------------")
        print(f"current vertex at q=0, closed forms vs spinor sandwiches            = {dV:.1e}")
        print(f"overlap <ks|k's'> = (1 + s s' e^(i(th'-th)))/2                       = {dF:.1e}")
        print("---- M and N from exact BdG vs analytic helicity formula ---------------------------------------")
        for name, (a, b) in worst.items():
            print(f"{name:38s} max| |M|num-|M|ana | = {a:.1e}    max| |N|num-|N|ana | = {b:.1e}")
        print(f"{res['nM']} (k,k',s,s') combinations per quantity;  largest |M|^2 = {res['scale']:.3f}")
        print("---- electromagnetic vertex at q = 0 ----------------------------------------------------------")
        print(f"intraband M(ks|ks) - A.grad(eps_ks)                = {dM_intra:.1e}")
        print(f"intraband pair creation |N(ks,-k s)|  (max)        = {nN_intra:.1e}")
        print(f"interband pair creation |N(ks,-k -s)| (median, min) = {res['N_inter_med']:.3f}, {res['N_inter_min']:.3f}")
    return res


# ------------------------------------------------------------------ E-space coherence factors, helicity-resolved band test
def brute_grid(p, Nk=300, t=T_HOP):
    k = (np.arange(Nk) + 0.5) * 2 * np.pi / Nk - np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    R = np.hypot(np.sin(kx), np.sin(ky))
    e0 = -2 * t * (np.cos(kx) + np.cos(ky))
    cols = {key: [] for key in ("s", "eps", "xi", "Db", "E")}
    for s in (+1, -1):
        eps = (e0 - 2 * s * p["VSO"] * R).ravel()
        xi = eps - p["mu"]
        Db = p["D0"] - p["Ds"] * eps / (4 * t)
        cols["s"].append(np.full(eps.size, s)); cols["eps"].append(eps); cols["xi"].append(xi)
        cols["Db"].append(Db); cols["E"].append(np.hypot(xi, Db))
    return {key: np.concatenate(v) for key, v in cols.items()}


def E_space_check(E1=1.10, E2=2.00, delta=0.004, verbose=True):
    """Coherence factors for pairs of states with E ~ E1 and E' ~ E2: observed on the k-grid vs the branch formula."""
    from delta_of_E import delta_of_E
    g = brute_grid(P)
    r1 = delta_of_E(E1, P["D0"], P["Ds"], P["mu"], t=T_HOP, VSO=P["VSO"])
    r2 = delta_of_E(E2, P["D0"], P["Ds"], P["mu"], t=T_HOP, VSO=P["VSO"])
    estar = r1["eps_star"]
    m1 = np.abs(g["E"] - E1) < delta
    m2 = np.abs(g["E"] - E2) < delta
    rows = []
    for sig, nm in ((+1, "+"), (-1, "-")):
        for sigp, nmp in ((+1, "+"), (-1, "-")):
            ok1 = bool(r1["ok_plus"] if sig > 0 else r1["ok_minus"])
            ok2 = bool(r2["ok_plus"] if sigp > 0 else r2["ok_minus"])
            sel1 = m1 & ((g["eps"] > estar) if sig > 0 else (g["eps"] <= estar))
            sel2 = m2 & ((g["eps"] > estar) if sigp > 0 else (g["eps"] <= estar))
            row = dict(sig=nm, sigp=nmp, ok1=ok1, ok2=ok2, n1=int(sel1.sum()), n2=int(sel2.sum()))
            if ok1 and ok2:
                xi1 = (r1["eps_plus"] if sig > 0 else r1["eps_minus"]) - P["mu"]
                D1 = r1["Delta_plus"] if sig > 0 else r1["Delta_minus"]
                xi2 = (r2["eps_plus"] if sigp > 0 else r2["eps_minus"]) - P["mu"]
                D2 = r2["Delta_plus"] if sigp > 0 else r2["Delta_minus"]
                for eta, key in ((+1, "C_I"), (-1, "C_II")):
                    row[key] = float(coherence(xi1, D1, E1, xi2, D2, E2, eta))
                    row["ob_" + key] = None
                row["sgn"] = float(np.sign(xi1 * xi2))
                # the paper's prescription xi = +sqrt(E^2 - Delta^2) for both states
                x1p, x2p = np.sqrt(max(E1**2 - D1**2, 0)), np.sqrt(max(E2**2 - D2**2, 0))
                row["paper_C_I"] = float(coherence(x1p, D1, E1, x2p, D2, E2, +1))
                row["paper_C_II"] = float(coherence(x1p, D1, E1, x2p, D2, E2, -1))
                # observed values over all pairs of grid states in the windows
                a, b = np.where(sel1)[0], np.where(sel2)[0]
                if a.size and b.size:
                    for eta, key in ((+1, "C_I"), (-1, "C_II")):
                        C = coherence(g["xi"][a][:, None], g["Db"][a][:, None], g["E"][a][:, None],
                                      g["xi"][b][None, :], g["Db"][b][None, :], g["E"][b][None, :], eta)
                        row["ob_" + key] = (float(C.min()), float(C.max()))
                row["hel1"] = {int(h): int((g["s"][sel1] == h).sum()) for h in (+1, -1)}
                row["hel2"] = {int(h): int((g["s"][sel2] == h).sum()) for h in (+1, -1)}
            rows.append(row)
    if verbose:
        print(f"---- E-space: pairs with E ~ {E1} and E' ~ {E2} (windows +-{delta}), eps* = {estar:.4f}")
        for rw in rows:
            print(rw)
    return rows, r1, r2


def helicity_band_threshold(eps_edge=4.0):
    """Quasiparticle energy of a state at the top of the s=+1 band (eps = 4t): above it the '+' branch exists only for s=-1."""
    from delta_of_E import E_of_eps
    return float(E_of_eps(eps_edge, P["D0"], P["Ds"], P["mu"], T_HOP))


# ------------------------------------------------------------------ average overlap weights vs momentum transfer
def overlap_weights(qs, Nk=500, direction=(1.0, 0.0)):
    """<|<ks|k+q,s'>|^2> over the Brillouin zone: intra-helicity (s'=s) and inter-helicity (s'=-s)."""
    k = (np.arange(Nk) + 0.5) * 2 * np.pi / Nk - np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    R = np.hypot(np.sin(kx), np.sin(ky))
    th = np.angle((np.sin(ky) - 1j * np.sin(kx)) / R)
    out = []
    for q in qs:
        kx2, ky2 = kx + q * direction[0], ky + q * direction[1]
        R2 = np.hypot(np.sin(kx2), np.sin(ky2))
        th2 = np.angle((np.sin(ky2) - 1j * np.sin(kx2)) / R2)
        c = np.cos(th - th2)
        out.append((q, float(np.mean(0.5 * (1 + c))), float(np.mean(0.5 * (1 - c)))))
    return out


# ------------------------------------------------------------------ figure
def make_figure(res, path="fig5_helicity_matrix_element.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7.4})
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.5))
    # (a) form factors
    a = ax[0]
    d = np.linspace(-np.pi, np.pi, 400)
    a.plot(d, 0.5 * (1 + np.cos(d)), color="#00798c", lw=1.8, label=r"$s'=s$:  $\frac{1}{2}(1+\cos\Delta\theta)$")
    a.plot(d, 0.5 * (1 - np.cos(d)), color="#d1495b", lw=1.8, label=r"$s'=-s$:  $\frac{1}{2}(1-\cos\Delta\theta)$")
    a.axhline(0.5, color="k", lw=0.8, ls=":")
    a.text(0, 0.535, "constant-$B$ value $\\frac{1}{2}$", fontsize=7.4, ha="center", va="bottom")
    a.set_xlim(-np.pi, np.pi); a.set_ylim(-0.02, 1.38)
    a.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]); a.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    a.set_xlabel(r"$\Delta\theta=\theta_{\mathbf{k}}-\theta_{\mathbf{k}'}$")
    a.set_ylabel(r"$|\langle \mathbf{k}s|\mathbf{k}'s'\rangle|^2$")
    a.set_title("(a) helicity overlap factor", pad=5)
    a.legend(loc="upper center"); a.grid(alpha=0.25)
    # (b) weights vs |q|
    b = ax[1]
    qs = np.concatenate([np.linspace(0.02, 1.0, 25), np.linspace(1.1, np.pi, 12)])
    W = overlap_weights(qs, Nk=400)
    b.plot([w[0] for w in W], [w[1] for w in W], "-", color="#00798c", lw=1.8, label=r"intra-helicity, $s'=s$")
    b.plot([w[0] for w in W], [w[2] for w in W], "-", color="#d1495b", lw=1.8, label=r"inter-helicity, $s'=-s$")
    b.axhline(0.5, color="k", lw=0.8, ls=":")
    b.set_xlabel(r"momentum transfer $q$ along $k_x$  ($1/a$)"); b.set_ylabel("zone-averaged overlap weight")
    b.set_title("(b) ultrasound: weight vs $q$", pad=5)
    b.set_ylim(-0.02, 1.05); b.legend(loc="upper right"); b.grid(alpha=0.25)
    # (c) verification
    c = ax[2]
    pm, pn = res["pts"]["M"], res["pts"]["N"]
    sc = max(pm[:, 1].max(), pn[:, 1].max()) * 1.05
    c.plot([0, sc], [0, sc], "k-", lw=0.8)
    for pts, mk, lab in ((pm, "o", r"$|M|^2$"), (pn, "s", r"$|N|^2$")):
        for eta, col, nm in ((+1, "#00798c", r"$\eta=+1$"), (-1, "#d1495b", r"$\eta=-1$")):
            sel = pts[:, 2] == eta
            c.plot(pts[sel, 1], pts[sel, 0], mk, ms=2.4, alpha=0.5, color=col, label=f"{lab}, {nm}")
    c.set_xlabel("analytic helicity formula (C), (N)"); c.set_ylabel("exact spin-space BdG")
    c.set_title(f"(c) {res['nM']} random $(\\mathbf{{k}},\\mathbf{{k}}',s,s')$ each", pad=5)
    c.legend(loc="upper left", markerscale=2.6); c.grid(alpha=0.25)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.rcParams.update(plt.rcParamsDefault)
    return path


if __name__ == "__main__":
    res = run_checks()
    rows, r1, r2 = E_space_check()
    print("---- average overlap weights (intra, inter) vs q along kx")
    for q, wi, wo in overlap_weights([0.05, 0.1, 0.3, 1.0, 2.0, np.pi]):
        print(f"q = {q:5.2f}   intra {wi:.4f}   inter {wo:.4f}")
    print("figure:", make_figure(res))
