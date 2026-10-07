"""
Delta(E) for the gap function of Hutchinson, Hirsch & Marsiglio, arXiv:1801.08988 (PRB 97, 184513)

The paper finds (Eq. 37)            Delta(eps) = D0 - Ds*eps/(4t)
with eps = eps_ks the Rashba band energy, and the Bogoliubov energy (Eq. 26)

                                    E = sqrt( (eps - mu)^2 + Delta(eps)^2 ).

Eliminating eps between the two gives TWO MATHEMATICAL BRANCHES of Delta(E):

    b      = Ds/(4t)
    D_F    = D0 - b*mu                     (= Delta at eps = mu, the "Fermi-level gap")
    E_min  = |D_F| / sqrt(1+b^2)           (paper Eq. 40)
    eps*   = (mu + b*D0)/(1+b^2)           (paper Eq. 39, where E = E_min)
    D*     = D_F/(1+b^2)                   (= Delta(eps*))

    eps_pm(E)   = eps* +- sqrt(E^2 - E_min^2)/sqrt(1+b^2)
    Delta_pm(E) = D*   -+ b/sqrt(1+b^2) * sqrt(E^2 - E_min^2)         E >= E_min

("+" branch: eps > eps*,  "-" branch: eps < eps*.)

BAND TEST (Step 6).  A branch describes real (k, s) states only if its band energy lies inside the Rashba band

    -W <= eps_pm(E) <= +W ,      W = 4t sqrt(1 + VSO^2/(2 t^2)) .

A branch that fails the test is discarded (Delta_pm(E) does not exist there).  In terms of E:
    '+' branch real  for  E(max(eps*,-W)) <= E <= E(+W)      (empty if eps* > +W)
    '-' branch real  for  E(min(eps*,+W)) <= E <= E(-W)      (empty if eps* < -W)
with E(eps) = sqrt((eps-mu)^2 + (D0 - b*eps)^2); for |eps*| <= W the lower ends are E_min.

Run this file to (i) solve the paper's gap equations for the parameters of its Fig. 8,
(ii) test the formula against a brute-force evaluation over the whole Brillouin zone,
(iii) test the band rule and the energy windows (also on random parameter sets, including
eps* outside the band), (iv) check the Jacobian / tunnelling-weight formulas and (v) make the figures.
"""
import numpy as np


# --------------------------------------------------------------------------------------
#  band and closed-form result
# --------------------------------------------------------------------------------------
def band_halfwidth(VSO, t=1.0):
    """Half-width W of the Rashba band, eps_ks in [-W, +W]  (W = 4t sqrt(1 + VSO^2/2t^2))."""
    return 4.0 * t * np.sqrt(1.0 + VSO ** 2 / (2.0 * t ** 2))


def E_of_eps(eps, D0, Ds, mu, t=1.0):
    """Quasiparticle energy of a state with band energy eps (paper Eqs. 26 + 37)."""
    return np.sqrt((eps - mu) ** 2 + (D0 - Ds * eps / (4.0 * t)) ** 2)


def delta_of_E(E, D0, Ds, mu, t=1.0, VSO=None):
    """Both mathematical branches of Delta(E) (NaN for E < E_min).

    If VSO is given the band test is applied as well and the dict contains
    ok_plus / ok_minus (True where the root lies inside the band) and
    Delta_plus_phys / Delta_minus_phys / eps_plus_phys / eps_minus_phys, which are NaN
    wherever the branch has to be discarded.
    """
    E = np.asarray(E, dtype=float)
    b = Ds / (4.0 * t)
    DF = D0 - b * mu
    s = np.sqrt(1.0 + b * b)
    Emin = abs(DF) / s
    eps_star = (mu + b * D0) / (1.0 + b * b)
    D_star = DF / (1.0 + b * b)

    with np.errstate(divide="ignore", invalid="ignore"):
        root = np.sqrt(np.where(E >= Emin, E ** 2 - Emin ** 2, np.nan))   # sqrt(E^2 - Emin^2)
        jac = E / (s * root)                    # |d eps / dE| = E / sqrt((1+b^2)E^2 - D_F^2)
    out = dict(
        Emin=Emin, eps_star=eps_star, Delta_star=D_star, Delta_F=DF, b=b,
        eps_plus=eps_star + root / s,           # branch eps > eps*
        eps_minus=eps_star - root / s,          # branch eps < eps*
        Delta_plus=D_star - (b / s) * root,
        Delta_minus=D_star + (b / s) * root,
        jac=jac,
    )
    if VSO is not None:
        W = band_halfwidth(VSO, t)
        with np.errstate(invalid="ignore"):
            ok_p = np.abs(out["eps_plus"]) <= W          # NaN -> False
            ok_m = np.abs(out["eps_minus"]) <= W
        out.update(W=W, ok_plus=ok_p, ok_minus=ok_m)
        for br, ok in (("plus", ok_p), ("minus", ok_m)):
            out[f"Delta_{br}_phys"] = np.where(ok, out[f"Delta_{br}"], np.nan)
            out[f"eps_{br}_phys"] = np.where(ok, out[f"eps_{br}"], np.nan)
    return out


def branch_windows(D0, Ds, mu, VSO, t=1.0):
    """Energy windows in which each branch passes the band test (None = branch never real)."""
    W = band_halfwidth(VSO, t)
    b = Ds / (4.0 * t)
    es = (mu + b * D0) / (1.0 + b * b)
    Emin = abs(D0 - b * mu) / np.sqrt(1.0 + b * b)
    Eof = lambda e: float(E_of_eps(e, D0, Ds, mu, t))
    return dict(
        W=W, eps_star=es, Emin=Emin, eps_star_in_band=bool(abs(es) <= W),
        E_plusW=Eof(W), E_minusW=Eof(-W),
        win_plus=(Eof(max(es, -W)), Eof(W)) if es <= W else None,
        win_minus=(Eof(min(es, W)), Eof(-W)) if es >= -W else None,
    )


# --------------------------------------------------------------------------------------
#  brute-force test over the full Brillouin zone, starting from the paper's k-space ansatz
# --------------------------------------------------------------------------------------
def brute_force_states(D0, Ds, mu, VSO, t=1.0, Nk=120):
    """E_ks and Delta_bar_ks for every (k, s), built from Eqs. (15), (31), (35), (26)
    -- i.e. WITHOUT using the simplified form (37).  Returns eps, Dbar, E, s (flat arrays)."""
    k = (np.arange(Nk) + 0.5) * 2 * np.pi / Nk - np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    out = []
    for s in (+1, -1):
        R = np.sqrt(np.sin(kx) ** 2 + np.sin(ky) ** 2)
        eps_k = -2 * t * (np.cos(kx) + np.cos(ky))
        eps_ks = eps_k - 2 * s * VSO * R                                     # Eq. (15)
        s_k = 0.5 * (np.cos(kx) + np.cos(ky))
        D_xy = VSO / (2 * t) * Ds                                            # Eq. (35)
        Dbar = D0 + Ds * s_k + D_xy * s * R                                  # Eq. (31)
        E = np.sqrt((eps_ks - mu) ** 2 + Dbar ** 2)                          # Eq. (26)
        out.append((eps_ks.ravel(), Dbar.ravel(), E.ravel(), np.full(eps_ks.size, s)))
    return [np.concatenate(c) for c in zip(*out)]


def helicity_band_edges(VSO, t=1.0, Nk=1200):
    """Band edges of each helicity band, on a grid that contains the points (0,0) and (pi,pi)."""
    k = np.linspace(-np.pi, np.pi, Nk + 1)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    R = np.sqrt(np.sin(kx) ** 2 + np.sin(ky) ** 2)
    e0 = -2 * t * (np.cos(kx) + np.cos(ky))
    out = {}
    for s_ in (+1, -1):
        e = e0 - 2 * s_ * VSO * R
        out[s_] = (float(e.min()), float(e.max()))
    return out


def stress_test_windows(nsets=300, Nk=300, seed=1, t=1.0):
    """Random (D0, Ds, mu, VSO) -- not necessarily self-consistent: the band rule is pure algebra.
    Checks, against brute force over every (k, s) state,
      (1) no state lies outside the formula windows of its branch  (rule never discards a real state
          and never keeps a root that has no state below its window),
      (2) the upper end of every window (= branch reaches a band edge) is filled up to grid resolution,
      (3) every state sits exactly on its branch:  |eps_ks - eps_pm(E_ks)| ~ 1e-13 and |eps_ks| <= W."""
    rng = np.random.default_rng(seed)
    viol = gap = ident = 0.0
    n_out = n_empty = n_unres = 0
    for _ in range(nsets):
        VSO, mu = rng.uniform(0.05, 1.2), rng.uniform(-6.0, 6.0)
        D0, Ds = rng.uniform(-1.5, 1.5), rng.uniform(-4.0, 4.0)
        bw = branch_windows(D0, Ds, mu, VSO, t)
        eps, Dbar, E, hel = brute_force_states(D0, Ds, mu, VSO, t, Nk)
        W = bw["W"]
        assert eps.max() <= W + 1e-12 and eps.min() >= -W - 1e-12      # band is inside [-W, W]
        plus = eps > bw["eps_star"]
        n_out += not bw["eps_star_in_band"]
        r = delta_of_E(E, D0, Ds, mu, t)
        e_br = np.where(plus, r["eps_plus"], r["eps_minus"])
        ident = max(ident, float(np.nanmax(np.abs(e_br - eps))))
        for sel, win in ((plus, bw["win_plus"]), (~plus, bw["win_minus"])):
            if win is None:
                assert not sel.any()
                n_empty += 1
                continue
            if not sel.any():
                n_unres += 1                                            # branch exists but grid has no state in it
                continue
            lo, hi = E[sel].min(), E[sel].max()
            viol = max(viol, win[0] - lo, hi - win[1], 0.0)
            gap = max(gap, win[1] - hi)
    return dict(nsets=nsets, n_eps_star_outside=int(n_out), n_empty_branches=int(n_empty), n_unresolved=int(n_unres),
                max_violation=viol, max_upper_gap=gap, max_identity_error=ident)


# --------------------------------------------------------------------------------------
#  figures
# --------------------------------------------------------------------------------------
CP, CM, CG, CO = "#d1495b", "#00798c", "#333333", "#e09f3e"   # '+' branch, '-' branch, gap line, E(eps)
DASH = (0, (4, 2))


def make_figures(D0, Ds, mu, VSO, label, outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "mathtext.fontset": "stix",
                         "font.family": "STIXGeneral"})
    W = band_halfwidth(VSO)
    r = delta_of_E(np.array([2.0]), D0, Ds, mu, VSO=VSO)
    b, DF, Emin, es, Ds_ = r["b"], r["Delta_F"], r["Emin"], r["eps_star"], r["Delta_star"]
    eps, Dbar, E, hel = brute_force_states(D0, Ds, mu, VSO)
    bw = branch_windows(D0, Ds, mu, VSO)
    EpW, EmW = bw["E_plusW"], bw["E_minusW"]
    s_ = np.sqrt(1 + b * b)
    Ed = 2.0                                                  # demonstration energy
    rd = delta_of_E(np.array([Ed]), D0, Ds, mu, VSO=VSO)
    ep_d, em_d = float(rd["eps_plus"][0]), float(rd["eps_minus"][0])

    # ---------------- Figure 1 : (a) Delta(eps), E(eps)   (b) geometry ----------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.9))
    a = ax[0]
    xlo, xhi = -5.6, 6.5
    xs = np.linspace(xlo, xhi, 1400)
    Dg = D0 - Ds * xs / 4
    Eg_ = np.sqrt((xs - mu) ** 2 + Dg ** 2)
    inb = np.abs(xs) <= W
    a.axvspan(xlo, -W, color="0.90", zorder=0); a.axvspan(W, xhi, color="0.90", zorder=0)
    a.text(-W - 0.12, 6.2, "no states", ha="right", va="center", fontsize=9, color="0.35")
    a.text(W + 0.12, 6.2, "no states", ha="left", va="center", fontsize=9, color="0.35")
    a.plot(xs, np.where(inb, Dg, np.nan), color=CG, lw=2.0, label=r"$\Delta(\varepsilon)$, paper Eq. 37")
    a.plot(xs, np.where(inb, np.nan, Dg), color=CG, lw=1.5, ls=DASH, alpha=0.75)
    a.plot(xs, np.where(inb, Eg_, np.nan), color=CO, lw=2.3, label=r"$E(\varepsilon)$  (dashed: outside the band)")
    a.plot(xs, np.where(inb, np.nan, Eg_), color=CO, lw=1.6, ls=DASH, alpha=0.8)
    a.axhline(0, color="k", lw=0.5)
    a.axvline(mu, color="gray", ls=":", lw=1.2)
    a.text(mu - 0.1, 5.0, r"$\mu$", color="gray", fontsize=13, ha="right")
    a.axhline(Ed, color="0.45", lw=1, ls=":")
    a.plot([em_d], [Ed], "o", ms=9, color=CM, zorder=6, label=r"$E=2t$: root inside the band, real state")
    a.plot([ep_d], [Ed], "o", ms=9, mfc="white", mec=CP, mew=2.0, zorder=6,
           label=r"$E=2t$: root outside the band, discarded")
    a.plot([es], [Emin], "o", color="k", zorder=5)
    a.annotate(r"minimum of $E(\varepsilon)$ at $\varepsilon^*\neq\mu$:" + f"\n$E_{{\\min}}$ = {Emin:.3f} t",
               (es, Emin), xytext=(-4.7, 3.0), fontsize=10, arrowprops=dict(arrowstyle="->"))
    a.axhline(Emin, color="k", lw=0.6, ls="--", alpha=0.5)
    for xe, tx in ((-W, r"$-W$"), (W, r"$+W$")):
        a.axvline(xe, color="k", lw=0.8, ls="-", alpha=0.6)
        a.text(xe, -1.75, tx, ha="center", va="bottom", fontsize=10)
    a.set_xlabel(r"$\varepsilon\ /\ t$"); a.set_ylabel("energy / t")
    a.set_title(r"(a)  $E(\varepsilon)$ is two-to-one, but only roots inside the band are real")
    a.legend(fontsize=8.4, loc="upper right", framealpha=0.95); a.grid(alpha=0.25)
    a.set_xlim(xlo, xhi); a.set_ylim(-1.9, 10.2)

    a = ax[1]
    xi_all = np.linspace(-7, 7, 200)
    DFline = DF - b * xi_all
    inb2 = np.abs(xi_all + mu) <= W
    a.plot(xi_all[inb2], DFline[inb2], color=CG, lw=2.2, label=r"gap line $\Delta=\Delta_F-b\,\xi$ (states exist)")
    a.plot(xi_all[~inb2], DFline[~inb2], color=CG, lw=1.4, ls=":", label="same line outside the band")
    th = np.linspace(0, 2 * np.pi, 400)
    Es = [1.1, 2.0, 4.0]
    a.plot(Emin * np.cos(th), Emin * np.sin(th), color="k", lw=1.8, label=r"circle $E=E_{\min}$ (tangent to line)")
    a.plot([], [], color="#777777", lw=1, ls="--", label=r"circles $E=1.1,\,2,\,4\ t$")
    for Ec in Es:
        a.plot(Ec * np.cos(th), Ec * np.sin(th), color="#777777", lw=1, ls="--")
        ang = np.deg2rad(125)
        a.text(Ec * np.cos(ang), Ec * np.sin(ang), rf"{Ec:g}", fontsize=9, color="#444444", ha="center", va="center",
               bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
        rc = delta_of_E(np.array([Ec]), D0, Ds, mu)
        for e_, d_, c_ in [(rc["eps_plus"][0], rc["Delta_plus"][0], CP), (rc["eps_minus"][0], rc["Delta_minus"][0], CM)]:
            inside = abs(e_) <= W
            a.plot([e_ - mu], [d_], "o", ms=8, color=c_ if inside else "white", mec=c_, mew=1.6, zorder=6)
    a.plot([es - mu], [Ds_], "s", color="k", ms=7, zorder=7)
    a.annotate(r"foot of the perpendicular:" + "\n" + r"$E=E_{\min}$ at $\varepsilon=\varepsilon^*$",
               (es - mu, Ds_), xytext=(-6.5, -4.6), fontsize=9.8, arrowprops=dict(arrowstyle="->"))
    a.plot([0], [0], "k+", ms=11)
    a.plot([], [], "o", color=CP, ms=7, label=r"root with $\varepsilon>\varepsilon^*$  ($+$ branch)")
    a.plot([], [], "o", color=CM, ms=7, label=r"root with $\varepsilon<\varepsilon^*$  ($-$ branch)")
    a.plot([], [], "o", color="white", mec="gray", ms=7, label="open: root outside the band, discarded")
    a.set_aspect("equal"); a.set_xlim(-6.8, 6.8); a.set_ylim(-5.6, 7.6)
    a.set_xlabel(r"$\xi=\varepsilon-\mu\ /\ t$"); a.set_ylabel(r"$\Delta\ /\ t$")
    a.set_title(r"(b)  $E=\sqrt{\xi^2+\Delta^2}$ is the distance from the origin")
    a.legend(fontsize=8.2, loc="upper right", ncol=1); a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig1_geometry.png", dpi=170)
    plt.close(fig)

    # ---------------- Figure 2 : the band test  eps_pm(E) vs E ---------------------------------
    Emax_plot = 9.3
    Eg = np.linspace(Emin, Emax_plot, 9000)
    rg = delta_of_E(Eg, D0, Ds, mu, VSO=VSO)
    ylo, yhi = -6.6, 12.8
    order = np.argsort(E)
    sel = np.zeros(eps.size, bool); sel[order[::40]] = True
    fig, a = plt.subplots(figsize=(8.6, 5.4))
    a.axhspan(W, yhi, color="0.90", zorder=0); a.axhspan(ylo, -W, color="0.90", zorder=0)
    a.axhline(W, color="k", lw=0.9, ls=":"); a.axhline(-W, color="k", lw=0.9, ls=":")
    a.text(0.12, W + 0.3, rf"$+W={W:.3f}\,t$", ha="left", fontsize=10)
    a.text(0.12, -W - 0.75, rf"$-W=-{W:.3f}\,t$", ha="left", fontsize=10)
    a.text(0.15, 11.45, r"outside the Rashba band, $|\varepsilon|>W$: no states", fontsize=9.6, color="0.3")
    for key, col, lab in (("plus", CP, r"$\varepsilon_+(E)$ ($+$ branch)"), ("minus", CM, r"$\varepsilon_-(E)$ ($-$ branch)")):
        e_ = rg[f"eps_{key}"]
        a.plot(Eg, rg[f"eps_{key}_phys"], color=col, lw=3.4, label=lab + ": inside the band = real")
        a.plot(Eg, np.where(rg[f"ok_{key}"], np.nan, e_), color=col, lw=1.7, ls=DASH, alpha=0.8)
    a.plot([], [], color="0.4", lw=1.7, ls=DASH, label="dashed: outside the band = discarded")
    a.plot(E[sel], eps[sel], "o", mfc="k", mec="none", ms=2.4, zorder=5, label=r"brute force: every $(\mathbf{k},s)$ state")
    a.plot([Emin], [es], "s", color="k", ms=6.5, zorder=6, label=rf"threshold: $E_{{\min}}={Emin:.3f}\,t$ at $\varepsilon^*={es:.2f}\,t$")
    for Ex, col in ((EpW, CP), (EmW, CM)):
        a.axvline(Ex, color=col, lw=1, ls=":")
    a.plot([EpW], [W], "D", color="white", mec=CP, mew=1.8, ms=7, zorder=7)
    a.plot([EmW], [-W], "D", color="white", mec=CM, mew=1.8, ms=7, zorder=7)
    a.annotate(rf"$\varepsilon_+$ reaches $+W$ at" + "\n" + rf"$E_{{+W}}={EpW:.2f}\,t$", (EpW, W), xytext=(1.9, 8.1),
               fontsize=9.6, arrowprops=dict(arrowstyle="->", color=CP), color=CP)
    a.annotate(rf"$\varepsilon_-$ reaches $-W$ at $E_{{-W}}={EmW:.2f}\,t$", (EmW, -W), xytext=(3.55, -5.05),
               fontsize=9.6, arrowprops=dict(arrowstyle="->", color=CM), color=CM, va="center")
    # worked example E = 2t
    a.plot([Ed], [ep_d], "o", ms=9, mfc="white", mec=CP, mew=2.0, zorder=8)
    a.plot([Ed], [em_d], "o", ms=9, color=CM, zorder=8)
    a.annotate(rf"$E=2t$: $\varepsilon_+={ep_d:.2f}\,t>W$" + "\n" + r"$\Rightarrow$ discarded (outside the band)", (Ed, ep_d), xytext=(2.75, 4.5),
               fontsize=9.6, arrowprops=dict(arrowstyle="->"), va="bottom")
    a.annotate(rf"$E=2t$: $\varepsilon_-={em_d:.2f}\,t$" + "\n" + r"$\Rightarrow$ real state", (Ed, em_d), xytext=(0.25, -1.4),
               fontsize=9.6, arrowprops=dict(arrowstyle="->"), va="bottom")
    # strip: number of real branches
    strip = [(0.0, Emin, "0", "#dddddd"), (Emin, min(EpW, EmW), "2", "#bfe3c0"), (min(EpW, EmW), max(EpW, EmW), "1", "#ffe6a8"),
             (max(EpW, EmW), Emax_plot, "0", "#dddddd")]
    for x0, x1, txt, col in strip:
        a.add_patch(Rectangle((x0, ylo), x1 - x0, 0.95, fc=col, ec="k", lw=0.6, zorder=3))
        a.text(0.5 * (x0 + x1), ylo + 0.47, txt, ha="center", va="center", fontsize=10, zorder=4)
    a.set_xlim(0, Emax_plot); a.set_ylim(ylo, yhi)
    a.set_xlabel(r"quasiparticle energy  $E\ /\ t$"); a.set_ylabel(r"band energy of the root,  $\varepsilon_\pm(E)\ /\ t$")
    a.grid(alpha=0.25)
    a.set_title(label, fontsize=10)
    from matplotlib.patches import Patch
    hs, ls = a.get_legend_handles_labels()
    hs.append(Patch(fc="#ffe6a8", ec="k", lw=0.6)); ls.append("bottom strip: number of real branches (0 | 2 | 1 | 0)")
    a.legend(hs, ls, fontsize=8.6, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig2_band_test.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figure 3 : Delta(E) : formula vs brute force ---------------------------
    fig, a = plt.subplots(figsize=(8.6, 5.9))
    Eg = np.linspace(Emin, Emax_plot, 9000)
    rg = delta_of_E(Eg, D0, Ds, mu, VSO=VSO)
    asym = Ds_ + (b / s_) * Eg
    a.plot(Eg, rg["Delta_plus_phys"], color=CP, lw=7, solid_capstyle="butt", alpha=0.85,
           label=rf"$\Delta_+(E)$ for $E\leq E_{{+W}}={EpW:.2f}\,t$ (branch $\varepsilon>\varepsilon^*$, real)")
    a.plot(Eg, rg["Delta_minus_phys"], color=CM, lw=7, solid_capstyle="butt", alpha=0.85,
           label=rf"$\Delta_-(E)$ for $E\leq E_{{-W}}={EmW:.2f}\,t$ (branch $\varepsilon<\varepsilon^*$, real)")
    a.plot(Eg, np.where(rg["ok_plus"], np.nan, rg["Delta_plus"]), color=CP, lw=1.9, ls=DASH, alpha=0.85)
    a.plot(Eg, np.where(rg["ok_minus"], np.nan, rg["Delta_minus"]), color=CM, lw=1.9, ls=DASH, alpha=0.85)
    a.plot([], [], color="0.35", lw=1.9, ls=DASH, label=r"same formulas where $|\varepsilon_\pm(E)|>W$: discarded, no states")
    a.plot(E[sel], Dbar[sel], "o", mfc="k", mec="none", ms=2.6, zorder=5,
           label=r"brute force: every $(\mathbf{k},s)$ state, built from Eqs. (15),(31),(35),(26)")
    a.plot(Eg, asym, color="gray", lw=1, ls="--", zorder=1)
    a.text(5.2, 0.12, r"asymptote $\Delta^*+\dfrac{b}{\sqrt{1+b^2}}\,E$", color="gray", fontsize=10, rotation=12, rotation_mode="anchor")
    a.axvline(Emin, color="k", ls=":", lw=1)
    a.text(Emin + 0.07, 1.78, rf"$E_{{\min}}={Emin:.3f}\,t$", fontsize=10.5)
    a.axhline(0, color="k", lw=0.5)
    a.plot([Emin], [Ds_], "s", color="k", ms=6, zorder=6,
           label=rf"threshold $E_{{\min}}$: both branches meet at $\Delta^*=\Delta(\varepsilon^*)={Ds_:.3f}\,t$")
    a.plot([EpW], [D0 - Ds * W / 4], "D", color="white", mec=CP, mew=1.8, ms=7, zorder=7)
    a.plot([EmW], [D0 + Ds * W / 4], "D", color="white", mec=CM, mew=1.8, ms=7, zorder=7,
           label=r"end of a real branch: its root reaches a band edge $\varepsilon=\pm W$")
    eps0 = 4 * D0 / Ds
    E0 = abs(eps0 - mu)
    a.plot([E0], [0], "o", color="k", ms=5, zorder=6)
    a.annotate(rf"gap changes sign at $\varepsilon_0=4t\,\Delta^0/\Delta^s={eps0:.2f}\,t$" + "\n" + rf"$\Rightarrow$ $\Delta_-(E)=0$ at $E_0=|\varepsilon_0-\mu|={E0:.2f}\,t$",
               (E0, 0), xytext=(3.55, -0.62), fontsize=9.6, arrowprops=dict(arrowstyle="->"))
    a.annotate(rf"$\Delta_+(E)$ continued beyond $E_{{+W}}$:" + "\n" + r"$\varepsilon_+>W$, no $(\mathbf{k},s)$ state exists", (6.3, float(rg["Delta_plus"][np.searchsorted(Eg, 6.3)])),
               xytext=(2.6, -2.75), fontsize=9.6, arrowprops=dict(arrowstyle="->", color=CP), color=CP)
    a.set_xlabel(r"quasiparticle energy  $E\ /\ t$"); a.set_ylabel(r"$\Delta(E)\ /\ t$")
    a.set_xlim(0.0, Emax_plot); a.set_ylim(-3.95, 2.1)
    a.grid(alpha=0.25)
    a.set_title(label, fontsize=10)
    a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.145), ncol=1, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig3_delta_of_E.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return dict(E_plusW=EpW, E_minusW=EmW, W=W, band_lo=eps.min(), band_hi=eps.max(), eps0=eps0, E0=E0)


if __name__ == "__main__":
    from solve_gap import Model

    # ---- (i) parameters of the paper's Fig. 8 ----------------------------------------
    U, dt, n, VSO, kT = 90.0, 4.5, 1.875, 0.5, 0.01
    m = Model(U=U, dt=dt, n=n, VSO=VSO, Nk=1600)
    Tc, x, ok, res = m.solve_low_T(kT)
    D0, Ds, mu = x
    b = Ds / 4
    print(f"U={U}t  dt={dt}t  n={n}  VSO={VSO}t  kT={kT}t   (converged={ok}, residual={res:.1e})")
    print(f"  kB Tc = {Tc:.4f} t")
    print(f"  Delta0 = {D0:.5f} t   Delta_s = {Ds:.5f} t   mu = {mu:.5f} t")
    r = delta_of_E(np.array([2.0]), D0, Ds, mu, VSO=VSO)
    print(f"  b = Ds/4t = {b:.5f},  Delta_F = {r['Delta_F']:.5f} t,  eps* = {r['eps_star']:.5f} t,"
          f"  Delta* = {r['Delta_star']:.5f} t")
    print(f"  E_min = {r['Emin']:.5f} t  (paper Eq. 40) ;  2 E_min / kB Tc = {2*r['Emin']/Tc:.3f}")
    print(f"  zero of the gap function: eps_0 = 4t D0/Ds = {4*D0/Ds:.4f} t")

    # ---- (ii) brute-force check -----------------------------------------------------------
    eps, Dbar, E, hel = brute_force_states(D0, Ds, mu, VSO)
    W = band_halfwidth(VSO)
    print(f"\nbrute force over {eps.size} (k,s) states:  min E over BZ = {E.min():.5f} t "
          f"(formula: {r['Emin']:.5f} t)")
    r_all = delta_of_E(E, D0, Ds, mu, VSO=VSO)
    upper = eps > r_all["eps_star"]
    Dformula = np.where(upper, r_all["Delta_plus"], r_all["Delta_minus"])
    epsformula = np.where(upper, r_all["eps_plus"], r_all["eps_minus"])
    ok_branch = np.where(upper, r_all["ok_plus"], r_all["ok_minus"])
    ok_pts = np.isfinite(Dformula)
    print(f"  max |Delta_bar(k,s) - Delta_pm(E_ks)| = {np.max(np.abs(Dbar[ok_pts]-Dformula[ok_pts])):.2e} t"
          f"   (finite points: {ok_pts.sum()} / {ok_pts.size})")
    print(f"  max |eps_ks      - eps_pm(E_ks)|   = {np.max(np.abs(eps[ok_pts]-epsformula[ok_pts])):.2e} t")
    print(f"  every state sits on a branch that passes the band test: {bool(ok_branch.all())}")

    # ---- (ii-b) band test and energy windows ---------------------------------------------
    print("\nband test (Step 6):")
    hb = helicity_band_edges(VSO)
    for s_ in (+1, -1):
        print(f"  helicity s={s_:+d}: eps_ks in [{hb[s_][0]:.4f}, {hb[s_][1]:.4f}] t   (analytic: s=+1: [-W, 4t], s=-1: [-4t, W])")
    print(f"  both helicities: [{eps.min():.4f}, {eps.max():.4f}] t ;  W = 4t sqrt(1+VSO^2/2t^2) = {W:.4f} t")
    bw = branch_windows(D0, Ds, mu, VSO)
    print(f"  eps* = {bw['eps_star']:.4f} t inside the band: {bw['eps_star_in_band']}")
    print(f"  E(+W) = {bw['E_plusW']:.4f} t   E(-W) = {bw['E_minusW']:.4f} t")
    for name, sel_, win in (("'+' (eps>eps*)", upper, bw["win_plus"]), ("'-' (eps<eps*)", ~upper, bw["win_minus"])):
        print(f"  branch {name}: formula window E in [{win[0]:.4f}, {win[1]:.4f}] t ;"
              f" brute force: [{E[sel_].min():.4f}, {E[sel_].max():.4f}] t  ({sel_.sum()} states)")
    print("  worked examples (formula, band test applied):")
    for Ex in (0.95, 1.2, 2.0, 5.0, 8.0, 8.5):
        rr = delta_of_E(np.array([Ex]), D0, Ds, mu, VSO=VSO)
        parts = []
        for br in ("plus", "minus"):
            e_, d_, ok_ = float(rr[f"eps_{br}"][0]), float(rr[f"Delta_{br}"][0]), bool(rr[f"ok_{br}"][0])
            parts.append(f"{'+' if br=='plus' else '-'}: eps={e_:7.3f} -> {'KEEP  Delta=%7.4f' % d_ if ok_ else 'discard'}")
        print(f"    E={Ex:5.2f}t   " + "   ".join(parts))
    st = stress_test_windows()
    print(f"\nrandom-parameter stress test of the band rule ({st['nsets']} sets of (D0,Ds,mu,VSO), brute force on a 300x300x2 grid):")
    print(f"  sets with eps* outside the band: {st['n_eps_star_outside']};  branches that never pass the test: {st['n_empty_branches']};"
          f"  unresolved by the grid: {st['n_unresolved']}")
    print(f"  max amount by which a state lies outside its window = {st['max_violation']:.1e} t   (expect 0)")
    print(f"  max unfilled part at the upper window end            = {st['max_upper_gap']:.1e} t   (grid resolution)")
    print(f"  max |eps_ks - eps_pm(E_ks)|                          = {st['max_identity_error']:.1e} t")

    # ---- (iii) Jacobian and tunnelling weights ---------------------------------------------
    # <F(E_ks)> and <u^2 F>, <v^2 F> computed (a) directly from the states, (b) from the formulas
    # N_s(E)=sum_{admissible} N(eps_s)|d eps_s/dE| ,  N_t(+-E)=sum_{admissible} N(eps_s)(E +- xi_s)/(2 sqrt(D)).
    eps_h, w_h = m.eps, m.w                                   # fine DOS histogram of the normal state
    dens = lambda e: np.interp(e, eps_h, w_h / np.gradient(eps_h), left=0.0, right=0.0)   # N(eps), integral = 2
    Ef = np.sqrt((eps_h - mu) ** 2 + (D0 - Ds * eps_h / 4) ** 2)
    xi_h = eps_h - mu
    F = lambda e: np.exp(-e / 2.0) * (1 + 0.3 * np.cos(e))
    direct = [np.dot(w_h, F(Ef)), np.dot(w_h, 0.5 * (1 + xi_h / Ef) * F(Ef)), np.dot(w_h, 0.5 * (1 - xi_h / Ef) * F(Ef))]
    # (the formulas contain integrable inverse-square-root edges -> substitute E = sqrt(Emin^2+q^2))
    q = np.linspace(0, np.sqrt(9.0 ** 2 - r["Emin"] ** 2), 400001)
    Eq = np.sqrt(r["Emin"] ** 2 + q ** 2)
    rq = delta_of_E(Eq, D0, Ds, mu, VSO=VSO)
    with np.errstate(divide="ignore", invalid="ignore"):
        sDq = np.sqrt((1 + b * b) * Eq ** 2 - rq["Delta_F"] ** 2)
        res_, res_nobt = [], []
        for fac in ("s", "t+", "t-"):
            tot = tot_nobt = 0.0
            for br in ("plus", "minus"):
                e_ = rq[f"eps_{br}"]; xi_ = e_ - mu
                keep = rq[f"ok_{br}"]                                   # <-- the band test
                Nbr = dens(e_) * rq["jac"] if fac == "s" else dens(e_) * (Eq + (1 if fac == "t+" else -1) * xi_) / (2 * sDq)
                integrand = np.nan_to_num(Nbr * (q / Eq) * F(Eq), nan=0.0, posinf=0.0)
                tot += np.trapezoid(np.where(keep, integrand, 0.0), q)
                tot_nobt += np.trapezoid(integrand, q)
            res_.append(tot); res_nobt.append(tot_nobt)
    print("\nJacobian / tunnelling-weight check  (F(E) = exp(-E/2)(1+0.3 cos E)):")
    for name, d_, f_ in zip(("N_s   (all)", "N_t(+E) (u^2)", "N_t(-E) (v^2)"), direct, res_):
        print(f"  {name:14s} direct sum over states = {d_:.5f}   via formula = {f_:.5f}   rel.diff = {abs(d_-f_)/abs(d_):.1e}")
    beta = (r["eps_star"] - mu) / r["Emin"]
    print(f"  xi*/E_min = sgn(DF) b/sqrt(1+b^2) = {beta:.5f}; edge weights N_t(+E)/N_t(-E) -> (1+beta)/(1-beta) = {(1+beta)/(1-beta):.4f}")

    # ---- (v) figures --------------------------------------------------------------------
    label = (rf"$U={U:.0f}t,\ \Delta t={dt}t,\ n={n},\ V_{{SO}}={VSO}t,\ k_BT={kT}t$   $\Rightarrow$   "
             rf"$\Delta^0={D0:.3f}t,\ \Delta^s={Ds:.3f}t,\ \mu={mu:.3f}t$")
    info = make_figures(D0, Ds, mu, VSO, label)
    print("\nfigures written; extra numbers:", {k: round(float(v), 4) for k, v in info.items()})
