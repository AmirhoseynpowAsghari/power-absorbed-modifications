"""W/W_N for eta = +1 from scratch: quasiparticle SCATTERING only, energy conservation on E' - E = hbar*omega,
signed coherence factor, branch-resolved energy space.  Inputs taken from the repository (branch master).

  C_a = 1/2 [1 + ( xi xi'      - Dbar Dbar') / (E E')]     (a) correct, signed xi
  C_b = 1/2 [1 + (|xi||xi'|    - Dbar Dbar') / (E E')]     (b) |xi||xi'|
  C_c = 1/2 [1 + (|xi||xi'|    + Dbar Dbar') / (E E')]     (c) the paper's final expression

with, for the lower state (E, branch sigma) and the upper state (E' = E + hbar*omega, branch sigma'):
  xi_s(E) = xi* + s l/kappa,  Dbar_s(E) = Delta* - s b l/kappa,  l = sqrt(E^2 - Emin^2),  kappa = sqrt(1 + b^2),  weight  nu_s(E) = N(eps_s(E)) E / (kappa l)

  python3 recalc_eta_plus_scattering_only.py
"""
import os
import sys
import json
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/user/asghari_master/project/ultrasonic"
cwd0 = os.getcwd()
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import parameters_ua as PUA
from kgrid_ua import build_dispersions_ua
from dos_ua import compute_dos
from physics_ua import Delta_T, find_mu, GAP_FIT_P, GAP_FIT_Q
from absorption import compute_W
os.chdir(cwd0)
from coherence_variants_energy_space import S_parts
from qp_absorption import qp_constants

OMEGA, KB, NTARGET, B = PUA.omega, PUA.kB, PUA.n_target, PUA.B
E_grid = np.linspace(PUA.E_grid_min, PUA.E_grid_max, PUA.NE)
stage1 = json.load(open(ROOT + "/../sc_params/results/tc_vs_vso.json"))["results"]
V1 = np.array(stage1["V_SO"])


class RepoDOS:
    """The repository's density-of-states table g(E) (Gaussian sigma = 0.05, 200 x 200 k-grid, 500-point grid on [-6, 6]) used as N(eps)."""

    def __init__(self, E_grid, gE):
        self.E_grid, self.gE = E_grid, gE
        w = gE * (E_grid[1] - E_grid[0])
        w[0] *= 0.5
        w[-1] *= 0.5
        self.eps_i, self.w_i = E_grid, w

    def N(self, eps):
        return np.interp(np.asarray(eps, float), self.E_grid, self.gE, left=0.0, right=0.0)


def mu_energy_gap(T, D0, Ds, VSO):
    """mu(T) from the number equation n = 1 - <(xi/E) tanh(E/2T)>  (both helicity bands, the repository's 200 x 200 grid) with the
    ENERGY-DEPENDENT gap Dbar(eps) = D0 - Ds eps/4t  (the repository's find_mu uses one constant gap)."""
    ep, em = build_dispersions_ua(VSO)

    def f(mu):
        tot = 0.0
        for eps in (ep, em):
            xi = eps - mu
            E = np.sqrt(xi ** 2 + (D0 - Ds * eps / (4.0 * PUA.t)) ** 2)
            tot += np.mean(xi / np.maximum(E, 1e-300) * (np.tanh(E / (2.0 * KB * T)) if T > 1e-10 else 1.0))
        return 1.0 - 0.5 * tot - NTARGET
    return brentq(f, -25.0, 25.0, xtol=1e-12)


VARIANTS = (("(a) signed xi xi'", "signed", -1), ("(b) |xi||xi'|", "abs", -1), ("(c) paper's final", "abs", +1))


def run(VSO, fracs, mu_mode="consistent"):
    i = int(np.argmin(np.abs(V1 - VSO)))
    Tc, D00, Ds0, Deff0 = stage1["Tc"][i], stage1["D0_T0"][i], stage1["DS_T0"][i], stage1["Deff_T0"][i]
    ep, em = build_dispersions_ua(VSO)
    gE = compute_dos(E_grid, ep, em)
    dos = RepoDOS(E_grid, gE)
    rows = []
    for f in fracs:
        T = f * Tc
        phi = Delta_T(T, Tc, 1.0) if T < Tc else 0.0                      # the repository's fitted shape (1 - (T/Tc)^p)^q
        D0T, DsT = phi * D00, phi * Ds0                                   # rigid scaling of BOTH amplitudes: <|Dbar|> = Delta_eff(T) of the repository
        if mu_mode == "consistent":
            mu = mu_energy_gap(T, D0T, DsT, VSO)
        else:                                                             # the repository's own mu(T) (constant gap Delta_eff(T))
            mu = find_mu(T, Delta_T(T, Tc, Deff0), VSO)
        WN = compute_W(T, mu, 0.0, E_grid, gE, include_gap=False)         # the repository's W_N: bracket 2 for every band pair, same mu, normal-state f
        S_N = abs(WN) / (2.0 * np.pi * OMEGA * B ** 2 * 2.0)
        muN = mu_energy_gap(T, 0.0, 0.0, VSO)
        res = {"f": f, "T": T, "mu": mu, "muN": muN, "Emin": qp_constants(D0T, DsT, mu)["Emin"], "S_N": S_N, "Deff": phi * Deff0}
        for name, xm, sg in VARIANTS:
            S, _ = S_parts(dos, T, OMEGA, D0T, DsT, mu, +1, xm, sg, pair=False)          # scattering only
            S0, _ = S_parts(dos, T, OMEGA, 0.0, 0.0, muN, +1, xm, sg, pair=False)        # the same formula at Delta -> 0 (mu_N)
            S_Nn = abs(compute_W(T, muN, 0.0, E_grid, gE, include_gap=False)) / (2.0 * np.pi * OMEGA * B ** 2 * 2.0)
            res[name] = S / S_N                                                          # W / W_N  (the paper's W_N)
            res[name + " own"] = S / S0 if S0 > 0 else np.nan                            # W / W(Delta = 0) of the same formula
            res[name + " lim"] = S0 / S_Nn                                               # Delta -> 0 limit of W/W_N
        rows.append(res)
    return dict(VSO=VSO, Tc=Tc, D00=D00, Ds0=Ds0, Deff0=Deff0, rows=rows, dos=dos, gE=gE)


if __name__ == "__main__":
    fracs = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    out = {v: run(v, fracs) for v in (0.0, 0.4082, 0.8)}
    print(f"hbar*omega = {OMEGA}, n = {NTARGET}, sigma_DOS = {PUA.sigma}, grid {PUA.NE} points on [{PUA.E_grid_min}, {PUA.E_grid_max}], gap shape (1-(T/Tc)^p)^q with p = {GAP_FIT_P}, q = {GAP_FIT_Q}")
    for v, r in out.items():
        print(f"\n===== V_SO = {v}:  stage-1 Tc = {r['Tc']:.4f}, Delta0(0) = {r['D00']:.4f}, Delta_s(0) = {r['Ds0']:.4f}, Delta_eff(0) = {r['Deff0']:.4f} =====")
        print(f"{'T/Tc':>6} {'Deff(T)':>8} {'mu':>7} {'2Emin':>6} | {'(a) signed':>10} {'(b) |xi||xi|':>13} {'(c) paper':>10} | {'(a)/own':>8} {'(b)/own':>8} {'(c)/own':>8}")
        for x in r["rows"]:
            print(f"{x['f']:6.3f} {x['Deff']:8.4f} {x['mu']:7.4f} {2 * x['Emin']:6.3f} | {x[VARIANTS[0][0]]:10.5f} {x[VARIANTS[1][0]]:13.5f} {x[VARIANTS[2][0]]:10.5f} | "
                  f"{x[VARIANTS[0][0] + ' own']:8.4f} {x[VARIANTS[1][0] + ' own']:8.4f} {x[VARIANTS[2][0] + ' own']:8.4f}")
        last = r["rows"][-1]
        print(f"   Delta -> 0 limit of W/W_N (T = {last['f']} Tc):  (a) {last[VARIANTS[0][0] + ' lim']:.4f}   (b),(c) {last[VARIANTS[1][0] + ' lim']:.4f}   [and for the paper's own band-energy formula: 1]")
        for name, _, _ in VARIANTS:
            vals = np.array([x[name] for x in r["rows"]])
            print(f"   {name:20s}: max over 0.02 <= T/Tc <= 0.999 = {vals.max():.4f} (at T/Tc = {r['rows'][int(vals.argmax())]['f']}),  last value (T/Tc = 0.999) = {vals[-1]:.4f}")
        on = [x["f"] for x in r["rows"] if 2 * x["Emin"] < OMEGA]
        print(f"   2 Emin(T) < hbar*omega (pair creation energetically allowed, NOT included here) for T/Tc >= {on[0] if on else 'none'}")
        # T >= Tc: Delta = 0
        tail = []
        for fct in (1.0, 1.2, 1.4):
            T = fct * r["Tc"]
            muN = mu_energy_gap(T, 0.0, 0.0, r["VSO"])
            S_Nn = abs(compute_W(T, muN, 0.0, E_grid, r["gE"], include_gap=False)) / (2.0 * np.pi * OMEGA * B ** 2 * 2.0)
            vals = [S_parts(r["dos"], T, OMEGA, 0.0, 0.0, muN, +1, xm, sg, pair=False)[0] / S_Nn for _, xm, sg in VARIANTS]
            tail.append((fct, vals))
        print("   T >= Tc (Delta = 0), scattering-only formula / W_N:  " + ";  ".join(f"T/Tc = {f}: (a) {v[0]:.3f}, (b),(c) {v[1]:.3f}" for f, v in tail) + "   [the paper sets W/W_N = 1 there]")

    # sensitivity to the choice of mu(T) (repository's constant-gap mu versus the number equation for the same energy-dependent gap)
    print("\n--- sensitivity to mu(T): V_SO = 0.4082, variant (a) signed, W/W_N ---")
    alt = run(0.4082, [0.3, 0.5, 0.7, 0.9], mu_mode="repo")
    for x, y in zip(out[0.4082]["rows"], [None]):
        pass
    base = {x["f"]: x for x in out[0.4082]["rows"]}
    for x in alt["rows"]:
        b0 = base.get(x["f"])
        if b0 is None:
            b0 = run(0.4082, [x["f"]])["rows"][0]
        print(f"   T/Tc = {x['f']}: mu consistent = {b0['mu']:.4f} -> {b0[VARIANTS[0][0]]:.5f};   repository's mu = {x['mu']:.4f} -> {x[VARIANTS[0][0]]:.5f}")

    # ---- figure -------------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.2), sharex=True)
    colors = {"(a) signed xi xi'": "#00798c", "(b) |xi||xi'|": "#e08e0b", "(c) paper's final": "#d1495b"}
    styles = {"(a) signed xi xi'": "-", "(b) |xi||xi'|": "--", "(c) paper's final": ":"}
    for j, (v, r) in enumerate(out.items()):
        f = np.array([x["f"] for x in r["rows"]])
        old = json.load(open(ROOT + f"/results/temp_sweep_VSO_{v:.4f}.json"))
        To, ro, Tco = np.array(old["T"]), np.array(old["ratio"], float), float(np.ravel(old["Tc"])[0])
        for row in (0, 1):
            a = ax[row, j]
            m = To > 1e-3
            a.plot(To[m] / Tco, ro[m], "-", color="0.65", lw=2, label="repository (old formula)" if row == 0 and j == 0 else None)
            for name, _, _ in VARIANTS:
                y = np.array([x[name] for x in r["rows"]])
                a.plot(f, y, styles[name], color=colors[name], lw=2, label=name if row == 0 and j == 0 else None)
            last = r["rows"][-1]
            a.plot([1.0], [last[VARIANTS[0][0] + " lim"]], "o", color=colors[VARIANTS[0][0]], ms=6)
            a.plot([1.0], [last[VARIANTS[1][0] + " lim"]], "s", color=colors[VARIANTS[1][0]], ms=6)
            on = [x["f"] for x in r["rows"] if 2 * x["Emin"] < OMEGA]
            if on:
                a.axvline(on[0], color="k", lw=0.8, ls=":")
            a.grid(alpha=0.25)
        ax[0, j].set_title(rf"$V_{{SO}}={v}$, $T_c={r['Tc']:.3f}$", fontsize=10)
        ax[0, j].set_ylim(-0.02, 1.08)
        ax[0, j].axhline(1, color="k", lw=0.6, ls=":")
        ax[1, j].set_yscale("log")
        ax[1, j].set_ylim(1e-8, 2)
        ax[1, j].set_xlabel(r"$T/T_c$")
    ax[0, 0].set_ylabel(r"$W/W_N$  (paper's $W_N$)")
    ax[1, 0].set_ylabel("log scale")
    ax[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(r"$\eta=+1$, quasiparticle scattering only, energy conservation on $E'-E=\hbar\omega$ (dotted vertical line: $2E_{\min}=\hbar\omega$; markers at $T_c$: $\Delta\to0$ limits)", fontsize=10)
    fig.tight_layout()
    fig.savefig("fig_eta_plus_scattering_only.png", dpi=170, bbox_inches="tight")
