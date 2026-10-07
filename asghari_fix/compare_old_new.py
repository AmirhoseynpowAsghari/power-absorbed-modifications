"""Old formula (project/ultrasonic/absorption.py, results read from the repo's JSON files) against the energy-space formula.

   python3 compare_old_new.py  [path to project/ultrasonic/results]            (about 1 min)

Parameters of the user's stage-1 run on branch master (project/sc_params/results/tc_vs_vso.json -> meta): U = 95, dt = 4.5, n = 1.885, omega = 0.7, t = 1.
The self-consistent Delta0(T), Delta_s(T), mu(T) of the SAME gap equations are obtained with solve_gap.Model (checked against stage 1: Tc agrees to 0.4-3%, Delta0 and Delta_s to 2%).
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qp_absorption import RashbaDOS, absorption, sc_state_vs_T
from solve_gap import Model

REPO = Path("/home/user/asghari_master/project/ultrasonic/results") if len(sys.argv) < 2 else Path(sys.argv[1])
U, DT, N_FILL, OMEGA = 95.0, 4.5, 1.885, 0.7
NK = 1200


def new_curve(VSO, fracs):
    m = Model(U=U, dt=DT, n=N_FILL, VSO=VSO, Nk=NK)
    dos = RashbaDOS(VSO, Nk=NK)
    out = sc_state_vs_T(m, fracs)
    Tc, Ts, D0, Ds, mu, muN = out
    res = {"I": [], "II": [], "Emin": []}
    for T, d0, ds, mu_, mun in zip(Ts, D0, Ds, mu, muN):
        for eta, key in ((+1, "I"), (-1, "II")):
            r = absorption(dos, T, OMEGA, d0, ds, mu_, mun, eta=eta)
            res[key].append(r["ratio"])
        res["Emin"].append(r["Emin"])
    return dict(Tc=Tc, T=Ts, frac=Ts / Tc, D0=D0, Ds=Ds, mu=mu, muN=muN, **{k: np.array(v) for k, v in res.items()})


def main():
    fracs = np.concatenate([np.linspace(0.04, 0.9, 44), np.linspace(0.91, 0.995, 18)])
    out = {}
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    for j, v in enumerate((0.0, 0.4082, 0.8)):
        old = json.load(open(REPO / f"temp_sweep_VSO_{v:.4f}.json"))
        To, ro, Tco = np.array(old["T"]), np.array(old["ratio"], float), float(np.ravel(old["Tc"])[0])
        new = new_curve(v, fracs)
        out[v] = new
        # plot the old curve without the artificial point at T = 0 (set to 0 by the T < 1e-4 shortcut)
        m = To > 1e-3
        for r_, a in enumerate(ax[:, j]):
            a.plot(To[m] / Tco, ro[m], "-", color="0.55", lw=2, label="old formula (repo)")
            a.plot(np.r_[new["frac"], 1.0, 1.1], np.r_[new["I"], 1.0, 1.0], "-", color="#00798c", lw=2, label=r"energy space, $\eta=+1$ (ultrasound)")
            a.plot(np.r_[new["frac"], 1.0, 1.1], np.r_[new["II"], 1.0, 1.0], "--", color="#d1495b", lw=1.6, label=r"energy space, $\eta=-1$ (EM)")
            a.grid(alpha=0.25)
        ax[0, j].set_title(rf"$V_{{SO}}={v}$,   $T_c={new['Tc']:.3f}$,   $E_{{min}}(0)={new['Emin'][0]:.2f}$", fontsize=10)
        ax[0, j].set_ylim(-0.03, 1.08)
        ax[1, j].set_yscale("log")
        ax[1, j].set_ylim(1e-9, 3)
        ax[1, j].set_xlabel(r"$T/T_c$")
        ax[0, j].axhline(1, color="k", lw=0.6, ls=":")
    ax[0, 0].set_ylabel(r"$W/W_N$"); ax[1, 0].set_ylabel(r"$W/W_N$  (log scale)")
    ax[0, 0].legend(fontsize=8, loc="center left")
    fig.suptitle(r"Ultrasonic absorption, $\hbar\omega=0.7t$, $U=95$, $\Delta_t=4.5$, $n=1.885$: repo formula (master) vs. quasiparticle energy-space formula", fontsize=11)
    fig.tight_layout()
    fig.savefig("fig_old_vs_new_T.png", dpi=170, bbox_inches="tight")
    np.savez("new_T_curves.npz", **{f"V{v}_{k}": a for v, d in out.items() for k, a in d.items()})

    # ---- V_SO scan at fixed T/Tc ------------------------------------------------------------------------------
    vs = np.round(np.arange(0.0, 0.8001, 0.05), 4)
    tf = (0.3, 0.5, 0.7, 0.9)
    R = {f: [] for f in tf}
    EM = {f: [] for f in tf}
    for v in vs:
        m = Model(U=U, dt=DT, n=N_FILL, VSO=v, Nk=NK)
        dos = RashbaDOS(v, Nk=NK)
        Tc, Ts, D0, Ds, mu, muN = sc_state_vs_T(m, list(tf))
        for f in tf:
            i = int(np.argmin(np.abs(Ts / Tc - f)))
            r = absorption(dos, Ts[i], OMEGA, D0[i], Ds[i], mu[i], muN[i], eta=+1)
            R[f].append(r["ratio"]); EM[f].append(r["Emin"])
    old = json.load(open(REPO / "vso_scan.json"))
    vo = np.array(old["VSO_arr"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    cols = ["#7f1d1d", "#dc2626", "#f97316", "#d4a900"]
    for c, f in zip(cols, tf):
        ro = np.array(old["results"][str(f)]["ratio"], float)
        ax[0].plot(vo, ro, "-", color=c, alpha=0.6, lw=1.4, label=f"T/Tc={f}")
        ax[1].plot(vs, R[f], "o-", color=c, ms=3.5, lw=1.4, label=f"T/Tc={f}")
    ax[0].set_title("repo formula (as in the paper's Fig.)", fontsize=10)
    ax[1].set_title(r"energy-space formula, $\eta=+1$ (U = 95)", fontsize=10)
    ax[1].set_yscale("log")
    for a in ax:
        a.set_xlabel(r"$V_{SO}\ [t]$"); a.set_ylabel(r"$W/W_N$"); a.grid(alpha=0.25); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_old_vs_new_VSO.png", dpi=170, bbox_inches="tight")
    json.dump(dict(VSO=vs.tolist(), ratio={str(f): [float(x) for x in R[f]] for f in tf}, Emin={str(f): [float(x) for x in EM[f]] for f in tf}),
              open("new_vso_scan.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
