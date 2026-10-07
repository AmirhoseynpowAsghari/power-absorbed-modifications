"""Energy-space W/W_N for another U (no old curves):   python3 run_new_only.py 95"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qp_absorption import RashbaDOS, absorption, sc_state_vs_T
from solve_gap import Model

U = float(sys.argv[1]) if len(sys.argv) > 1 else 95.0
DT, N_FILL, OMEGA, NK = 4.5, 1.885, 0.7, 1200
fracs = np.concatenate([np.linspace(0.04, 0.9, 44), np.linspace(0.91, 0.995, 18)])
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
for v, c in ((0.0, "#1f77b4"), (0.4082, "#d62728"), (0.8, "#2ca02c")):
    m = Model(U=U, dt=DT, n=N_FILL, VSO=v, Nk=NK)
    dos = RashbaDOS(v, Nk=NK)
    out = sc_state_vs_T(m, fracs)
    if out is None:
        print("no SC solution for V_SO =", v); continue
    Tc, Ts, D0, Ds, mu, muN = out
    for eta, ls in ((+1, "-"), (-1, "--")):
        R = [absorption(dos, T, OMEGA, d0, ds, mu_, mn, eta=eta) for T, d0, ds, mu_, mn in zip(Ts, D0, Ds, mu, muN)]
        r = np.array([x["ratio"] for x in R])
        for a in ax:
            a.plot(np.r_[Ts / Tc, 1.0, 1.1], np.r_[r, 1.0, 1.0], ls, color=c, lw=1.8, label=(f"V_SO={v}, Tc={Tc:.3f}, Emin(0)={R[0]['Emin']:.2f}" if eta == 1 else None))
    print(f"U={U}, V_SO={v}: Tc={Tc:.4f}, Emin(T->0)={R[0]['Emin']:.3f}, mu(T->0)={mu[0]:.3f}, W={dos.W:.3f}")
ax[1].set_yscale("log"); ax[1].set_ylim(1e-9, 3)
for a in ax:
    a.set_xlabel(r"$T/T_c$"); a.set_ylabel(r"$W/W_N$"); a.grid(alpha=0.25)
ax[0].legend(fontsize=8)
ax[0].set_title(rf"energy-space formula, $U={U:.0f}$, $\Delta_t=4.5$, $n=1.885$, $\hbar\omega=0.7t$  (solid $\eta=+1$, dashed $\eta=-1$)", fontsize=9)
fig.tight_layout()
fig.savefig(f"fig_new_T_U{U:.0f}.png", dpi=170, bbox_inches="tight")
