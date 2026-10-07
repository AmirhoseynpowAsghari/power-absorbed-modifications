"""What is E, what are its units, and what 'dimension' does it have in code?

Runs in about a second:   python3 E_dimension_demo.py
Writes fig4_E_on_kgrid.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------
# 1. parameters.  EVERY energy is in units of the hopping t (t = 1), so every energy is a plain float.
#    Delta0, Deltas, mu: converged values of solve_gap.py for U=90t, dt=4.5t, n=1.875, kBT=0.01t, VSO=0.5t
# ------------------------------------------------------------------------------------------------
t, VSO = 1.0, 0.5               # hopping, spin-orbit coupling                   [energy]
D0, Ds, mu = 0.29255, 1.33119, 3.72649   # Delta^0, Delta^s, chemical potential   [energy]
b = Ds / (4 * t)                # dimensionless (energy / energy)

# ------------------------------------------------------------------------------------------------
# 2. the 2-D k-grid.  k is dimensionless here: k*a with lattice constant a = 1, so kx, ky in (-pi, pi)
# ------------------------------------------------------------------------------------------------
Nk = 120
k = (np.arange(Nk) + 0.5) * 2 * np.pi / Nk - np.pi
kx, ky = np.meshgrid(k, k, indexing="ij")                 # two arrays, each of shape (Nk, Nk)
R = np.sqrt(np.sin(kx) ** 2 + np.sin(ky) ** 2)
eps_k = -2 * t * (np.cos(kx) + np.cos(ky))
s_k = 0.5 * (np.cos(kx) + np.cos(ky))

eps = np.empty((2, Nk, Nk)); Dbar = np.empty_like(eps); E = np.empty_like(eps)
for i, s in enumerate((+1, -1)):                          # axis 0 of the stacked arrays = helicity
    eps[i] = eps_k - 2 * s * VSO * R                      # paper Eq. 15              [energy]
    Dbar[i] = D0 + Ds * s_k + (VSO / (2 * t)) * Ds * s * R   # paper Eqs. 31 + 35    [energy]
    E[i] = np.sqrt((eps[i] - mu) ** 2 + Dbar[i] ** 2)     # paper Eq. 26: Bogoliubov quasiparticle energy [energy]

print("---- shapes and units --------------------------------------------------------------------------")
print(f"kx, ky        : shape {kx.shape} each, dimensionless (k*a), range {k.min():.3f} .. {k.max():.3f}")
print(f"E[s,ikx,iky]  : shape {E.shape}, dtype {E.dtype}  -> ONE real number (an energy) per state (s,kx,ky)")
print(f"                {E.size} numbers = 2 helicities x {Nk} x {Nk};   axis 0: s=+1,-1   axis 1: kx   axis 2: ky")
print(f"E range       : {E.min():.4f} .. {E.max():.4f}  (units of t, i.e. E/t);   eps range {eps.min():.4f} .. {eps.max():.4f}")
t_meV = 100.0   # ILLUSTRATION ONLY: if you decide that t = 100 meV
print(f"to get physical units multiply by your t:  E.min = {E.min():.4f} t = {E.min() * t_meV:.1f} meV  (for t = {t_meV:.0f} meV)")

# ------------------------------------------------------------------------------------------------
# 3. E is a scalar FIELD on a 2-D domain: it inherits the lattice symmetry of the (kx, ky) plane
# ------------------------------------------------------------------------------------------------
print("\n---- E over the (kx,ky) plane has the square-lattice symmetry ---------------------------------")
print("E(kx,ky) = E(ky,kx)  :", np.allclose(E[0], E[0].transpose()))
print("E(kx,ky) = E(-kx,ky) :", np.allclose(E[0], E[0][::-1, :]))
print("E(kx,ky) = E(kx,-ky) :", np.allclose(E[0], E[0][:, ::-1]))
n_unique = np.unique(np.round(E.ravel(), 10)).size
print(f"{E.size} states, but only {n_unique} different values of E (states on the same constant-energy contour share E)")

# ------------------------------------------------------------------------------------------------
# 4. k enters ONLY through eps_ks  ->  the 2-D dependence collapses onto one variable
# ------------------------------------------------------------------------------------------------
E_from_eps = np.sqrt((eps - mu) ** 2 + (D0 - b * eps) ** 2)          # uses eps only, no kx, ky, no s
print("\n---- k enters only through eps_ks -----------------------------------------------------------")
print("max |Dbar - (D0 - b*eps)|        :", np.abs(Dbar - (D0 - b * eps)).max())
print("max |E - E(eps)| (eps alone)     :", np.abs(E - E_from_eps).max())


# ------------------------------------------------------------------------------------------------
# 5. Delta(E): ONE variable E (scalar or an array of ANY shape, elementwise); two outputs = two branches
# ------------------------------------------------------------------------------------------------
def delta_of_E(E, D0, Ds, mu, VSO, t=1.0):
    """Physical branches of Delta(E): (Delta_plus, Delta_minus).
    NaN for E < Emin and where the root eps_pm lies outside the band (|eps| > W)."""
    E = np.asarray(E, float)
    b = Ds / (4 * t); DF = D0 - b * mu; s = np.sqrt(1 + b * b)
    W = 4 * t * np.sqrt(1 + VSO ** 2 / (2 * t ** 2))
    Emin = abs(DF) / s; estar = (mu + b * D0) / (1 + b * b); Dstar = DF / (1 + b * b)
    with np.errstate(invalid="ignore"):
        ell = np.sqrt(E ** 2 - Emin ** 2)
        eps_p, eps_m = estar + ell / s, estar - ell / s
        Dp = np.where(np.abs(eps_p) <= W, Dstar - b * ell / s, np.nan)
        Dm = np.where(np.abs(eps_m) <= W, Dstar + b * ell / s, np.nan)
    return Dp, Dm, estar


print("\n---- Delta(E): the input can have any shape, the output has the same shape ----------------------")
for label, Ein in (("scalar", 2.0), ("1-D (5 energies)", np.linspace(1, 5, 5)), ("2-D (the k-grid of s=+1)", E[0])):
    Dp, Dm, estar = delta_of_E(Ein, D0, Ds, mu, VSO, t)
    print(f"E {label:26s} shape {np.shape(Ein)!s:11s} -> Delta_plus, Delta_minus shapes {np.shape(Dp)!s}, {np.shape(Dm)!s}")

Dp, Dm, estar = delta_of_E(E, D0, Ds, mu, VSO, t)                     # whole (2, Nk, Nk) array at once
own_branch = np.where(eps > estar, Dp, Dm)                            # each state sits on the branch eps > eps* or eps < eps*
print(f"\neach of the {E.size} states, evaluated on the branch it belongs to: "
      f"max |Dbar - Delta_branch(E)| = {np.nanmax(np.abs(Dbar - own_branch)):.1e}, NaN count = {np.isnan(own_branch).sum()}")
other = np.where(eps > estar, Dm, Dp)
print(f"states whose energy E_ks ALSO has a real second root (two-valued window): {np.isfinite(other).sum()} of {E.size} "
      f"({100 * np.isfinite(other).mean():.1f} %); for all the others the second root has |eps| > W and is discarded (NaN)")

# ------------------------------------------------------------------------------------------------
# 6. figure
# ------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(14.6, 4.5))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.30, 1.25], wspace=0.14)
ax_a, ax_b, ax_cb, ax_gap, ax_c = (fig.add_subplot(gs[j]) for j in range(5))
ax_gap.axis("off")
vmin, vmax = E.min(), E.max()
for i, (a, s) in enumerate(zip((ax_a, ax_b), (+1, -1))):
    im = a.imshow(E[i].T, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi], vmin=vmin, vmax=vmax, cmap="viridis")
    cs = a.contour(kx, ky, E[i], levels=[1.0, 2.0, 4.0, 6.0], colors="w", linewidths=0.7)
    a.clabel(cs, fmt="%g", fontsize=7)
    a.set_title(f"({'ab'[i]})  $s={s:+d}$\n$E_{{\\mathbf{{k}}s}}$: one number per point $(k_x,k_y)$", fontsize=9.8)
    a.set_xlabel(r"$k_x$  (dimensionless, $k_x a$)")
    a.set_xticks([-np.pi, 0, np.pi]); a.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    a.set_yticks([-np.pi, 0, np.pi])
    if i == 0:
        a.set_ylabel(r"$k_y$  (dimensionless, $k_y a$)"); a.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    else:
        a.set_yticklabels([])                       # same k_y axis as panel (a)
cb = fig.colorbar(im, cax=ax_cb)
cb.set_label(r"$E\,/\,t$  (an energy; white lines: $E/t=1,2,4,6$)", fontsize=9)

a = ax_c
Eg = np.linspace(E.min(), E.max() + 0.2, 2000)
Dpg, Dmg, _ = delta_of_E(Eg, D0, Ds, mu, VSO, t)
for i, (s, col) in enumerate(((+1, "#d1495b"), (-1, "#00798c"))):
    a.plot(E[i].ravel(), Dbar[i].ravel(), ".", ms=2.2, color=col, alpha=0.55, label=rf"states with $s={s:+d}$ ({Nk * Nk} points)")
a.plot(Eg, Dpg, "k-", lw=1.1, label=r"$\Delta_+(E)$, $\Delta_-(E)$ from (R) after the band test")
a.plot(Eg, Dmg, "k-", lw=1.1)
a.set_xlabel(r"$E\,/\,t$"); a.set_ylabel(r"$\bar\Delta_{\mathbf{k}s}\,/\,t$")
a.set_title("(c)  all 28,800 points of (a), (b) lie on ONE curve:\n$\\Delta$ depends on the single variable $E$", fontsize=9.8)
a.grid(alpha=0.25); a.legend(fontsize=7.6, loc="upper left")
fig.savefig("fig4_E_on_kgrid.png", dpi=170, bbox_inches="tight")
print("\nfigure written: fig4_E_on_kgrid.png")
