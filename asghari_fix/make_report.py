"""Writes REVIEW_AND_FIX.md.  All numbers come from the saved results of compare_old_new.py / validate_qp.py and from the repo's own JSON files (branch master)."""
import json
import re
from pathlib import Path
import numpy as np
from qp_absorption import RashbaDOS, absorption, sc_state_vs_T
from solve_gap import Model

M = Path("/home/user/asghari_master/project/")
REPO = M / "ultrasonic/results"
npz = np.load("new_T_curves.npz")
vscan = json.load(open("new_vso_scan.json"))
old_vscan = json.load(open(REPO / "vso_scan.json"))
stage1 = json.load(open(M / "sc_params/results/tc_vs_vso.json"))["results"]
V1 = np.array(stage1["V_SO"])
log = open("validate_qp_output.txt").read()


def grab(pattern):
    return float(re.search(pattern, log).group(1))


A = grab(r"A worst \|sum-1\| = ([0-9.e+-]+)")
B = grab(r"B worst rel\. diff = ([0-9.e+-]+)")
D2 = grab(r"D2 worst rel\. diff = ([0-9.e+-]+)")
Dpair = grab(r"pair channel just open: max \|R_es - R_bf\| = [0-9.]+ \(relative ([0-9.]+)\)")


def at(v, frac, key):
    return float(np.interp(frac, npz[f"V{v}_frac"], npz[f"V{v}_{key}"]))


def old_json(v):
    t = json.load(open(REPO / f"temp_sweep_VSO_{v:.4f}.json"))
    return np.array(t["T"]), np.array(t["ratio"], float), float(np.ravel(t["Tc"])[0]), np.array(t["mu"])


def old_at(v, frac):
    T, r, Tc, _ = old_json(v)
    return float(np.interp(frac, T / Tc, r))


rows = []
for v in (0.0, 0.4082, 0.8):
    for frac in (0.3, 0.5, 0.7, 0.9):
        rows.append(f"| {v} | {frac} | {old_at(v, frac):.4f} | {at(v, frac, 'I'):.2e} | {at(v, frac, 'II'):.2e} |")
tbl = "\n".join(rows)

# onset of the pair channel, end of the old sweeps, spectra
info = {}
for v in (0.0, 0.4082, 0.8):
    fr, Em = npz[f"V{v}_frac"], npz[f"V{v}_Emin"]
    idx = np.where(2 * Em < 0.7)[0]
    onset = 0.5 * (fr[idx[0] - 1] + fr[idx[0]])
    T, r, Tc, mu_old = old_json(v)
    i = int(np.argmin(abs(V1 - v)))
    info[v] = dict(onset=onset, old_end=T[-1] / Tc, Emin=Em[0], Deff=stage1["Deff_T0"][i], mu_old=mu_old[1], mu_sc=npz[f"V{v}_mu"][0],
                   muN=npz[f"V{v}_muN"][0], Tc1=stage1["Tc"][i], Tc_mine=float(npz[f"V{v}_Tc"]), W=4 * np.sqrt(1 + v ** 2 / 2))
inf_tbl = "\n".join(
    f"| {v} | {d['Tc1']:.4f} / {d['Tc_mine']:.4f} | {d['Deff']:.3f} | {d['Emin']:.3f} | {d['mu_old']:.3f} / {d['mu_sc']:.3f} / {d['muN']:.3f} | {d['old_end']:.2f} | {d['onset']:.2f} |"
    for v, d in info.items())

# effect of the reference chemical potential in W_N (V_SO = 0.4082)
v = 0.4082
m = Model(U=95.0, dt=4.5, n=1.885, VSO=v, Nk=1200)
dos = RashbaDOS(v, Nk=1200)
Tc, Ts, D0, Ds, mu, muN = sc_state_vs_T(m, [0.3, 0.5, 0.7, 0.9])
mu_eff = []
for T, d0, ds, mu_, mun in list(zip(Ts, D0, Ds, mu, muN))[1:]:
    a = absorption(dos, T, 0.7, d0, ds, mu_, mun)["ratio"]
    b = absorption(dos, T, 0.7, d0, ds, mu_, mu_)["ratio"]
    mu_eff.append((T / Tc, a, b))
mu_txt = "; ".join(f"T/Tc={f:.1f}: {a:.3g} (μ_N) vs {b:.3g} (μ_SC)" for f, a, b in mu_eff)
const_eff = []
for T, d0, ds, mu_, mun in list(zip(Ts, D0, Ds, mu, muN))[1:]:
    full = absorption(dos, T, 0.7, d0, ds, mu_, mun)["ratio"]
    Deff = np.sum(dos.w_i * np.abs(d0 - ds * dos.eps_i / 4)) / 2
    const = absorption(dos, T, 0.7, Deff, 0.0, mu_, mun)["ratio"]
    const_eff.append((T / Tc, const / full))
const_txt = ", ".join(f"×{r:.2f} at T/Tc = {f:.1f}" for f, r in const_eff)

# V_SO scan
vs = np.array(vscan["VSO"])
vo = np.array(old_vscan["VSO_arr"])
scan_rows = []
for f in ("0.3", "0.5", "0.7", "0.9"):
    r = np.array(vscan["ratio"][f]); ro = np.array(old_vscan["results"][f]["ratio"], float)
    pick = lambda x: r[int(np.argmin(abs(vs - x)))]
    valid_note = f", valid only for V_SO ≥ {vo[~np.isnan(ro)][0]:.2f}" if np.isnan(ro).any() else ""
    scan_rows.append(f"| {f} | {np.nanmin(ro):.3f} – {np.nanmax(ro):.3f} (maximum at V_SO = {vo[int(np.nanargmax(ro))]:.2f}{valid_note}) | {pick(0.0):.2e} | {pick(0.4):.2e} | {pick(0.8):.2e} |")
scan_tbl = "\n".join(scan_rows)
r9 = np.array(vscan["ratio"]["0.9"])
jump = int(np.argmax(r9[:-1] / r9[1:]))
step_txt = f"between V_SO = {vs[jump]:.2f} (W/W_N = {r9[jump]:.2f}) and {vs[jump + 1]:.2f} ({r9[jump + 1]:.2f})"

md = f"""# Review of the ultrasonic-absorption code and an energy-space replacement

Repo reviewed: `AmirhoseynpowAsghari/power-absorbed-in-the-presence-of-external-time-dependent-perturbation`, branch **master**, commit `cda0d56` ("ultrasonic"; line numbers refer to it).
(The default branch `main` has the same `absorption.py` but other stage-1 parameters; everything below is for `master`.) Dimensionless units (t = ħ = k_B = 1), as in your code.
Files of this review are in the folder `asghari_fix/`.

## 1. Summary

**The central problem** is in `project/ultrasonic/absorption.py`. The integration variable `E_grid` is the *normal-state band energy* ε
(the DOS `gE_table` is the DOS of ε_ks), energy conservation is imposed on it (`Ep_grid = E_grid + hbar*omega`, line 34), and the thermal factor is the
*normal-state* Fermi function of ε (`fermi_vec(·, T, mu)`, line 71). Superconductivity enters only through the coherence bracket with one constant gap.
The golden rule needs energy conservation and occupation for the **quasiparticle** energy E = sqrt(ξ² + Δ̄²) ≥ E_min, with f(E) = 1/(e^(E/T)+1) (no μ) and the
quasiparticle density of states (the Jacobian dε/dE). Consequence, visible in your saved results: W/W_N stays at 0.91 – 0.96 at *every* temperature and never
goes to zero as T → 0 (the `T < 1e-4 → 0.0` shortcut and the `2Δ > ħω` cut-off hide exactly this).

**The fix keeps the k-dependence** (what you asked for): every k-sum is a sum over ε_ks with the normal DOS N(ε), and the gap stays
Δ̄(ε) = Δ⁰ − Δˢ ε/4t instead of a k-average. For a given quasiparticle energy E there are two band energies ε_±(E) (both must lie inside the band),
and the delta function is solved on them with the Jacobian |dε/dE| = E / (sqrt(1+b²) sqrt(E²−E_min²)), b = Δˢ/4t. Both channels are included:
quasiparticle scattering and pair creation (zero below 2E_min, so no domain-of-validity cut is needed).

**What changes** (V_SO, T/Tc, ħω = 0.7, your stage-1 parameters U = 95, Δt = 4.5, n = 1.885; η = +1 ultrasound, η = −1 EM; "old" = the repo's saved results):

| V_SO | T/Tc | W/W_N, old (repo) | W/W_N, new, η=+1 | W/W_N, new, η=−1 |
|---|---|---|---|---|
{tbl}

Values above 1 are the pair-breaking channel, which opens when 2E_min(T) = ħω (a step, as at any BCS edge; see below).

![old vs new](fig_old_vs_new_T.png)

| V_SO | T_c: your stage 1 / my solver | Δ_eff(0) (your gap) | E_min(0) (true) | μ(T→0): your stage 2 / self-consistent / normal | your sweep stops at T/Tc | true pair-breaking onset T/Tc |
|---|---|---|---|---|---|---|
{inf_tbl}

**Validation** (`validate_qp.py`, log in `validate_qp_output.txt`):

| test | result |
|---|---|
| A. Δ → 0: scattering + pair creation must equal the normal-state absorption (checks both channels and all prefactors) | worst deviation {A:.1e} |
| B. textbook BCS limit (flat DOS, constant gap, ω → 0, η=+1): W_s/W_n = 2 f(Δ) | worst relative deviation {B:.1e} |
| D2. flat DOS with energy-dependent gap, pair channel open: analytic branches/Jacobians vs a 2-D energy-grid sum | worst relative deviation {D2:.1e} |
| D. all pairs of (k,s) states on a k-grid, with the helicity overlap ½(1+ss′cosΔθ), vs the energy-space formula; V_SO = 0.4082, T/Tc = 0.7 | brute force 0.119 / 0.122 / 0.124 / 0.124 for N = 64 / 96 / 128 / 160; energy space 0.131 / 0.129 / 0.127 for DOS smoothing 0.04 / 0.02 / 0.01 |
| D. same, T/Tc = 0.96 (pair channel just open) | agree to {100 * Dpair:.1f} % |
| C. limits: W/W_N → 0 exponentially as T → 0, → 1 at T_c | satisfied (table above and figure) |

Test D also shows that the helicity overlap averages to exactly ½ over constant-energy contours, so the energy-space result *is* the all-pairs sum including
the helicity form factor of the previous note. The residual 2–5 % in the lattice comparison is regularisation and finite-grid sampling (both sequences converge towards each other), not the formula.

## 2. Findings

### A. Conceptual (they change the results)

| # | where | problem | fix |
|---|---|---|---|
| A1 | `absorption.py` 34, 37, 54–61, 71–72; `lecture_note…md` §1; the diagnostics `visualize_integrand.py`, `phase_space_cutoff_test_for_repo.py`, `diagnose_full_VSO.py`, `dashboard.py`, `visualize_phase_space_cutoff.py` (they copy or call `compute_W` / `fermi_vec`) | band energy used as if it were the quasiparticle energy (see Summary): no BCS DOS, no f(E_qp), energy conservation on ε. `physics_ua.fermi_qp` (line 23, no μ) already exists but is not used by `compute_W` | `qp_absorption.absorption` |
| A2 | `absorption.py` 42–50; `physics_ua.py` 63–76 (`pairbreaking_valid`); `scanning_ua.py` 79–84, 161–167; `check_pairbreaking_*.py`; `absorption_u.py` | `T<1e-4 → 0`, NaN / stop when 2Δ_eff ≤ ħω. The threshold is 2E_min, not 2Δ_eff (table above: your sweeps stop at T/Tc = 0.74 / 0.84 / 0.94, the true onset is 0.81 / 0.91 / 0.98), and the missing pair channel is the real reason | include the pair channel; no cut needed |
| A3 | `absorption.py` 60–61, 67 | `xi_abs` = \\|ξ\\| discards the sign of ξ (the lecture note has the signed ξξ′; the paper's own step ε−μ = +√(E²−Δ²) has the same effect); wrong whenever ξξ′ < 0 (see `Energy_space_mathematics.pdf` §5) | the sign is carried by the branch ε_±(E) |
| A4 | `absorption.py` 69 (and `absorption_u.py`); note §1 | `+ η Δ²/(EE′)`. With M = B(uu′ − ηvv′) the coherence factor is ½[1 + ξξ′/EE′ − η ΔΔ′/EE′]; the code's η=+1 is the electromagnetic (case II, coherence-peak) factor | sign and ΔΔ′ (signed, not Δ²) |
| A5 | `gap_equations.py` 75–81; `sc_loader.py` (`'Delta0'` := Δ_eff); `physics_ua.py` 44–61 (`Delta_T`); `scanning_ua.py` 77 | one gap Δ_eff(T) for all k, Δ_eff = ⟨\\|Δ̄\\|⟩. The fitted shape (1−(T/Tc)^p)^q you introduced is a good improvement of the T-dependence, but the gap is still k-independent, whereas Δ̄(ε) runs from about +1.4 to −0.9 across the band (V_SO = 0.41) and changes sign; E_min is 13 – 49 % larger than Δ_eff (table above). Effect of the k-average alone, inside the *corrected* formula (same code with Δˢ = 0, Δ⁰ = Δ_eff; V_SO = 0.41): W/W_N changes by {const_txt}. So most of the old-vs-new difference comes from A1, but the gap average shifts E_min and the pair-breaking threshold and matters at low T | `sc_state_vs_T` (self-consistent Δ⁰(T), Δˢ(T), μ(T)) |
| A6 | `physics_ua.py` 78–135 (`compute_electron_density`, `find_mu`); `scanning_ua.py` 86, 88–90 | μ(T) from the number equation with the constant gap (table above: 3.73 vs 3.63 self-consistent at V_SO = 0.41), and the *same* μ in W_N. W_N must use the normal-state μ_N(T) at the same filling. Effect on the ratio at V_SO = 0.41: {mu_txt} | μ_N(T) from `Model.mu_normal` |
| A7 | `sc_params/solvers.py` 43–45, 64–66 (latent) | `abs(sol[0])`, `abs(sol[1])`: only the overall sign of (Δ⁰, Δˢ) is a gauge, the relative sign is physical. Harmless here (Δ⁰Δˢ > 0 at U = 95), but for parameter sets with a negative relative sign (I found one at U = 45) it silently changes the solution and the node of Δ̄ | store signed values |

### B. Numerical / housekeeping

| # | where | problem |
|---|---|---|
| B1 | `dos_ua.py` 24–25 | prefactor 1/(π^1.5 σ t) (as in HHM Eq. 18) with `norm = dk²/(2π)²` gives ∫g dE = 0.643 instead of 2 (a Gaussian representation of δ needs 1/(√π σ t)). Ratios are unaffected, absolute values are |
| B2 | `kgrid_ua.py` 11–15; `sc_params/kgrid.py` 10–11; `parameters_tc.py` 17 (`Nk = 80`) | `linspace(-π, π, N)` contains both ±π; `linspace(0, π, Nk)` gives full weight to the mirror-plane points; 80 × 80 is coarse. My solver on a fine midpoint grid differs from your stage 1 by +2.9 % / +1.9 % / +0.4 % in T_c at V_SO = 0 / 0.41 / 0.8; Δ⁰, Δˢ within 2 % |
| B3 | README (U/t = 75, "U = −75t" in titles) vs `parameters_tc.py` (`U_over_t = 95`, matches the paper's U = 95t V_SO scan) | the gap equations use +U (repulsive, as in HHM); please make README, titles and parameters agree |
| B4 | `parameters_u.py` (stage 3, master) | V_SO = 0, n = 0.8, U ∈ [35, 60]; the README says V_SO = 0.4, n = 1.885, U ∈ [75, 125]; `ultrasonic_U/results/` mixes U = 15 … 125. I did not review stages 3–4 beyond noting they use the same `compute_W` |
| B5 | `absorption.py` 74 | `mask = abs(E_grid) > eps` has no meaning for a band-energy grid |
| B6 | `absorption.py` 84, `scanning_ua.py` 90 | `abs(W/WN)` hides sign errors; define the thermal factor as f(E) − f(E+ħω) ≥ 0 and assert positivity |

### C. Physics remarks

* **C1 – the lecture note.** Its explanation of the structure of W/W_N versus V_SO (g(E)g(E+ħω) sampled in the window [μ, μ+ħω] near a Rashba band edge) is a property of the old integrand (A1), not of the physics.
  Your old curves (repo, T/Tc = 0.3 … 0.7) have a broad maximum near V_SO ≈ 0.5. With the corrected formula, W/W_N at fixed T/Tc is smooth and monotonic in V_SO, dominated by exp(−E_min/T):

| T/Tc | old formula: range over V_SO | new, V_SO = 0 | new, V_SO = 0.4 | new, V_SO = 0.8 |
|---|---|---|---|---|
{scan_tbl}

  The only sharp feature is at T/Tc = 0.9: a step {step_txt}, where 2E_min(T) crosses ħω (pair-breaking onset; broadening by a finite quasiparticle lifetime would round it). I have not repeated the U scan.
* **C2 – q and the sums.** The corrected code keeps the paper's assumptions: constant B, unrestricted sums over k and k′ (momentum not conserved: dirty limit; the helicity overlap averages to ½), ħω = 0.7 as a model frequency (as sound it would need q·a ≈ 43, outside the zone).
  Since ħω is comparable to 2E_min here, the pair channel matters near T_c; for ultrasound with ħω ≪ 2E_min only the scattering part (the textbook quantity) survives.

## 3. The energy-space formulas (what `qp_absorption.py` computes)

With ξ = ε−μ, Δ̄(ε) = Δ⁰ − Δˢε/4t, E = sqrt(ξ²+Δ̄²), f(E) = 1/(e^(E/T)+1), N(ε) the normal DOS (∫N = 2, both helicities),
C = ½[1 + ξξ′/EE′ − ηΔ̄Δ̄′/EE′] and C_p = ½[1 − ξξ′/EE′ + ηΔ̄Δ̄′/EE′]:

```
S_sc   = ∫dε_b N(ε_b) Σ_± N(ε_a^±) [E_a / (sqrt(1+b²) sqrt(E_a²−E_min²))] C      [f(E_b) − f(E_a)],   E_a = E_b + ħω
S_pair = ∫dε_b N(ε_b) Σ_± N(ε_a^±) [E_a / (sqrt(1+b²) sqrt(E_a²−E_min²))] C_p [1 − f(E_a) − f(E_b)],  E_a = ħω − E_b,  E_b < ħω/2
S_N    = ∫dε   N(ε) N(ε+ħω) [f_N(ε) − f_N(ε+ħω)],           f_N with the normal-state μ_N(T)
W/W_N  = (S_sc + S_pair) / S_N
ε_a^± = ε* ± sqrt(E_a²−E_min²)/sqrt(1+b²),   ε* = (μ + bΔ⁰)/(1+b²),   E_min = |Δ⁰ − bμ|/sqrt(1+b²),   b = Δˢ/4t,   both roots must satisfy |ε| ≤ W
```

Implementation note: in `absorption()` the summation variable of the state b is the band energy ε_b on the grid `dos.eps_i` (all of [−W, W]); the state a is eliminated by the delta function. This is an exact change of variables of the integral over the quasiparticle energy, ∫dE Σ_σ 𝒩_σ(E) … over [E_min, E_max] with 𝒩_σ(E) = N(ε_σ(E)) E/(sqrt(1+b²) sqrt(E²−E_min²)) and E_max = max(E(−W), E(+W)) (not ∞, since there are no states beyond the band). `absorption_E()` implements that literal E-form; the two agree to 2·10⁻⁴ (scattering) and 2·10⁻³ (pair creation), see `verify_three_points.py`.

Derivation and the band test are in `Delta_of_E_derivation.pdf`; the coherence factors (and the helicity overlap) in `Helicity_matrix_element.pdf`.

## 4. How to plug it into your code (stages 2 and 4)

Add `qp_absorption.py` and `solve_gap.py` to `project/ultrasonic/` (numpy + scipy only). In `scanning_ua.py` replace the body of the temperature loop
(lines 76–98, and the analogous block in `vso_scan`) by

```python
from qp_absorption import RashbaDOS, absorption, sc_state_vs_T
from solve_gap import Model
m   = Model(U=95.0, dt=4.5, n=n_target, VSO=VSO, Nk=1200)
dos = RashbaDOS(VSO, Nk=1200)
Tc, Ts, D0, Ds, mu, muN = sc_state_vs_T(m, fracs)               # self-consistent Delta0(T), Delta_s(T), mu(T), mu_N(T)
for T, d0, ds, mu_, mun in zip(Ts, D0, Ds, mu, muN):
    ratio = absorption(dos, T, omega, d0, ds, mu_, mun, eta=eta)["ratio"]     # eta = +1 ultrasound, -1 EM
```

`physics_ua.Delta_T`, `find_mu`, `pairbreaking_valid`, `absorption.compute_W` and the `T<1e-4` shortcut are then no longer needed.
If you prefer to keep your own stage-1 solver, it must store signed Δ⁰(T), Δˢ(T) and μ(T) (A7) instead of Δ_eff only, and use a finer, midpoint k-grid (B2).

## 5. Reproduce

```
python3 validate_qp.py                       # tests A–D2 (about 1 min)
python3 compare_old_new.py <path to project/ultrasonic/results>     # old vs new, writes the two figures
python3 run_new_only.py 95                   # corrected curves only, for any U
python3 verify_three_points.py              # integration variable/limits, absence of q, energy-dependent gap (direct checks on the code)
python3 make_report.py                       # regenerates this file
```
"""
open("REVIEW_AND_FIX.md", "w").write(md)
print("written", len(md), "characters")
print(info)
print(mu_txt)
print(step_txt)
