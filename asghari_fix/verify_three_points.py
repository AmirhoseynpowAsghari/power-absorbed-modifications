"""Direct verification of three points about qp_absorption.py   (python3 verify_three_points.py, about 20 s)

 1. integration variable and limits (band energy eps vs quasiparticle energy E)
 2. is the momentum transfer q in the calculation?
 3. is the energy-dependent gap  Dbar(eps) = D0(T) - Ds(T) eps / 4t  used, with D0(T), Ds(T) from the self-consistent solution?
"""
import inspect
import re
import numpy as np
import qp_absorption as qa
from qp_absorption import RashbaDOS, absorption, absorption_E, sc_state_vs_T, qp_constants, fermi_qp
from solve_gap import Model

VSO, U, DT, NFILL, OMEGA = 0.4082, 95.0, 4.5, 1.885, 0.7
m = Model(U=U, dt=DT, n=NFILL, VSO=VSO, Nk=1200)
dos = RashbaDOS(VSO, Nk=1200)
fracs = [0.3, 0.7, 0.96]
Tc, Ts, D0, Ds, mu, muN = sc_state_vs_T(m, fracs)
cases = [(2, "T/Tc = 0.70  (below the pair-breaking onset)"), (3, "T/Tc = 0.96  (pair channel open)")]
src = open("qp_absorption.py").read().splitlines()


def show(pattern, fname="qp_absorption.py", lines=None, first=None):
    """print the lines of a source file matching `pattern`, with their CURRENT line numbers"""
    lines = lines if lines is not None else open(fname).read().splitlines()
    found = [(n + 1, l) for n, l in enumerate(lines) if re.search(pattern, l)]
    for n, l in (found if first is None else found[:first]):
        print(f"  {fname}:{n:<4d} {l.strip()[:118]}")
    if not found:
        print(f"  (no line of {fname} matches {pattern!r})")


def code_lines_with(word, fname="qp_absorption.py"):
    """line numbers of CODE (not comments / docstrings) that contain `word`"""
    out, in_doc = [], False
    for n, l in enumerate(open(fname).read().splitlines(), 1):
        stripped = l.strip()
        if stripped.count('"""') % 2 == 1:
            in_doc = not in_doc
            continue
        if in_doc or stripped.startswith("#"):
            continue
        if re.search(rf"\b{word}\b", l.split("#")[0]):
            out.append(n)
    return out

print("=" * 110)
print(f"parameters: U = {U}, dt = {DT}, n = {NFILL}, V_SO = {VSO}, hbar*omega = {OMEGA}   (t = hbar = k_B = 1);  Tc = {Tc:.4f};  band half-width W = {dos.W:.4f}")
print("=" * 110)

# ------------------------------------------------------------------------------------------------------------------
print("\nPOINT 1  --  integration variable and limits")
print("-" * 110)
print("code (qp_absorption.absorption):")
for pat in (r"eb, wb = dos\.eps_i", r"Ea = Eb \+ omega", r"ell_a = np\.sqrt", r"jac = Ea", r"ea = P\[\"estar\"\] \+ sgn \* ell_a", r"S_sc \+= np\.sum", r"sel = Eb < 0\.5", r"Eas = omega - Ebs",
            r"ea = P\[\"estar\"\] \+ sgn \* ell /", r"S_pair \+= np\.sum"):
    show(pat, first=1)
print(f"\n  state b: sum over the band-energy grid dos.eps_i : {len(dos.eps_i)} points from eps = {dos.eps_i[0]:+.4f} to {dos.eps_i[-1]:+.4f}  (spacing {dos.eps_i[1] - dos.eps_i[0]:.5f}),  w_i = N(eps_i) d_eps")
print("  state a: NOT integrated; the delta function fixes E_a (= E_b + omega, or omega - E_b) and the two roots eps_a^+- carry the Jacobian")
for i, name in cases:
    T, d0, ds, mu_, mun = Ts[i], D0[i], Ds[i], mu[i], muN[i]
    P = qp_constants(d0, ds, mu_)
    Emax = max(qa.qp_energy(-dos.W, d0, ds, mu_), qa.qp_energy(dos.W, d0, ds, mu_))
    r = absorption(dos, T, OMEGA, d0, ds, mu_, mun, eta=+1, diag=True)
    Eb = r["internals"]["Eb"]
    sel = Eb < 0.5 * OMEGA
    half = np.sqrt(max(0.25 * OMEGA ** 2 - P["Emin"] ** 2, 0.0)) / P["s"]
    print(f"\n  [{name}]   Emin = {P['Emin']:.4f}, Emax = max(E(-W), E(+W)) = {Emax:.4f}, eps* = {P['estar']:.4f}")
    print(f"     scattering: eps_b over the whole band [-W, W]  <=>  E_b in [Emin, Emax] = [{P['Emin']:.4f}, {Emax:.4f}];  E_a = E_b + omega;  a-roots outside the band are dropped by N(eps)=0")
    if sel.any():
        print(f"     pair creation: only grid points with E_b < omega/2:  E_b in [Emin, omega/2) = [{P['Emin']:.4f}, {0.5 * OMEGA:.4f})  <=>  eps_b in ({P['estar'] - half:.4f}, {P['estar'] + half:.4f}),  {int(sel.sum())} grid points;  E_a = omega - E_b")
    else:
        print(f"     pair creation: no grid point has E_b < omega/2 = {0.5 * OMEGA:.3f} (since Emin = {P['Emin']:.4f} >= omega/2): the term is exactly 0")
    for eta in (+1, -1):
        a = absorption(dos, T, OMEGA, d0, ds, mu_, mun, eta=eta)          # integration variable eps_b (band energy)
        b = absorption_E(dos, T, OMEGA, d0, ds, mu_, mun, eta=eta)        # integration variable E (explicit limits)
        rel = lambda x, y: abs(x - y) / max(abs(x), 1e-300) if x != 0 else abs(y)
        print(f"     eta={eta:+d}:  S_sc   eps-variable {a['S_sc']:.6e}   E-variable {b['S_sc']:.6e}   rel.diff {rel(a['S_sc'], b['S_sc']):.1e}")
        print(f"             S_pair eps-variable {a['S_pair']:.6e}   E-variable {b['S_pair']:.6e}   rel.diff {rel(a['S_pair'], b['S_pair']):.1e}   ->  W/W_N = {a['ratio']:.5f} vs {b['ratio']:.5f}")

# ------------------------------------------------------------------------------------------------------------------
print("\n\nPOINT 2  --  momentum transfer q")
print("-" * 110)
print("  signature absorption  :", inspect.signature(absorption))
print("  signature absorption_E:", inspect.signature(absorption_E))
hits = [(n + 1, l.strip()) for n, l in enumerate(src) if re.search(r"(?<![A-Za-z_0-9])q(?![A-Za-z_0-9])|\bqa\b|\bq_", l)]
print(f"  occurrences of a variable named q / qa / q_... in qp_absorption.py (code and comments): {len(hits)}")
print("  code lines containing omega (it enters only through the energy-conservation condition and through the shift in W_N):")
for n in code_lines_with("omega"):
    print(f"      qp_absorption.py:{n:<4d} {src[n - 1].strip()[:110]}")
import glob
calc = ["qp_absorption.py", "solve_gap.py", "compare_old_new.py", "run_new_only.py", "validate_qp.py"]
other = [f for f in calc if re.search(r"\b43\b|\bqa\b|(?<![A-Za-z_0-9])q(?![A-Za-z_0-9])", open(f).read())]
print("  numerical modules that contain a q, qa or the number 43 anywhere:", other if other else "none  (qa ~ 43 appears only in explanatory text: make_report.py -> REVIEW_AND_FIX.md, section C2)")

# the distribution of the momentum transfer carried by the pairs that DO contribute (all-pairs sum on a 48 x 48 k-grid)
Nk = 48
k = (np.arange(Nk) + 0.5) * 2 * np.pi / Nk - np.pi
kx, ky = np.meshgrid(k, k, indexing="ij"); kx, ky = kx.ravel(), ky.ravel()
R = np.hypot(np.sin(kx), np.sin(ky)); e0 = -2 * (np.cos(kx) + np.cos(ky))
eps = np.concatenate([e0 - 2 * s * VSO * R for s in (+1, -1)])
KX, KY = np.concatenate([kx, kx]), np.concatenate([ky, ky])
i = 2
xi = eps - mu[i]; Db = D0[i] - Ds[i] * eps / 4; E = np.hypot(xi, Db); f = fermi_qp(E, Ts[i])
qs, ws = [], []
for i0 in range(0, eps.size, 400):
    sl = slice(i0, min(i0 + 400, eps.size))
    EaEb = E[sl, None] * E[None, :]
    C = 0.5 * (1 + xi[sl, None] * xi[None, :] / EaEb - Db[sl, None] * Db[None, :] / EaEb)
    w = C * (f[None, :] - f[sl, None]) * np.exp(-((E[sl, None] - E[None, :] - OMEGA) / 0.05) ** 2)
    dqx = (KX[sl, None] - KX[None, :] + np.pi) % (2 * np.pi) - np.pi
    dqy = (KY[sl, None] - KY[None, :] + np.pi) % (2 * np.pi) - np.pi
    q = np.hypot(dqx, dqy)
    keep = w > 1e-14 * max(w.max(), 1e-300)
    qs.append(q[keep]); ws.append(w[keep])
qs, ws = np.concatenate(qs), np.concatenate(ws)
o = np.argsort(qs); cw = np.cumsum(ws[o]) / ws.sum()
pct = lambda p: qs[o][np.searchsorted(cw, p)]
print(f"\n  all-pairs sum (T/Tc = 0.7, scattering term, {eps.size} states): share of the absorbed weight carried by momentum transfers |k - k'| a")
print(f"     10 % of the weight at |q|a < {pct(0.1):.2f},  50 % at < {pct(0.5):.2f},  90 % at < {pct(0.9):.2f}   (zone corner: |q|a = {np.pi * np.sqrt(2):.2f});  weight with |q|a < 0.1: {ws[qs < 0.1].sum() / ws.sum():.3f}")

# ------------------------------------------------------------------------------------------------------------------
print("\n\nPOINT 3  --  the gap")
print("-" * 110)
print("code (qp_absorption.absorption): the gap enters only through these lines")
show(r"Dbb = D0 - Ds \* eb")
show(r"xia, Dba = ea - mu, D0 - Ds \* ea")
print("  code (qp_absorption.sc_state_vs_T): the state comes from the self-consistent solver")
show(r"model\.solve_low_T|model\.solve_at|model\.mu_normal")
print("  qp_absorption.sc_state_vs_T  ->  solve_gap.Model.solve_at(T, x0): root of the three self-consistency equations (HHM Eqs. 32, 33, 38):")
for ln, l in enumerate(open("solve_gap.py").read().splitlines(), 1):
    if 91 <= ln <= 102:
        print(f"  {ln:4d}  {l.strip()[:120]}")
print("\n  (a) D0(T), Ds(T), mu(T) used above are self-consistent:")
x00 = np.array([D0[0], Ds[0], mu[0]])
for j in range(len(Ts)):
    res = m.residual(np.array([D0[j], Ds[j], mu[j]]), Ts[j])
    print(f"     T = {Ts[j]:.4f} (T/Tc = {Ts[j] / Tc:.2f}):  D0 = {D0[j]:+.5f}  Ds = {Ds[j]:+.5f}  mu = {mu[j]:.5f}   max|residual of the 3 equations| = {np.max(np.abs(res)):.1e}")
print("     not a rigid scaling:  D0(T)/D0(Tlow) =", [round(float(D0[j] / D0[0]), 4) for j in range(1, len(Ts))], "  Ds(T)/Ds(Tlow) =", [round(float(Ds[j] / Ds[0]), 4) for j in range(1, len(Ts))])
print("\n  (b) the gap used inside absorption() is Dbar(eps) at the band energies of the two roots; check against the closed forms Delta_pm(E) = Delta* -+ b ell/sqrt(1+b^2):")
for i, name in cases:
    d0, ds, mu_ = D0[i], Ds[i], mu[i]
    r = absorption(dos, Ts[i], OMEGA, d0, ds, mu_, muN[i], diag=True)["internals"]
    P = r["P"]; b_ = P["b"]; Dstar = P["DF"] / (1 + b_ * b_)
    err_gap, err_E = 0.0, 0.0
    for sgn in (+1, -1):
        Dba, ea = r["branches"][sgn]["Dba"], r["branches"][sgn]["ea"]
        closed = Dstar - sgn * b_ * r["ell_a"] / P["s"]                   # Delta_+(E_a) for sgn=+1, Delta_-(E_a) for sgn=-1
        err_gap = max(err_gap, np.max(np.abs(Dba - closed)))
        err_E = max(err_E, np.max(np.abs(np.hypot(ea - mu_, Dba) - r["Ea"])))
    mid = len(r["Ea"]) // 2
    print(f"     [{name}] max|Dbar(eps_pm(E_a)) - Delta_pm(E_a)| = {err_gap:.1e};  max|sqrt(xi^2 + Dbar^2) - E_a| = {err_E:.1e}   (over {len(r['Ea'])} states, both branches)")
    print(f"        example (E_a = {r['Ea'][mid]:.3f}):  Delta_+ = {r['branches'][+1]['Dba'][mid]:+.4f} at eps_+ = {r['branches'][+1]['ea'][mid]:.3f}   Delta_- = {r['branches'][-1]['Dba'][mid]:+.4f} at eps_- = {r['branches'][-1]['ea'][mid]:.3f}")
for i in (1, 2):
    Dbar_band = D0[i] - Ds[i] * dos.eps_i / 4
    Deff = np.sum(dos.w_i * np.abs(Dbar_band)) / 2
    full = absorption(dos, Ts[i], OMEGA, D0[i], Ds[i], mu[i], muN[i])
    const = absorption(dos, Ts[i], OMEGA, Deff, 0.0, mu[i], muN[i])          # the same code with Ds = 0, D0 = <|Dbar|>: a constant gap
    print(f"\n  (c) T/Tc = {Ts[i] / Tc:.1f}: Dbar(eps) over the band runs from {Dbar_band.min():+.3f} to {Dbar_band.max():+.3f} (node at eps_0 = {4 * D0[i] / Ds[i]:.3f});  <|Dbar|> = {Deff:.3f} is NOT used anywhere in the code")
    print(f"      same code, full gap Dbar(eps):      Emin = {full['Emin']:.4f},  W/W_N = {full['ratio']:.5e}")
    print(f"      same code, constant gap <|Dbar|>:   Emin = {const['Emin']:.4f},  W/W_N = {const['ratio']:.5e}   (what a Delta_eff calculation of the SAME formula would give; ratio {const['ratio'] / full['ratio']:.2f})")
