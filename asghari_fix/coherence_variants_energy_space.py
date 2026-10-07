"""Sign choices in the coherence factor, inside the QUASIPARTICLE-energy framework (scattering + pair creation, f(E), physical W_N).

   scattering:    C  = 1/2 [ 1 + X/(E_a E_b) + s*eta*D_a D_b/(E_a E_b) ]
   pair creation: Cp = 1/2 [ 1 - X/(E_a E_b) - s*eta*D_a D_b/(E_a E_b) ]
   X = xi_a xi_b (signed)  or  |xi_a||xi_b|;   s = -1 (derived from M = B(uu' - eta vv'))  or  s = +1 (the paper's final formula)

   python3 coherence_variants_energy_space.py
"""
import numpy as np
from qp_absorption import RashbaDOS, sc_state_vs_T, qp_constants, fermi_qp, fermi_normal, absorption
from solve_gap import Model


def S_parts(dos, T, omega, D0, Ds, mu, eta, xi_mode, s_sign, pair=True, t=1.0):
    """(S_sc, S_pair) for the chosen variant (same integrals as qp_absorption.absorption)."""
    P = qp_constants(D0, Ds, mu, t)
    eb, wb = dos.eps_i, dos.w_i
    xib, Dbb = eb - mu, D0 - Ds * eb / (4.0 * t)
    Eb = np.maximum(np.hypot(xib, Dbb), 1e-12)
    X = lambda xa, xb: xa * xb if xi_mode == "signed" else np.abs(xa) * np.abs(xb)
    # scattering
    Ea = Eb + omega
    ell = np.sqrt(Ea ** 2 - P["Emin"] ** 2)
    jac = Ea / (P["s"] * ell)
    S_sc = 0.0
    for sg in (+1, -1):
        ea = P["estar"] + sg * ell / P["s"]
        xia, Dba = ea - mu, D0 - Ds * ea / (4.0 * t)
        C = 0.5 * (1.0 + X(xia, xib) / (Eb * Ea) + s_sign * eta * Dbb * Dba / (Eb * Ea))
        S_sc += np.sum(wb * dos.N(ea) * jac * C * (fermi_qp(Eb, T) - fermi_qp(Ea, T)))
    # pair creation
    S_pair = 0.0
    sel = Eb < 0.5 * omega
    if pair and np.any(sel):
        Ebs, xibs, Dbbs, wbs = Eb[sel], xib[sel], Dbb[sel], wb[sel]
        Eas = omega - Ebs
        ell = np.sqrt(np.maximum(Eas ** 2 - P["Emin"] ** 2, 1e-300))
        jac = Eas / (P["s"] * ell)
        occ = 1.0 - fermi_qp(Eas, T) - fermi_qp(Ebs, T)
        for sg in (+1, -1):
            ea = P["estar"] + sg * ell / P["s"]
            xia, Dba = ea - mu, D0 - Ds * ea / (4.0 * t)
            Cp = 0.5 * (1.0 - X(xia, xibs) / (Ebs * Eas) - s_sign * eta * Dbbs * Dba / (Ebs * Eas))
            S_pair += np.sum(wbs * dos.N(ea) * jac * Cp * occ)
    return S_sc, S_pair


def main():
    VSO, OMEGA, eta = 0.4082, 0.7, +1
    m = Model(U=95.0, dt=4.5, n=1.885, VSO=VSO, Nk=1200)
    dos = RashbaDOS(VSO, Nk=1200)
    fr = [0.3, 0.5, 0.7, 0.9, 0.96]
    Tc, Ts, D0, Ds, mu, muN = sc_state_vs_T(m, fr)
    print(f"V_SO = {VSO}, hbar*omega = {OMEGA}, eta = +1, U = 95 (self-consistent Delta0(T), Delta_s(T), mu(T)); physical normal state W_N (mu_N(T)); pair-breaking onset at T/Tc = 0.91\n")

    # --- sum rule Delta -> 0:  (scattering + pair creation) must equal the normal-state absorption ---------------------------------
    T0 = Ts[3]                                   # index 0 is T = 0.01; fr = [0.3, 0.5, 0.7, ...]  ->  index 3 is T/Tc = 0.7
    S_N0 = np.sum(dos.w_i * dos.N(dos.eps_i + OMEGA) * (fermi_normal(dos.eps_i, T0, muN[3]) - fermi_normal(dos.eps_i + OMEGA, T0, muN[3])))
    print("sum rule at Delta = 0 (T/Tc = 0.7):   (S_sc + S_pair) / S_N  with  S_N = physical normal-state absorption")
    for name, xm in (("signed xi xi'  ", "signed"), ("|xi||xi'|      ", "abs")):
        a, b = S_parts(dos, T0, OMEGA, 0.0, 0.0, muN[3], eta, xm, -1)
        print(f"   {name}: scattering {a / S_N0:.4f} + pair creation {b / S_N0:.4f} = {(a + b) / S_N0:.4f}")

    variants = [("(1) signed xi xi', -eta D D'   CORRECT", "signed", -1, True), ("(2) |xi||xi'|,   -eta D D'", "abs", -1, True),
                ("(3) signed xi xi', +eta D D'", "signed", +1, True), ("(4) |xi||xi'|,   +eta D D'   paper's final (no pair term)", "abs", +1, False)]
    print("\nW/W_N  (physical W_N):")
    print(f"{'T/Tc':>5} {'2Emin':>6} | " + " | ".join(f"{v[0][:34]:>34}" for v in variants))
    for i in range(1, len(Ts)):
        T = Ts[i]
        S_N = np.sum(dos.w_i * dos.N(dos.eps_i + OMEGA) * (fermi_normal(dos.eps_i, T, muN[i]) - fermi_normal(dos.eps_i + OMEGA, T, muN[i])))
        Em = qp_constants(D0[i], Ds[i], mu[i])["Emin"]
        vals = []
        for name, xm, sg, pair in variants:
            a, b = S_parts(dos, T, OMEGA, D0[i], Ds[i], mu[i], eta, xm, sg, pair)
            vals.append((a + b) / S_N)
        ref = absorption(dos, T, OMEGA, D0[i], Ds[i], mu[i], muN[i], eta=eta)["ratio"]
        assert abs(vals[0] - ref) < 1e-12 * max(1, ref), (vals[0], ref)               # variant (1) is exactly qp_absorption.absorption
        print(f"{T / Tc:5.2f} {2 * Em:6.3f} | " + " | ".join(f"{v:34.5f}" for v in vals))
    print("variant (1) reproduces qp_absorption.absorption to 1e-12")

    # --- is W/W_N < 1 ?   (eta = +1, correct formula) for several frequencies --------------------------------------------------------
    print("\ncorrect formula, eta = +1:  max over T < Tc of W/W_N, and the temperature where the pair channel opens (2 Emin(T) = hbar*omega)")
    fr2 = np.concatenate([np.linspace(0.05, 0.9, 35), np.linspace(0.905, 0.9995, 120)])
    Tc2, Ts2, D02, Ds2, mu2, muN2 = sc_state_vs_T(m, list(fr2))
    Em2 = np.array([qp_constants(D02[i], Ds2[i], mu2[i])["Emin"] for i in range(len(Ts2))])
    for om in (0.7, 0.2, 0.05):
        R = np.array([absorption(dos, Ts2[i], om, D02[i], Ds2[i], mu2[i], muN2[i], eta=+1)["ratio"] for i in range(1, len(Ts2))])
        idx = np.where(2 * Em2[1:] < om)[0]
        onset = (Ts2[1:][idx[0]] / Tc2) if len(idx) else float("nan")
        print(f"   hbar*omega = {om:4.2f}: max W/W_N = {np.nanmax(R):.3f} at T/Tc = {Ts2[1:][int(np.nanargmax(R))] / Tc2:.4f};   below the onset (T/Tc < {onset:.4f}) max = {np.nanmax(R[:idx[0]]) if len(idx) else np.nanmax(R):.3f}")


if __name__ == "__main__":
    main()
