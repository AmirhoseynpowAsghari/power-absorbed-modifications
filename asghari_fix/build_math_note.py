"""Builds Energy_space_mathematics.pdf / .html  (derivations for three questions).   python3 build_math_note.py"""
import re
import numpy as np
import sys
sys.path.insert(0, "..")
import doclib
from doclib import T, M, D, NUM
from qp_absorption import RashbaDOS, absorption, absorption_E, sc_state_vs_T, qp_constants, qp_energy, fermi_normal
from solve_gap import Model


def sci(x):
    if x == 0:
        return "0"
    mant, ex = f"{x:.0e}".split("e")
    return f"{int(mant)}&times;10<sup>&minus;{abs(int(ex))}</sup>"


# ------------------------------------------------------------------------------------------------------------------
# numbers quoted in the note (recomputed here)
# ------------------------------------------------------------------------------------------------------------------
VSO = 0.4082
m = Model(U=95.0, dt=4.5, n=1.885, VSO=VSO, Nk=1200)
dos = RashbaDOS(VSO, Nk=1200, nbins=6000)
Tc, Ts, D0, Ds, mu, muN = sc_state_vs_T(m, [0.7, 0.96])
d0, ds, mu_ = D0[1], Ds[1], mu[1]
P = qp_constants(d0, ds, mu_)
b, kap, Em, es, DF = P["b"], P["s"], P["Emin"], P["estar"], P["DF"]
Dst = DF / (1 + b * b)
xst = es - mu_
rng = np.random.default_rng(3)
err = np.zeros(2)
for _ in range(20000):
    Ea, Eb = Em + rng.uniform(0.01, 6, 2)
    sa, sb = rng.choice([-1, 1], 2)
    la, lb = np.sqrt(Ea ** 2 - Em ** 2), np.sqrt(Eb ** 2 - Em ** 2)
    ea, eb = es + sa * la / kap, es + sb * lb / kap
    xia, xib, Da, Db = ea - mu_, eb - mu_, d0 - ds * ea / 4, d0 - ds * eb / 4
    plus = Em ** 2 + sa * sb * la * lb
    minus = -(1 - b * b) * Dst ** 2 + 2 * b * Dst * (sa * la + sb * lb) / kap + sa * sb * (1 - b * b) * la * lb / kap ** 2
    err = np.maximum(err, [abs(xia * xib + Da * Db - plus), abs(xia * xib - Da * Db - minus)])
Emax = max(qp_energy(-dos.W, d0, ds, mu_), qp_energy(dos.W, d0, ds, mu_))
x = np.linspace(0, np.sqrt(Emax - Em), 400001)
E = Em + x ** 2
ell = np.sqrt(np.maximum(E ** 2 - Em ** 2, 0))
idn = []
for F in (lambda e: np.cos(e) * np.exp(-e ** 2 / 8), lambda e: np.exp(-(e - 3) ** 2), lambda e: e ** 2):
    lhs = np.sum(dos.w_i * F(dos.eps_i))
    rhs = sum(np.trapezoid(dos.N(es + sg * ell / kap) * E * 2.0 / (kap * np.sqrt(E + Em)) * F(es + sg * ell / kap), x) for sg in (+1, -1))
    idn.append(abs(lhs - rhs) / abs(lhs))
rel = {}
for i, key in ((1, "sc7"), (2, "sc96")):
    a = absorption(dos, Ts[i], 0.7, D0[i], Ds[i], mu[i], muN[i])
    bb = absorption_E(dos, Ts[i], 0.7, D0[i], Ds[i], mu[i], muN[i])
    rel[key] = abs(a["S_sc"] - bb["S_sc"]) / a["S_sc"]
    rel["pair" + key] = abs(a["S_pair"] - bb["S_pair"]) / a["S_pair"] if a["S_pair"] > 0 else 0.0
log = open("verify_three_points_output.txt").read()
q10, q50, q90 = (float(v) for v in re.search(r"10 % of the weight at \|q\|a < ([0-9.]+),  50 % at < ([0-9.]+),  90 % at < ([0-9.]+)", log).groups())
NUM.update(errA=sci(max(err)), idn1=sci(idn[0]), idn3=sci(idn[2]), relsc=sci(max(rel["sc7"], rel["sc96"])), relpair=sci(rel["pairsc96"]),
           q10=f"{q10:.2f}", q50=f"{q50:.2f}", q90=f"{q90:.2f}", Emin=f"{Em:.3f}", Emax=f"{Emax:.2f}", W=f"{dos.W:.3f}")

# ------------------------------------------------------------------------------------------------------------------
H = []
H.append("<h1>Energy-space absorption: the mathematics</h1>")
H.append("<div class='sub'>Three questions: the change of variables &epsilon; &rarr; <i>E</i>, the momentum transfer <i>q</i> and the dirty approximation, and the two gap values &Delta; and &Delta;&prime;</div>")
H.append("<div class='ref'>Model and notation of HHM, arXiv:1801.08988 (&Delta;&#772;<sub>ks</sub> = &Delta;<sup>0</sup> &minus; &Delta;<sup>s</sup>&epsilon;<sub>ks</sub>/4t, Eq. 37; "
         "E<sub>ks</sub> = &radic;(&xi;&sup2; + |&Delta;<sub>ks</sub>|&sup2;), Eq. 26); absorption formalism of arXiv:2607.02812. Equations are numbered (1), (2), &hellip;; "
         "&ldquo;state <i>a</i>&rdquo; = (<b>k</b>,s), &ldquo;state <i>b</i>&rdquo; = (<b>k</b>&prime;,s&prime;).</div>")

H.append("<div class='box'>" + T(
    r"<b>Answers in one line each.</b> (1) The sum over states can be written over $\varepsilon$ or over $E$: it is the <i>same integral</i> (two branches, Jacobian $E/\kappa\ell$, limits $E_{\min}\le E\le E_{\max}$). "
    r"What is <i>not</i> equivalent is imposing energy conservation on $\varepsilon$ instead of $E$. "
    r"(2) $q$ enters only through the Kronecker delta $\delta_{\mathbf{k},\mathbf{k}'+\mathbf{q}}$ of the perturbation; the dirty approximation replaces it by its average $1/N_k$, "
    r"so $\mathbf{k}$ and $\mathbf{k}'$ are summed independently and $q$ disappears (it survives only in the validity condition $q\ell\ll1$). "
    r"(3) There is one gap function $\bar\Delta(\varepsilon)$; $\Delta$ and $\Delta'$ are its values at the band energies of the two states. In energy space each state has two branches, "
    r"$\Delta_\sigma(E)=\Delta^*-\sigma b\ell/\kappa$, and $\xi\xi'+\Delta\Delta'=E_{\min}^2+\sigma\sigma'\ell\ell'$ for $\eta=-1$.") + "</div>")

# ---- 0 notation ---------------------------------------------------------------------------------------------------
H.append("<h2>0. Notation</h2>")
H.append("<p class='lead'>" + T(
    r"$\bar\Delta(\varepsilon)=\Delta^0-b\,\varepsilon$ with $b=\Delta^s/4t$ (HHM Eq. 37), $\xi=\varepsilon-\mu$, so that $\bar\Delta=\Delta_F-b\xi$ with $\Delta_F=\Delta^0-b\mu$. "
    r"Put $\kappa=\sqrt{1+b^2}$, $E_{\min}=|\Delta_F|/\kappa$, $\varepsilon^*=\mu+b\Delta_F/\kappa^2$, $\Delta^*=\Delta_F/\kappa^2$, $\xi^*=\varepsilon^*-\mu=b\Delta^*$. "
    r"$N(\varepsilon)=N_k^{-1}\sum_{\mathbf{k}s}\delta(\varepsilon-\varepsilon_{\mathbf{k}s})$ is the normal density of states of both helicity bands ($\int N\,d\varepsilon=2$, zero for $|\varepsilon|>W$), "
    r"so $\sum_{\mathbf{k}s}F(\varepsilon_{\mathbf{k}s})=N_k\int d\varepsilon\,N(\varepsilon)F(\varepsilon)$. The quasiparticle energy depends on $(\mathbf{k},s)$ only through $\varepsilon$ "
    r"(HHM Eq. 26 with $|\Delta_{\mathbf{k}s}|=|\bar\Delta_{\mathbf{k}s}|$):") + "</p>")
H.append(D(r"E(\varepsilon)^2=\xi^2+\bar\Delta(\varepsilon)^2=(1+b^2)\xi^2-2b\Delta_F\xi+\Delta_F^2=E_{\min}^2+\kappa^2(\varepsilon-\varepsilon^*)^2", label="1"))

# ---- 1 -----------------------------------------------------------------------------------------------------------------
H.append("<h2>1. Integral over &epsilon; or over <i>E</i>?</h2>")
H.append("<div class='step'>" + T(r"Step 1 &mdash; the map $\varepsilon\mapsto E$ is two-to-one.") + "</div>")
H.append("<p class='lead'>" + T(r"By (1), $E\ge E_{\min}$ with the minimum at $\varepsilon^*$. For every $E>E_{\min}$ there are exactly two band energies, one on each side of $\varepsilon^*$:") + "</p>")
H.append(D(r"\varepsilon_\sigma(E)=\varepsilon^*+\sigma\,\frac{\ell}{\kappa},\qquad \ell\equiv\sqrt{E^2-E_{\min}^2},\qquad \sigma=\pm1,\qquad \left|\frac{dE}{d\varepsilon}\right|_{\varepsilon_\sigma}=\frac{\kappa^2|\varepsilon-\varepsilon^*|}{E}=\frac{\kappa\,\ell}{E}", label="2"))
H.append("<div class='step'>" + T(r"Step 2 &mdash; insert $1=\int_{E_{\min}}^{\infty}dE\,\delta(E-E(\varepsilon))$.") + "</div>")
H.append("<p class='lead'>" + T(r"With $\delta(E-E(\varepsilon))=\sum_\sigma\delta(\varepsilon-\varepsilon_\sigma(E))/|dE/d\varepsilon|_{\varepsilon_\sigma}$:") + "</p>")
H.append(D(r"\sum_{\mathbf{k}s}F(\varepsilon_{\mathbf{k}s})=N_k\int d\varepsilon\,N(\varepsilon)F(\varepsilon)=N_k\int_{E_{\min}}^{E_{\max}}dE\,\sum_{\sigma=\pm}\nu_\sigma(E)\,F\left(\varepsilon_\sigma(E)\right),\qquad \nu_\sigma(E)=N\left(\varepsilon_\sigma(E)\right)\frac{E}{\kappa\sqrt{E^2-E_{\min}^2}}", label="3"))
H.append("<p>" + T(
    r"This is an identity (a change of variables), not an approximation. The factor $E/\kappa\ell$ is the Jacobian $|d\varepsilon/dE|$; it diverges like $(E-E_{\min})^{-1/2}$ at the lower limit, which is integrable "
    r"(it is the BCS density-of-states peak) and absent in the $\varepsilon$-form, where $dE=(\kappa\ell/E)\,d\varepsilon\to0$ cancels it. "
    r"<b>Limits:</b> the lower limit $E_{\min}$ is the minimum of (1). The upper limit is $\infty$ only for an unbounded band; the lattice band has $N(\varepsilon)=0$ for $|\varepsilon|>W$, so $\nu_\sigma(E)=0$ "
    r"whenever $|\varepsilon_\sigma(E)|>W$ (the band test), i.e. effectively $E\le E_{\max}=\max\{E(-W),E(W)\}$ ($E_{\min}=@Emin@\,t$, $E_{\max}=@Emax@\,t$ in the example of the code review).") + "</p>")
H.append("<div class='step'>" + T(r"Step 3 &mdash; the absorbed power.") + "</div>")
H.append("<p class='lead'>" + T(
    r"The golden rule with a constant matrix element $B$ (Sec. 2) gives, for quasiparticle scattering ($a$ = upper state, $E_a=E_b+\hbar\omega$) and for pair creation ($E_a+E_b=\hbar\omega$, $b$ the lower member, $E_b<E_a$),") + "</p>")
H.append(D(r"W_{sc}=2\pi\omega B^2\sum_{a,b}C_{ab}\left[f(E_b)-f(E_a)\right]\delta(E_a-E_b-\hbar\omega),\qquad W_{pair}=2\pi\omega B^2\sum_{a,b}^{E_b<E_a}C^{p}_{ab}\left[1-f(E_a)-f(E_b)\right]\delta(E_a+E_b-\hbar\omega)", label="4"))
H.append("<p class='lead'>" + T(r"with $f(E)=1/(e^{E/T}+1)$ ($E$ measured from the condensate: no $\mu$). Applying (3) to both sums and using the delta function to remove $E_a$:") + "</p>")
H.append(D(r"W_{sc}=2\pi\omega B^2N_k^2\int_{E_{\min}}^{E_{\max}-\hbar\omega}dE\sum_{\sigma\sigma'}\nu_{\sigma'}(E+\hbar\omega)\,\nu_{\sigma}(E)\;C_{\sigma'\sigma}(E+\hbar\omega,E)\left[f(E)-f(E+\hbar\omega)\right]", label="5"))
H.append(D(r"W_{pair}=2\pi\omega B^2N_k^2\int_{E_{\min}}^{\hbar\omega/2}dE\sum_{\sigma\sigma'}\nu_{\sigma}(E)\,\nu_{\sigma'}(\hbar\omega-E)\;C^{p}_{\sigma\sigma'}(E,\hbar\omega-E)\left[1-f(E)-f(\hbar\omega-E)\right]", label="6"))
H.append("<p>" + T(
    r"In (5) the energy conservation $\delta(E_a-E_b-\hbar\omega)$ and the occupations act on the quasiparticle energies; the sums over $\sigma,\sigma'$ run over the admissible branches. "
    r"<b>The $\varepsilon$-form used in the code is (5)&ndash;(6) written with $\int d\varepsilon_b N(\varepsilon_b)$ instead of $\int dE\sum_\sigma\nu_\sigma(E)$: by (3) the two are equal.</b> "
    r"Numerically: (3) holds to @idn1@ &ndash; @idn3@ for three test functions, and the literal $E$-integrals (5)&ndash;(6) agree with the $\varepsilon$-code to @relsc@ (scattering) and @relpair@ (pair creation).") + "</p>")
H.append("<div class='box warn'>" + T(
    r"<b>What is not equivalent.</b> The formula of the old code is $\int d\varepsilon\,N(\varepsilon)N(\varepsilon+\hbar\omega)\,\tilde C\,[f_N(\varepsilon)-f_N(\varepsilon+\hbar\omega)]$: the delta function is $\delta(\varepsilon_a-\varepsilon_b-\hbar\omega)$ and the occupation is the normal-state $f_N$. "
    r"Since $E(\varepsilon)$ in (1) is nonlinear and two-to-one, no change of variables turns $\delta(\varepsilon_a-\varepsilon_b-\hbar\omega)$ into $\delta(E_a-E_b-\hbar\omega)$: it is a different integral, not a different parametrisation of (5). "
    r"<b>Check:</b> for $b=0$ and a flat $N(\varepsilon)=N_0$ one has $\nu_\sigma(E)=N_0E/\ell$ and (5) gives the textbook $W_s/W_n\to2f(\Delta)$ as $\omega\to0$ (verified numerically to $10^{-4}$).") + "</div>")

# ---- 2 -----------------------------------------------------------------------------------------------------------------
H.append("<h2>2. Where is q? What is the dirty approximation?</h2>")
H.append("<div class='step'>" + T(r"Step 1 &mdash; where $q$ comes from.") + "</div>")
H.append("<p class='lead'>" + T(r"A sound wave of wavevector $\mathbf{q}$ and frequency $\omega$ couples to the electrons through $H'(t)=\hat H'e^{-i\omega t}+\mathrm{h.c.}$ with") + "</p>")
H.append(D(r"\hat H'=\sum_{\mathbf{k}\mathbf{k}'}\sum_{ss'}B(\mathbf{k}s|\mathbf{k}'s')\,a^{\dagger}_{\mathbf{k}s}a_{\mathbf{k}'s'},\qquad B(\mathbf{k}s|\mathbf{k}'s')=B_{\mathbf{q}}\,\langle\mathbf{k}s|\mathbf{k}'s'\rangle\,\delta_{\mathbf{k},\mathbf{k}'+\mathbf{q}}", label="7"))
H.append("<p class='lead'>" + T(
    r"($\langle\mathbf{k}s|\mathbf{k}'s'\rangle$ is the helicity-spinor overlap, $\omega=v_s|\mathbf{q}|$.) The quasiparticle matrix element is $M_{ab}=B(a|b)\,[u_au_b-\eta\,v_av_b]$, so $|M_{ab}|^2=|B(a|b)|^2\,C_{ab}$ with $C_{ab}=\frac{1}{2}[1+\xi\xi'/EE'-\eta\bar\Delta\bar\Delta'/EE']$. "
    r"<b>Clean limit</b> (momentum conserved): the Kronecker delta removes the sum over $\mathbf{k}'$, and (4) becomes") + "</p>")
H.append(D(r"W_{sc}(\mathbf{q})=2\pi\omega|B_{\mathbf{q}}|^2\sum_{\mathbf{k}ss'}\left|\langle\mathbf{k}s|\mathbf{k}{-}\mathbf{q},s'\rangle\right|^2C(\mathbf{k}s;\mathbf{k}{-}\mathbf{q},s')\left[f(E_{\mathbf{k}-\mathbf{q},s'})-f(E_{\mathbf{k}s})\right]\delta\left(E_{\mathbf{k}s}-E_{\mathbf{k}-\mathbf{q},s'}-\hbar\omega\right)", label="8"))
H.append("<p>" + T(
    r"Here $q$ appears in three places: in the arguments $\mathbf{k}-\mathbf{q}$ of $\varepsilon,E,\bar\Delta,u,v$ (hence in $C$), in the helicity overlap, and in the energy condition; for small $q$, $E_{\mathbf{k}s}-E_{\mathbf{k}-\mathbf{q},s}\approx\hbar\,\mathbf{v}_g(\mathbf{k})\cdot\mathbf{q}$, "
    r"and with $\omega=v_sq$ the resonance condition is $\mathbf{v}_g\cdot\hat{\mathbf{q}}=v_s$. This is a $\mathbf{k}$-resolved calculation, not an energy-space one.") + "</p>")
H.append("<div class='step'>" + T(r"Step 2 &mdash; the dirty approximation.") + "</div>")
H.append("<p class='lead'>" + T(
    r"<b>Statement.</b> The paper averages $B$ over the angles of $\mathbf{k},\mathbf{k}'$ and the spin indices and treats it as a constant (following its reference to Tinkham, Ch. 3); it does not use the word &ldquo;dirty&rdquo;, which is my name for the physical situation in which such an average is justified. "
    r"Heuristically: with elastic impurity scattering of mean free path $\ell_{mfp}$ the states are no longer plane waves, and for $q\,\ell_{mfp}\ll1$ (and $\ell_{mfp}\ll\xi_0$, the usual dirty limit) the phonon cannot select $\mathbf{k}'=\mathbf{k}-\mathbf{q}$. "
    r"Mathematically the approximation replaces the Kronecker delta in (7) by its average over $\mathbf{q}$, $\;N_k^{-1}\sum_{\mathbf{q}}\delta_{\mathbf{k},\mathbf{k}'+\mathbf{q}}=1/N_k$, i.e. the matrix element becomes independent of $\mathbf{k},\mathbf{k}'$:") + "</p>")
H.append(D(r"\sum_{\mathbf{k}\mathbf{k}'}|B_{\mathbf{q}}|^2\delta_{\mathbf{k},\mathbf{k}'+\mathbf{q}}\,F(\mathbf{k},\mathbf{k}')\;\longrightarrow\;\frac{|B_{\mathbf{q}}|^2}{N_k}\sum_{\mathbf{k}\mathbf{k}'}F(\mathbf{k},\mathbf{k}')=\frac{|B_{\mathbf{q}}|^2}{N_k}\sum_{\mathbf{q}'}\sum_{\mathbf{k}}F(\mathbf{k},\mathbf{k}-\mathbf{q}')", label="9"))
H.append("<p class='lead'>" + T(
    r"The right-hand side is the clean result (8) summed over <i>all</i> momentum transfers $\mathbf{q}'$ with equal weight. In this model $F$ depends on $(\mathbf{k},s)$ and $(\mathbf{k}',s')$ only through $\varepsilon,\varepsilon'$ (the coherence factor $C$ and the occupations) "
    r"and through the helicity overlap. The overlap averages to $\frac{1}{2}$ over each constant-energy contour, because $\hat{\mathbf{g}}_{-\mathbf{k}}=-\hat{\mathbf{g}}_{\mathbf{k}}$ while $\varepsilon_{\mathbf{k}s}$ is even in $\mathbf{k}$:") + "</p>")
H.append(D(r"\left\langle\tfrac{1}{2}\left(1+ss'\,\hat{\mathbf{g}}_{\mathbf{k}}\cdot\hat{\mathbf{g}}_{\mathbf{k}'}\right)\right\rangle_{\varepsilon,\varepsilon'}=\tfrac{1}{2}\left(1+ss'\,\langle\hat{\mathbf{g}}_{\mathbf{k}}\rangle_\varepsilon\cdot\langle\hat{\mathbf{g}}_{\mathbf{k}'}\rangle_{\varepsilon'}\right)=\tfrac{1}{2}", label="10"))
H.append("<p class='lead'>" + T(r"Therefore the sums factorise into energy integrals, with $B^2=|B_{\mathbf{q}}|^2/2N_k$:") + "</p>")
H.append(D(r"W=2\pi\omega B^2N_k^2\int d\varepsilon\,d\varepsilon'\,N(\varepsilon)N(\varepsilon')\,C(\varepsilon,\varepsilon')\left[f(E')-f(E)\right]\delta(E-E'-\hbar\omega)", label="11"))
H.append("<p>" + T(
    r"which is (5) after Step 2 of Sec. 1, and which has the same $B^2$ in $W_N$, so $B^2$ cancels in $W/W_N$. <b>Where is $q$ now?</b> In no formula: it has been averaged over. The physical value $q=\omega/v_s$ appears only in the validity condition $q\ell_{mfp}\ll1$; "
    r"a number such as $qa\approx43$ for $\hbar\omega=0.7t$ is a statement about that condition (and about the lattice zone), not an input. In the all-pairs sum the absorbed weight is spread over all momentum transfers: "
    r"10 %, 50 % and 90 % of it lies at $|q|a<@q10@$, $@q50@$, $@q90@$ (zone corner 4.44).") + "</p>")
H.append("<div class='box warn'>" + T(
    r"<b>What this is, and is not.</b> (9) is an <i>assumption</i> of the paper's formalism; I did not derive its range of validity for the Rashba model. If you want a fixed $\mathbf{q}$ (clean limit, $q\ell_{mfp}\gg1$) you need (8), which is $\mathbf{k}$-resolved. "
    r"One supporting fact: the pairing kernel of HHM is $V_{ss'}\propto U-2(\Delta t/t)(\varepsilon_{\mathbf{k}s}+\varepsilon_{\mathbf{k}'s'})$ (Eq. 20 with $8\Delta t\,s_{\mathbf{k}}+4(\Delta t/t)V_{SO}sR_{\mathbf{k}}=-2(\Delta t/t)\varepsilon_{\mathbf{k}s}$), "
    r"so all inputs depend on a state only through its band energy: energy is the natural label once momentum is not conserved.") + "</div>")

# ---- 3 -----------------------------------------------------------------------------------------------------------------
H.append("<h2>3. Two gap values, &Delta; and &Delta;&prime;</h2>")
H.append("<p class='lead'>" + T(
    r"There is <b>one</b> gap function, $\bar\Delta(\varepsilon)=\Delta_F-b\xi$ (HHM Eq. 37). In the coherence factor $C=\frac{1}{2}[1+\xi\xi'/EE'-\eta\Delta\Delta'/EE']$ the unprimed quantities belong to state $a$ and the primed ones to state $b$: "
    r"$\Delta=\bar\Delta(\varepsilon_a)$, $\Delta'=\bar\Delta(\varepsilon_b)$ &mdash; the same function at two arguments. (If you mean the complex $\Delta_{\mathbf{k}s}=s\,e^{-i\theta_{\mathbf{k}}}\bar\Delta_{\mathbf{k}s}$: $|\Delta_{\mathbf{k}s}|=|\bar\Delta_{\mathbf{k}s}|$, and the phase cancels against the time-reversal phase of the helicity vertex, "
    r"see the helicity note.)") + "</p>")
H.append("<div class='step'>" + T(r"Step 1 &mdash; in energy space each state has two branches.") + "</div>")
H.append("<p class='lead'>" + T(r"Evaluate $\xi=\varepsilon-\mu$ and $\bar\Delta=\Delta_F-b\xi$ at the roots (2): with $\xi_\sigma=\xi^*+\sigma\ell/\kappa$ and $\xi^*=b\Delta^*$,") + "</p>")
H.append(D(r"\xi_\sigma(E)=\xi^*+\sigma\frac{\ell}{\kappa},\qquad \Delta_\sigma(E)=\bar\Delta\left(\varepsilon_\sigma(E)\right)=\Delta_F-b\xi_\sigma=\Delta^*-\sigma\frac{b\,\ell}{\kappa},\qquad \xi_\sigma^2+\Delta_\sigma^2=E_{\min}^2+\ell^2=E^2", label="12"))
H.append("<p>" + T(
    r"(The cross term $2\sigma(\ell/\kappa)(\xi^*-b\Delta^*)$ in $\xi_\sigma^2+\Delta_\sigma^2$ vanishes, and $\xi^{*2}+\Delta^{*2}=\kappa^2\Delta^{*2}=E_{\min}^2$.) For energies $E_a=E_b+\hbar\omega$ there are four combinations $(\sigma_a,\sigma_b)$, "
    r"each with its own pair $(\xi,\Delta),(\xi',\Delta')$ and weight $\nu_{\sigma_a}(E_a)\nu_{\sigma_b}(E_b)$; a branch with $|\varepsilon_\sigma|>W$ is dropped.") + "</p>")
H.append("<div class='step'>" + T(r"Step 2 &mdash; the product of the two gaps in closed form.") + "</div>")
H.append("<p class='lead'>" + T(r"Multiplying the branch values of (12) for the two states ($\ell_a,\ell_b$ their $\ell$'s):") + "</p>")
H.append(D(r"\xi_a\xi_b+\Delta_a\Delta_b=E_{\min}^2+\sigma_a\sigma_b\,\ell_a\ell_b", label="13"))
H.append(D(r"\xi_a\xi_b-\Delta_a\Delta_b=-(1-b^2)\Delta^{*2}+\frac{2b\Delta^*}{\kappa}\left(\sigma_a\ell_a+\sigma_b\ell_b\right)+\sigma_a\sigma_b\,\frac{(1-b^2)}{\kappa^2}\,\ell_a\ell_b", label="14"))
H.append("<p class='lead'>" + T(r"Hence the coherence factors of (5) and (6) are explicit functions of $E_a,E_b,\sigma_a,\sigma_b$ only:") + "</p>")
H.append(D(r"C^{(\eta=-1)}_{\sigma_a\sigma_b}=\frac{1}{2}\left[1+\frac{E_{\min}^2+\sigma_a\sigma_b\ell_a\ell_b}{E_aE_b}\right],\qquad C^{(\eta=+1)}_{\sigma_a\sigma_b}=\frac{1}{2}\left[1+\frac{\xi_a\xi_b-\Delta_a\Delta_b}{E_aE_b}\right]\ \mathrm{with\ (14)},\qquad C^{p}=\frac{1}{2}\left[1-\frac{\xi_a\xi_b-\eta\Delta_a\Delta_b}{E_aE_b}\right]", label="15"))
H.append("<p>" + T(
    r"<b>BCS limit</b> $b\to0$: $\xi^*=0$, $\Delta^*=\Delta_F=\Delta$, $\kappa=1$, $E_{\min}=|\Delta|$, $\xi_\sigma=\sigma\ell$, and (13), (14) reduce to $\Delta^2+\sigma_a\sigma_b\ell_a\ell_b$ and $-\Delta^2+\sigma_a\sigma_b\ell_a\ell_b$: the standard case-II and case-I coherence factors. "
    r"For $b\ne0$ the extra terms are exactly the particle&ndash;hole asymmetry of the gap. <b>Check:</b> (12)&ndash;(14) hold to @errA@ over 20000 random $(E_a,E_b,\sigma_a,\sigma_b)$.") + "</p>")

# ---- 4 -----------------------------------------------------------------------------------------------------------------
H.append("<h2>4. Result</h2>")
H.append("<p class='lead'>" + T(r"Collecting (5), (6), (15) and the normal-state absorption with its own chemical potential $\mu_N(T)$ at the same filling:") + "</p>")
H.append(D(r"\frac{W}{W_N}=\frac{W_{sc}+W_{pair}}{W_N},\qquad W_N=2\pi\omega B^2N_k^2\int d\varepsilon\,N(\varepsilon)N(\varepsilon+\hbar\omega)\left[f_N(\varepsilon)-f_N(\varepsilon+\hbar\omega)\right]", label="16", cls="res"))
H.append("<p>" + T(
    r"Inputs: $\Delta^0(T),\Delta^s(T),\mu(T)$ from the self-consistent solution of HHM Eqs. 32, 33, 38 (no $\Delta_{\rm eff}$); the lattice enters only through $N(\varepsilon)$, the band half-width $W$ and (via $\varepsilon$) the helicity bands. "
    r"Assumptions: constant $B$ and unrestricted $\mathbf{k},\mathbf{k}'$ sums (Sec. 2). Not covered: fixed $q$ (clean limit), a direction-dependent $B_{\mathbf{q}}$, and the $q\to0$ electromagnetic interband vertex &mdash; these need the $\mathbf{k}$-resolved form (8).") + "</p>")


# ---- 5 -----------------------------------------------------------------------------------------------------------------
from coherence_variants_energy_space import S_parts
from validate_qp import FlatDOS

fr5 = [0.3, 0.5, 0.7, 0.9, 0.96]
Tc5, Ts5, D05, Ds5, mu5, muN5 = sc_state_vs_T(m, fr5)
OM = 0.7
rows5 = []
for i in range(1, len(Ts5)):
    S_N = np.sum(dos.w_i * dos.N(dos.eps_i + OM) * (fermi_normal(dos.eps_i, Ts5[i], muN5[i]) - fermi_normal(dos.eps_i + OM, Ts5[i], muN5[i])))
    vals = []
    for xm, sg, pair in (("signed", -1, True), ("abs", -1, True), ("signed", +1, True), ("abs", +1, False)):
        a, b_ = S_parts(dos, Ts5[i], OM, D05[i], Ds5[i], mu5[i], +1, xm, sg, pair)
        vals.append((a + b_) / S_N)
    rows5.append((fr5[i - 1], 2 * qp_constants(D05[i], Ds5[i], mu5[i])["Emin"], vals))
i7 = 3
S_N0 = np.sum(dos.w_i * dos.N(dos.eps_i + OM) * (fermi_normal(dos.eps_i, Ts5[i7], muN5[i7]) - fermi_normal(dos.eps_i + OM, Ts5[i7], muN5[i7])))
sr = {}
for xm in ("signed", "abs"):
    a, b_ = S_parts(dos, Ts5[i7], OM, 0.0, 0.0, muN5[i7], +1, xm, -1)
    sr[xm] = (a / S_N0, b_ / S_N0)
flat = FlatDOS(W=12.0, n=40000)
bcs = [(x, absorption(flat, 0.02, 1.0 * x, 0.5, 0.0, 0.0, 0.0, eta=+1)["ratio"], absorption(flat, 0.02, 1.0 * x, 0.5, 0.0, 0.0, 0.0, eta=-1)["ratio"]) for x in (0.9, 1.005, 1.1, 1.3, 2.0, 3.0, 6.0)]
fr2 = np.concatenate([np.linspace(0.05, 0.9, 35), np.linspace(0.905, 0.9995, 120)])
Tc2, Ts2, D02, Ds2, mu2, muN2 = sc_state_vs_T(m, list(fr2))
Em2 = np.array([qp_constants(D02[i], Ds2[i], mu2[i])["Emin"] for i in range(len(Ts2))])
mx = {}
for om in (0.7, 0.2, 0.05):
    R = np.array([absorption(dos, Ts2[i], om, D02[i], Ds2[i], mu2[i], muN2[i], eta=+1)["ratio"] for i in range(1, len(Ts2))])
    idx = np.where(2 * Em2[1:] < om)[0]
    mx[om] = (R.max(), Ts2[1:][int(R.argmax())] / Tc2, (R[:idx[0]].max() if len(idx) else R.max()), (Ts2[1:][idx[0]] / Tc2 if len(idx) else None))
oldlog = open("coherence_variants_old_output.txt").read()
oldrows = []
for line in oldlog.splitlines():
    mm = re.match(r"^\s*(0\.\d+)\s+(\d\.\d+) \|(.*)$", line)
    if mm:
        pairs = re.findall(r"(\d+\.\d+) \[ *(\d+\.\d+)\]", mm.group(3))
        if len(pairs) == 4:
            oldrows.append((float(mm.group(1)), [float(p[0]) for p in pairs]))
brk = re.findall(r"\[\+([0-9.]+), ([0-9.]+)\]", oldlog)
NUM.update(s0=f"{sr['signed'][0]:.4f}", s1=f"{sr['signed'][1]:.4f}", s2=f"{sum(sr['signed']):.4f}", a0=f"{sr['abs'][0]:.4f}", a1=f"{sr['abs'][1]:.4f}", a2=f"{sum(sr['abs']):.4f}")
NUM.update(bl=brk[0][0], bh=brk[0][1])

H.append("<h2>5. The sign of &xi; in the coherence factor, and what to expect for W/W<sub>N</sub></h2>")
H.append("<p class='lead'>" + T(
    r"<b>What the paper does.</b> Its expansion of $[uu'-\eta vv']^2$ (written with positive square roots, $v\ge0$) gives $\frac12\left(1+\xi\xi'/EE'-\Delta\Delta'/EE'\right)$ with the <i>signed</i> $\xi\xi'$ &mdash; the correct case-I scattering factor, and the paper remarks that this term cannot be dropped. "
    r"Two later steps spoil it: (i) the final $W$ carries $+\eta\Delta\Delta'$, whereas $(uu'-\eta vv')^2$ gives $-\eta\Delta\Delta'$; (ii) &ldquo;in terms of $E$&rdquo; the paper replaces $\varepsilon-\mu$ by $\sqrt{E^2-\Delta^2}$. "
    r"Squaring $E^2=\xi^2+\Delta^2$ erases the sign of $\xi$, and taking the positive root puts every state on the electron side: $\xi\xi'\to|\xi||\xi'|$.") + "</p>")
H.append("<div class='step'>" + T(r"Why the sign of $\xi$ cannot be dropped.") + "</div>")
H.append("<p class='lead'>" + T(r"Write the Bogoliubov coefficients as $u=\cos\varphi$, $v=\sin\varphi$, with $2\varphi$ the polar angle of the vector $(\xi,\bar\Delta)$ (both components signed):") + "</p>")
H.append(D(r"\cos2\varphi=\frac{\xi}{E},\qquad \sin2\varphi=\frac{\bar\Delta}{E},\qquad uu'-\eta vv'=\cos(\varphi+\eta\varphi'),\qquad uv'+\eta u'v=\sin(\varphi'+\eta\varphi)", label="17"))
H.append("<p class='lead'>" + T(r"Hence, for scattering and for pair creation,") + "</p>")
H.append(D(r"C=\cos^2(\varphi+\eta\varphi')=\frac12\left[1+\frac{\xi\xi'-\eta\bar\Delta\bar\Delta'}{EE'}\right],\qquad C^{p}=\sin^2(\varphi'+\eta\varphi)=\frac12\left[1-\frac{\xi\xi'-\eta\bar\Delta\bar\Delta'}{EE'}\right],\qquad C+C^{p}=1", label="18", cls="res"))
H.append("<p>" + T(
    r"For $\eta=+1$ (ultrasound) the combination is $\xi\xi'-\bar\Delta\bar\Delta'$ with $\xi,\bar\Delta$ signed. The sign of $\xi$ is the electron/hole label: for $\bar\Delta\to0$, $\varphi=0$ ($\xi>0$) or $\pi/2$ ($\xi<0$), so (18) gives "
    r"$C=1$ for two states of the same type and $C=0$ for opposite types (a scalar vertex cannot turn an electron into a hole), and $C^p$ is the complement &mdash; opposite-type pairs are created, not scattered. "
    r"With $|\xi||\xi'|$ one gets $C=1$ for <i>every</i> pair at $\bar\Delta=0$ and no pair creation. <b>Sum rule at $\bar\Delta\to0$</b> (scattering + pair creation must reproduce the normal-state absorption, $T/T_c=0.7$): "
    r"signed $\xi\xi'$: @s0@ + @s1@ = @s2@; with $|\xi||\xi'|$: @a0@ + @a1@ = @a2@.") + "</p>")
H.append("<div class='step'>" + T(r"Does the choice change the behaviour of $W/W_N$?") + "</div>")
H.append("<p class='lead'>" + T(
    r"<b>In the paper's (and the repo's) formula</b> the bracket is $1+\mathbf n\cdot\mathbf n'$ with unit vectors $\mathbf n=(|\xi|,\Delta)/E$, $\mathbf n'=(|\xi'|,\eta\Delta')/E'$, hence lies in $[0,2]$; $W_N$ has bracket $2$. "
    r"Therefore $W/W_N\le1$ <i>identically</i>, whatever the data: for the repo (bracket in [@bl@, @bh@]) the ratio is the average of $C=\mathrm{bracket}/2\in[0.75,1]$ over the thermal window, 0.93&ndash;0.96. "
    r"This bound is a property of the formula, not evidence for it. It disappears as soon as the sign of $\xi$ is kept in the old integrand (band-energy conservation, normal-state $f$): normalised by its own $\Delta\to0$ limit, which vanishes for pairs straddling $\mu$, "
    r"the ratio becomes (T/T<sub>c</sub> = 0.1, 0.3, 0.5, 0.7, 0.8): "
    + ", ".join(f"{r[1][1]:.2f}" for r in oldrows) + r" with $+\eta\Delta^2$, and "
    + ", ".join(f"{r[1][3]:.2f}" for r in oldrows) + r" with $-\eta\Delta^2$ (repo inputs, $\hbar\omega=0.7$). That framework cannot be repaired by choosing signs: the pairs straddling $\mu$ need quasiparticle energy conservation and the pair-creation factor $C^p$.") + "</p>")
hdr = ("<tr><th>T/T<sub>c</sub></th><th>2E<sub>min</sub></th><th>(1) signed &xi;&xi;&prime;, &minus;&eta;&Delta;&Delta;&prime;<br><b>correct</b></th><th>(2) |&xi;||&xi;&prime;|,<br>&minus;&eta;&Delta;&Delta;&prime;</th>"
       "<th>(3) signed &xi;&xi;&prime;,<br>+&eta;&Delta;&Delta;&prime;</th><th>(4) |&xi;||&xi;&prime;|, +&eta;&Delta;&Delta;&prime;<br>paper's final, no pair term</th></tr>")
body = "".join(f"<tr><td>{f}</td><td>{e:.3f}</td>" + "".join(f"<td>{v:.4f}</td>" for v in vals) + "</tr>" for f, e, vals in rows5)
H.append("<p class='lead'>" + T(r"<b>In the quasiparticle framework</b> (energy conservation on $E$, $f(E)$, scattering + pair creation, physical $W_N$; $V_{SO}=0.4082$, $\hbar\omega=0.7$, $\eta=+1$; (1)&ndash;(3) include $C^p$ with the same sign choice, (4) has no pair term as in the paper):") + "</p>")
H.append("<table class='compact'>" + hdr + body + "</table>")
H.append("<p>" + T(
    r"Below the pair-breaking onset ($2E_{\min}>\hbar\omega$, $T/T_c<0.91$) all four stay below 1 but differ by tens of per cent ((4) is 50 % above (1) at $T/T_c=0.7$ and 0.9); above the onset the sign of the $\eta$ term and the missing pair channel change the result by a factor 2.") + "</p>")
H.append("<div class='step'>" + T(r"Should $W/W_N<1$ be expected?") + "</div>")
H.append("<p class='lead'>" + T(
    r"In the correct formalism there is no bound like the one above. Textbook BCS limit (flat $N$, constant gap, $\mu=0$, $T\to0$; bare golden rule, no vertex corrections):") + "</p>")
bt = ("<table class='compact'><tr><th>&hbar;&omega;/2&Delta;</th>" + "".join(f"<td>{x}</td>" for x, _, _ in bcs) + "</tr>"
      "<tr><th>W<sub>s</sub>/W<sub>n</sub>, &eta; = +1 (ultrasound)</th>" + "".join(f"<td>{a:.3f}</td>" for _, a, _ in bcs) + "</tr>"
      "<tr><th>W<sub>s</sub>/W<sub>n</sub>, &eta; = &minus;1 (EM)</th>" + "".join(f"<td>{c:.3f}</td>" for _, _, c in bcs) + "</tr></table>")
H.append(bt)
H.append("<p>" + T(
    r"For $\hbar\omega<2\Delta$ only scattering contributes and $W_s/W_n<1$ ($\to2f(\Delta)$ as $\omega\to0$, verified to $10^{-4}$). Above $2\Delta$ the pair-creation factor $C^p=\sin^2(\varphi+\varphi')$ of case I jumps to $\approx\pi/2$ at the threshold (the two BCS edges, $C^p\to1$) and decays to 1 from above; "
    r"for $\eta=-1$ it rises continuously from 0. <b>In our model</b> ($V_{SO}=0.4082$, correct formula, $\eta=+1$): "
    + "; ".join(f"$\\hbar\\omega={om}$: maximum {v[0]:.3f} at $T/T_c={v[1]:.4f}$, below the onset {('(T/T<sub>c</sub> &lt; %.4f)' % v[3]) if v[3] else '(none reached)'} maximum {v[2]:.3f}" for om, v in mx.items())
    + r". So $W/W_N<1$ is expected for $\hbar\omega<2E_{\min}(T)$, in particular for ultrasound with $\hbar\omega\ll\Delta$ (apart from a window of relative width $\sim(\hbar\omega/2\Delta_0)^2$ below $T_c$); "
    r"with $\hbar\omega=0.7t$, comparable to the gap, the bare formalism gives a bump above 1 for $\eta=+1$ near $T_c$. At $\hbar\omega\gtrsim2\Delta$ the density-type vertex is also where vertex corrections (gauge invariance) are expected to matter, which are not included.") + "</p>")

doclib.build(H, "Energy-space absorption: the mathematics", "Energy_space_mathematics", "Energy-space absorption: the mathematics")
print("built", sum(h.count("<math") for h in H), "math pieces")
