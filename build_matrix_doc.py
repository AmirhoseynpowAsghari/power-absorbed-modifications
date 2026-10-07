"""Build  Helicity_matrix_element.html / .pdf   (note on the matrix element of arXiv:2607.02812 in helicity space)."""
import html as _html
import numpy as np
import doclib
from doclib import T, M, D, img, table, NUM
import helicity_matrix_element as hme
from delta_of_E import band_halfwidth


def fm(x, fmt=".4f"):
    return format(x, fmt).replace("-", "\u2212")


def sci(x):
    if x == 0:
        return "0 (exactly)"
    mant, ex = f"{x:.0e}".split("e")
    return f"{int(mant)}&times;10<sup>&minus;{abs(int(ex))}</sup>"


# ------------------------------------------------------------------------------------------------
# numbers (everything below is computed here, nothing is typed in by hand)
# ------------------------------------------------------------------------------------------------
P = hme.P
D0, Ds, mu, VSO = P["D0"], P["Ds"], P["mu"], P["VSO"]
res = hme.run_checks(verbose=False)
rows, r1, r2 = hme.E_space_check(verbose=False)
wts = hme.overlap_weights([0.05, 0.1, 0.3, 1.0, 2.0, np.pi])
E4 = hme.helicity_band_threshold()
hme.make_figure(res)

ALT = [dict(VSO=0.9, D0=-0.35, Ds=2.10, mu=-1.30), dict(VSO=0.2, D0=0.80, Ds=-0.90, mu=2.20)]
_base = dict(hme.P)
alt_res = []
for a_ in ALT:
    hme.P.update(a_)
    ra = hme.run_checks(nsamp=600, seed=11, verbose=False)
    alt_res.append((a_, max(max(v) for v in ra["worst"].values()),
                    max(ra["dT"], ra["dG"], ra["dE"], ra["dV"], ra["dF"], ra["dM_intra"])))
hme.P.update(_base)
W = band_halfwidth(VSO)
b = Ds / 4
eps0 = 4 * D0 / Ds
by = {(r["sig"], r["sigp"]): r for r in rows}
rp, rm_ = by[("+", "-")], by[("-", "-")]
e1p, e1m = float(r1["eps_plus"]), float(r1["eps_minus"])
e2p, e2m = float(r2["eps_plus"]), float(r2["eps_minus"])
wc = res["worst"]
names = list(wc)
nM_ = res["nM"]


def rng(t_):
    return f"[{fm(t_[0])}, {fm(t_[1])}]"


NUM.update(
    VSO=fm(VSO, ".1f"), D0=fm(D0, ".5f"), Ds=fm(Ds, ".5f"), mu=fm(mu, ".5f"), b=fm(b, ".4f"), W=fm(W, ".4f"), eps0=fm(eps0, ".3f"),
    E1="1.10", E2="2.00", e1p=fm(e1p, ".3f"), e1m=fm(e1m, ".3f"), e2p=fm(e2p, ".3f"), e2m=fm(e2m, ".3f"), E4=fm(E4, ".3f"),
    n_hel1_p=str(rp["hel1"][-1]), n_hel1_p_plus=str(rp["hel1"][1]),
    dT=sci(res["dT"]), dG=sci(res["dG"]), dE=sci(res["dE"]), dV=sci(res["dV"]), dF=sci(res["dF"]),
    nM=str(nM_), scale=fm(res["scale"], ".2f"), dMi=sci(res["dM_intra"]),
    nNi=("exactly zero" if res["nN_intra"] == 0 else "below " + sci(res["nN_intra"])),
    alt1=sci(alt_res[0][1]), alt2=sci(alt_res[1][1]), altg=sci(max(alt_res[0][2], alt_res[1][2])),
    a1="(" + ", ".join(fm(ALT[0][k_], ".2f") for k_ in ("VSO", "D0", "Ds", "mu")) + ")",
    a2="(" + ", ".join(fm(ALT[1][k_], ".2f") for k_ in ("VSO", "D0", "Ds", "mu")) + ")",
)

# ------------------------------------------------------------------------------------------------
# document
# ------------------------------------------------------------------------------------------------
H = []
H.append("<h1>The transition matrix element of arXiv:2607.02812 in helicity space</h1>")
H.append("<div class='sub'>Rashba&ndash;Hubbard superconductor: quasiparticle scattering and pair creation, helicity vertices for ultrasound and "
         "electromagnetic absorption, and an exact numerical check</div>")
H.append("<div class='ref'><b>The paper:</b> R. Afzali, B. Takrim, A. H. Asghari, <i>Theoretical investigation of power absorbed by superconductors with Rashba "
         "spin-orbit coupling in the presence external time-dependent perturbation</i>, arXiv:2607.02812 (2026). Its model, helicity basis and Bogoliubov "
         "transformation are those of J. Hutchinson, J. E. Hirsch, F. Marsiglio, Phys. Rev. B <b>97</b>, 184513 (2018), arXiv:1801.08988 (&ldquo;HHM&rdquo;, "
         "&ldquo;HHM Eq. n&rdquo;). The paper's formulas carry no equation numbers in the version I read, so they are quoted by name; equations of this note are labelled (1), (2), &hellip;</div>")

# ---- result ---------------------------------------------------------------------------------------
H.append("<div class='box'><b>Result.</b> " + T(
    r"In the helicity basis the paper's matrix element keeps its form ($\alpha\to s$; unprimed $=(\mathbf{k},s)$, primed $=(\mathbf{k}',s')$),") + "</div>")
H.append(D(r"M(\mathbf{k}s|\mathbf{k}'s')=B(\mathbf{k}s|\mathbf{k}'s')\left[u_{\mathbf{k}s}u_{\mathbf{k}'s'}-\eta\,v_{\mathbf{k}s}v_{\mathbf{k}'s'}\right]", label="M", cls="res"))
H.append("<div class='box'>" + T(
    r"<b>with three points that the constant-$B$ treatment misses.</b> "
    r"(i) $B(\mathbf{k}s|\mathbf{k}'s')=\langle\mathbf{k}s|\hat{O}|\mathbf{k}'s'\rangle$ is the <i>helicity-space vertex</i> of the perturbation $\hat{O}$; it is not a constant (Sec. 3). "
    r"(ii) $u_{\mathbf{k}s}=\sqrt{\frac{1}{2}(1+\xi_{\mathbf{k}s}/E_{\mathbf{k}s})}$ and $v_{\mathbf{k}s}=\mathrm{sgn}(\bar\Delta_{\mathbf{k}s})\sqrt{\frac{1}{2}(1-\xi_{\mathbf{k}s}/E_{\mathbf{k}s})}$ are real, "
    r"with $\xi_{\mathbf{k}s}=\varepsilon_{\mathbf{k}s}-\mu$, $E_{\mathbf{k}s}=\sqrt{\xi_{\mathbf{k}s}^2+\bar\Delta_{\mathbf{k}s}^2}$ and the <i>signed</i> gap $\bar\Delta_{\mathbf{k}s}=\Delta^0-\Delta^s\varepsilon_{\mathbf{k}s}/4t$ (HHM Eq. 37). "
    r"(iii) The time-reversal phases are those of the helicity states, $\rho_{\mathbf{k}s}=s\,e^{-i\theta_{\mathbf{k}}}$; they cancel exactly against the phase of the gap $\Delta_{\mathbf{k}s}=\rho_{\mathbf{k}s}\bar\Delta_{\mathbf{k}s}$. "
    r"$\eta=+1$ for a time-reversal-even operator (ultrasound, the paper's case I), $\eta=-1$ for a time-reversal-odd one (current, case II).") + "</div>")
H.append(D(r"|M|^2=|B(\mathbf{k}s|\mathbf{k}'s')|^2\;\frac{1}{2}\left[1+\frac{\xi\,\xi'}{E\,E'}-\eta\,\frac{\bar\Delta\,\bar\Delta'}{E\,E'}\right]", label="C", cls="res"))
H.append("<div class='box'>" + T(
    r"<b>Pair creation</b> (the other two terms of the paper's $H'$), with $X=\langle\mathbf{k}s|\hat{O}|T(\mathbf{k}'s')\rangle$, $T$ = time reversal:") + "</div>")
H.append(D(r"N(\mathbf{k}s,\mathbf{k}'s')=X\left[u_{\mathbf{k}s}v_{\mathbf{k}'s'}+\eta\,u_{\mathbf{k}'s'}v_{\mathbf{k}s}\right],\qquad |N|^2=|X|^2\,\frac{1}{2}\left[1-\frac{\xi\,\xi'}{E\,E'}+\eta\,\frac{\bar\Delta\,\bar\Delta'}{E\,E'}\right]", label="N", cls="res"))
H.append("<div class='box'>" + T(
    r"<b>Vertices</b> (Sec. 3). Ultrasound, spin-independent coupling of strength $B_q$, $\eta=+1$: "
    r"$|B|^2=|B_q|^2\,\frac{1}{2}\left(1+ss'\cos\Delta\theta\right)$, $\Delta\theta=\theta_{\mathbf{k}}-\theta_{\mathbf{k}'}$. "
    r"Electromagnetic absorption at $q\to0$, $\eta=-1$: $B=\mathbf{A}\cdot\mathbf{V}_{ss'}(\mathbf{k})$ with $\mathbf{V}_{ss}=\nabla_{\mathbf{k}}\varepsilon_{\mathbf{k}s}$ and an interband part "
    r"$V^x_{s,-s}=2iV_{SO}\,s\cos k_x\sin k_y/R_{\mathbf{k}}$, $V^y_{s,-s}=-2iV_{SO}\,s\cos k_y\sin k_x/R_{\mathbf{k}}$, $R_{\mathbf{k}}=\sqrt{\sin^2k_x+\sin^2k_y}$.") + "</div>")

# ---- 1 setting -----------------------------------------------------------------------------------------
H.append("<h2>1. Setting and conventions</h2>")
H.append("<p class='lead'>" + T(
    r"The non-interacting part is $H_0(\mathbf{k})=\varepsilon_{\mathbf{k}}-\mu+H_{SO}(\mathbf{k})$ with $\varepsilon_{\mathbf{k}}=-2t(\cos k_x+\cos k_y)$ and the Rashba term of the paper (HHM Eq. 14),") + "</p>")
H.append(D(r"H_{SO}(\mathbf{k})=-2V_{SO}\left(\sin k_y\,\sigma_x-\sin k_x\,\sigma_y\right)=-2V_{SO}R_{\mathbf{k}}\;\hat{\mathbf{g}}_{\mathbf{k}}\cdot\boldsymbol{\sigma},\qquad \hat{\mathbf{g}}_{\mathbf{k}}=(\cos\theta_{\mathbf{k}},\sin\theta_{\mathbf{k}})", label="1"))
H.append("<p class='lead'>" + T(
    r"Its eigenstates are the helicity states (HHM Eqs. 15&ndash;17), $e^{i\theta_{\mathbf{k}}}=(\sin k_y-i\sin k_x)/R_{\mathbf{k}}$:") + "</p>")
H.append(D(r"|\mathbf{k}s\rangle=\frac{1}{\sqrt{2}}\left(1,\;s\,e^{i\theta_{\mathbf{k}}}\right)^{T},\qquad \varepsilon_{\mathbf{k}s}=\varepsilon_{\mathbf{k}}-2sV_{SO}R_{\mathbf{k}},\qquad s=\pm1", label="2"))
H.append("<p class='lead'>" + T(
    r"<b>Time reversal</b> $T=i\sigma_yK$ keeps the helicity and maps $\mathbf{k}\to-\mathbf{k}$; since $e^{i\theta_{-\mathbf{k}}}=-e^{i\theta_{\mathbf{k}}}$,") + "</p>")
H.append(D(r"T|\mathbf{k}s\rangle=\rho_{\mathbf{k}s}|-\mathbf{k},s\rangle,\qquad \rho_{\mathbf{k}s}=s\,e^{-i\theta_{\mathbf{k}}},\qquad \rho_{-\mathbf{k}s}=-\rho_{\mathbf{k}s},\qquad T^2=-1", label="3"))
H.append("<p class='lead'>" + T(
    r"(HHM write the time-reversed partner of $c_{\mathbf{k}s}$ as $-s\,e^{i\theta}c_{-\mathbf{k}s}$: the same up to the overall sign convention of $T$.) "
    r"<b>The gap.</b> Pairing is intraband, between $|\mathbf{k}s\rangle$ and its time-reverse, and HHM (text after Eq. 22) find") + "</p>")
H.append(D(r"\Delta_{\mathbf{k}s}=s\,e^{-i\theta_{\mathbf{k}}}\bar\Delta_{\mathbf{k}s}=\rho_{\mathbf{k}s}\bar\Delta_{\mathbf{k}s},\qquad \bar\Delta_{\mathbf{k}s}=\Delta^0+\Delta^s s_{\mathbf{k}}+\Delta^{x-y}sR_{\mathbf{k}}=\Delta^0-\frac{\Delta^s}{4t}\varepsilon_{\mathbf{k}s}\quad\mathrm{(real)}", label="4"))
H.append("<p>" + T(
    r"In spin space this is a singlet $\Delta^0+\Delta^s s_{\mathbf{k}}$ plus a triplet with $\mathbf{d}_{\mathbf{k}}=\Delta^{x-y}(\sin k_y,-\sin k_x,0)$ along the Rashba vector, $\Delta^{x-y}=V_{SO}\Delta^s/2t$ (HHM Eq. 35). "
    r"I checked numerically that this spin-space gap matrix projects onto <i>exactly</i> the form (4), diagonal in $s$ (Sec. 5). "
    r"<b>Bogoliubov transformation</b> (HHM Eq. 23), for the helicity operators $a_{\mathbf{k}s}$ ($a^{\dagger}_{\mathbf{k}s}$ creates $|\mathbf{k}s\rangle$):") + "</p>")
H.append(D(r"a_{\mathbf{k}s}=u_{\mathbf{k}s}\,\gamma_{\mathbf{k}s}-\rho_{\mathbf{k}s}\,v_{\mathbf{k}s}\,\gamma^{\dagger}_{-\mathbf{k}s},\qquad u_{\mathbf{k}s}v_{\mathbf{k}s}=\frac{\bar\Delta_{\mathbf{k}s}}{2E_{\mathbf{k}s}},\qquad u^2_{\mathbf{k}s}-v^2_{\mathbf{k}s}=\frac{\xi_{\mathbf{k}s}}{E_{\mathbf{k}s}}", label="5"))
H.append("<p class='small'>" + T(
    r"$u,v$ are real and even in $\mathbf{k}$; the oddness of the gap sits in $\rho_{\mathbf{k}s}$. In HHM's convention the pairing term is $-\frac{1}{2}\Delta c^{\dagger}c^{\dagger}$ and $v$ has the opposite overall sign; only products $vv'$ enter below.") + "</p>")

# ---- 2 derivation ----------------------------------------------------------------------------------------
H.append("<h2>2. Derivation</h2>")
H.append("<p class='lead'>" + T(
    r"The perturbation is a one-body operator, the paper's $H'=\sum B(\mathbf{k}\alpha|\mathbf{k}'\alpha')c^{\dagger}_{\mathbf{k}\alpha}c_{\mathbf{k}'\alpha'}$; in the helicity basis") + "</p>")
H.append(D(r"H'=\sum_{\mathbf{k}\mathbf{k}'}\sum_{ss'}B(\mathbf{k}s|\mathbf{k}'s')\,a^{\dagger}_{\mathbf{k}s}a_{\mathbf{k}'s'},\qquad B(\mathbf{k}s|\mathbf{k}'s')=\langle\mathbf{k}s|\hat{O}|\mathbf{k}'s'\rangle", label="6"))
H.append("<div class='step'>" + T(r"Step 1 &mdash; insert (5).") + "</div>")
H.append(D(r"a^{\dagger}_{\mathbf{k}s}a_{\mathbf{k}'s'}=u\,u'\,\gamma^{\dagger}_{\mathbf{k}s}\gamma_{\mathbf{k}'s'}+\rho^{*}_{\mathbf{k}s}\rho_{\mathbf{k}'s'}\,v\,v'\,\gamma_{-\mathbf{k}s}\gamma^{\dagger}_{-\mathbf{k}'s'}-u\,\rho_{\mathbf{k}'s'}v'\,\gamma^{\dagger}_{\mathbf{k}s}\gamma^{\dagger}_{-\mathbf{k}'s'}-\rho^{*}_{\mathbf{k}s}v\,u'\,\gamma_{-\mathbf{k}s}\gamma_{\mathbf{k}'s'}", label="7"))
H.append("<div class='step'>" + T(r"Step 2 &mdash; the scattering amplitude.") + "</div>")
H.append("<p class='lead'>" + T(
    r"The second term of (7) is $-\rho^{*}_{\mathbf{k}s}\rho_{\mathbf{k}'s'}vv'\,\gamma^{\dagger}_{-\mathbf{k}'s'}\gamma_{-\mathbf{k}s}$ plus a constant. Relabelling $(\mathbf{k},s,\mathbf{k}',s')\to(-\mathbf{k}',s',-\mathbf{k},s)$, using $\rho_{-\mathbf{k}s}=-\rho_{\mathbf{k}s}$ and the evenness of $v$, "
    r"the coefficient of $\gamma^{\dagger}_{\mathbf{k}s}\gamma_{\mathbf{k}'s'}$ is") + "</p>")
H.append(D(r"M(\mathbf{k}s|\mathbf{k}'s')=B(\mathbf{k}s|\mathbf{k}'s')\,u_{\mathbf{k}s}u_{\mathbf{k}'s'}-\rho_{\mathbf{k}s}\rho^{*}_{\mathbf{k}'s'}\,v_{\mathbf{k}s}v_{\mathbf{k}'s'}\,B(-\mathbf{k}'s'|-\mathbf{k}s)", label="8"))
H.append("<p>" + T(
    r"This is the paper's general expression $M=B\,u_ku_{k'}-v_kv_{k'}\sum\rho\rho\,B(-k'\alpha'|-k\alpha)$ with the spin phases replaced by the helicity phases.") + "</p>")
H.append("<div class='step'>" + T(r"Step 3 &mdash; time-reversal lemma.") + "</div>")
H.append("<p class='lead'>" + T(
    r"Let $\hat{O}$ be Hermitian with $T\hat{O}T^{-1}=\eta\hat{O}$. For an antiunitary $T$, $\langle T\phi|\hat{O}|T\psi\rangle=\eta\langle\psi|\hat{O}|\phi\rangle$. "
    r"With $|-\mathbf{k}s\rangle=\rho^{*}_{\mathbf{k}s}\,T|\mathbf{k}s\rangle$ from (3) this gives") + "</p>")
H.append(D(r"B(-\mathbf{k}'s'|-\mathbf{k}s)=\eta\,\rho_{\mathbf{k}'s'}\rho^{*}_{\mathbf{k}s}\,B(\mathbf{k}s|\mathbf{k}'s')", label="9"))
H.append("<p>" + T(
    r"This is the paper's relation $\sum\rho\rho\,B(-k'\alpha'|-k\alpha)=\eta B(k\alpha|k'\alpha')$, again with helicity phases: $\eta=+1$ for a scalar (density-like) operator, $\eta=-1$ for a current-like one.") + "</p>")
H.append("<div class='step'>" + T(r"Step 4 &mdash; the phases cancel.") + "</div>")
H.append("<p>" + T(
    r"Insert (9) into (8): the phase factor is $\rho_{\mathbf{k}s}\rho^{*}_{\mathbf{k}'s'}\cdot\rho_{\mathbf{k}'s'}\rho^{*}_{\mathbf{k}s}=1$, so the second term of (8) is $-\eta\,B\,v_{\mathbf{k}s}v_{\mathbf{k}'s'}$ and (M) follows. "
    r"This works <i>because</i> the gap carries exactly the phase $\rho_{\mathbf{k}s}$ (4), which makes $v$ real. Had the phase of the gap been dropped, or $|\Delta_{\mathbf{k}s}|$ used, a factor like $e^{i(\theta_{\mathbf{k}}-\theta_{\mathbf{k}'})}$ would survive in the coherence term.") + "</p>")
H.append("<div class='step'>" + T(r"Step 5 &mdash; square it.") + "</div>")
H.append("<p class='lead'>" + T(r"With $u^2=\frac{1}{2}(1+\xi/E)$, $v^2=\frac{1}{2}(1-\xi/E)$ and $uv=\bar\Delta/2E$ from (5):") + "</p>")
H.append(D(r"(uu'-\eta vv')^2=u^2u'^2+v^2v'^2-2\eta\,uu'vv'=\frac{1}{2}\left[1+\frac{\xi\xi'}{EE'}-\eta\,\frac{\bar\Delta\bar\Delta'}{EE'}\right]", label="10"))
H.append("<p>" + T(r"which is (C). Because $\bar\Delta$ changes sign at the node $\varepsilon_0=4t\Delta^0/\Delta^s$, the product $\bar\Delta\bar\Delta'$ must keep its sign.") + "</p>")
H.append("<div class='step'>" + T(r"Step 6 &mdash; pair creation.") + "</div>")
H.append("<p>" + T(
    r"The two cross terms of (7) create or annihilate pairs of quasiparticles. Antisymmetrising the creation term and using $\langle\psi|\hat{O}|T\phi\rangle=-\eta\langle\phi|\hat{O}|T\psi\rangle$ "
    r"(from the lemma and $T^2=-1$) gives the amplitude $N_{ab}=X_{ab}(u_av_b+\eta\,u_bv_a)$ for $\gamma^{\dagger}_a\gamma^{\dagger}_b$, $X_{ab}=\langle a|\hat{O}|Tb\rangle$, and $|N|^2$ as in (N).") + "</p>")

# ---- 3 vertices ----------------------------------------------------------------------------------------------
H.append("<h2>3. The helicity vertex B(ks|k's')</h2>")
H.append("<h3>3.1 Ultrasound: spin-independent coupling, &eta; = +1</h3>")
H.append("<p class='lead'>" + T(
    r"A deformation-potential coupling $\hat{O}=B_q\rho_{\mathbf{q}}+\mathrm{h.c.}$ is the identity in spin space, so $B(\mathbf{k}s|\mathbf{k}'s')=B_q\langle\mathbf{k}s|\mathbf{k}'s'\rangle$ with $\mathbf{k}=\mathbf{k}'+\mathbf{q}$, and from (2)") + "</p>")
H.append(D(r"\langle\mathbf{k}s|\mathbf{k}'s'\rangle=\frac{1}{2}\left(1+ss'\,e^{i(\theta_{\mathbf{k}'}-\theta_{\mathbf{k}})}\right),\qquad |\langle\mathbf{k}s|\mathbf{k}'s'\rangle|^2=\frac{1}{2}\left(1+ss'\cos\Delta\theta\right),\quad \cos\Delta\theta=\hat{\mathbf{g}}_{\mathbf{k}}\cdot\hat{\mathbf{g}}_{\mathbf{k}'}", label="11"))
H.append("<p class='lead'>" + T(r"Together with (C), the complete ultrasonic matrix element for scattering is") + "</p>")
H.append(D(r"|M(\mathbf{k}s|\mathbf{k}'s')|^2=\frac{|B_q|^2}{4}\left(1+ss'\cos\Delta\theta\right)\left[1+\frac{\xi\xi'}{EE'}-\frac{\bar\Delta\bar\Delta'}{EE'}\right]", label="12", cls="res"))
H.append("<p>" + T(
    r"For pair creation ($\mathbf{k}+\mathbf{k}'=\mathbf{q}$) the overlap is with $|-\mathbf{k}'s'\rangle$, and since $\theta_{-\mathbf{k}}=\theta_{\mathbf{k}}+\pi$ the roles of intra- and inter-helicity are exchanged: "
    r"$|X|^2=|B_q|^2\frac{1}{2}(1-ss'\cos\Delta\theta)$.") + "</p>")
H.append("<p>" + T(
    r"<b>Selection rule.</b> For $q\ll\pi/a$ the Rashba vectors at $\mathbf{k}$ and $\mathbf{k}'=\mathbf{k}-\mathbf{q}$ are nearly parallel: the weight is $\approx1$ for $s'=s$ and $\approx0$ for $s'=-s$. "
    r"The Brillouin-zone average of the two weights (independent of $V_{SO}$, since only $\theta_{\mathbf{k}}$ enters), for $\mathbf{q}$ along $k_x$, is") + "</p>")
qrow = "<tr><th class='l'>" + T(r"$q$ (units of $1/a$)") + "</th>" + "".join(f"<td>{fm(w[0], '.2f')}</td>" for w in wts) + "</tr>"
irow = "<tr><th class='l'>" + T(r"intra-helicity, $s'=s$") + "</th>" + "".join(f"<td>{fm(w[1], '.4f')}</td>" for w in wts) + "</tr>"
orow = "<tr><th class='l'>" + T(r"inter-helicity, $s'=-s$") + "</th>" + "".join(f"<td>{fm(w[2], '.4f')}</td>" for w in wts) + "</tr>"
H.append("<table class='compact'>" + qrow + irow + orow + "</table>")
H.append("<p>" + T(
    r"Only for $q\sim\pi/a$ does the weight reach the value $\frac{1}{2}$ in all four $(s,s')$ channels, which is what a constant $B$ assumes (Sec. 4).") + "</p>")

H.append("<h3>3.2 Electromagnetic absorption: current vertex, &eta; = &minus;1</h3>")
H.append("<p class='lead'>" + T(
    r"In the dipole limit ($q\to0$) the coupling is $\mathbf{A}\cdot\hat{\mathbf{V}}$ with the velocity operator $\mathbf{V}=\partial_{\mathbf{k}}H_0$ (all constants absorbed in $\mathbf{A}$). It is odd under $T$ and diagonal in $\mathbf{k}$:") + "</p>")
H.append(D(r"V_x=2t\sin k_x+2V_{SO}\cos k_x\,\sigma_y,\qquad V_y=2t\sin k_y-2V_{SO}\cos k_y\,\sigma_x", label="13"))
H.append("<p class='lead'>" + T(
    r"With (2), $\langle\mathbf{k}s|\sigma_x|\mathbf{k}s'\rangle=s\cos\theta_{\mathbf{k}}$ and $\langle\mathbf{k}s|\sigma_y|\mathbf{k}s'\rangle=s\sin\theta_{\mathbf{k}}$ for $s'=s$, "
    r"and $-i\,s\sin\theta_{\mathbf{k}}$, $i\,s\cos\theta_{\mathbf{k}}$ for $s'=-s$; with $\cos\theta_{\mathbf{k}}=\sin k_y/R_{\mathbf{k}}$, $\sin\theta_{\mathbf{k}}=-\sin k_x/R_{\mathbf{k}}$ this gives") + "</p>")
H.append(D(r"V^x_{ss}=2t\sin k_x-2sV_{SO}\frac{\sin k_x\cos k_x}{R_{\mathbf{k}}}=\partial_{k_x}\varepsilon_{\mathbf{k}s},\qquad V^x_{s,-s}=2iV_{SO}\,s\,\frac{\cos k_x\sin k_y}{R_{\mathbf{k}}}", label="14"))
H.append(D(r"V^y_{ss}=2t\sin k_y-2sV_{SO}\frac{\sin k_y\cos k_y}{R_{\mathbf{k}}}=\partial_{k_y}\varepsilon_{\mathbf{k}s},\qquad V^y_{s,-s}=-2iV_{SO}\,s\,\frac{\cos k_y\sin k_x}{R_{\mathbf{k}}}", label="15"))
H.append("<p class='lead'>" + T(r"The interband element equals $(\varepsilon_{\mathbf{k},-s}-\varepsilon_{\mathbf{k}s})\langle\mathbf{k}s|\partial_{\mathbf{k}}|\mathbf{k},-s\rangle$ with $\varepsilon_{\mathbf{k},-s}-\varepsilon_{\mathbf{k}s}=4sV_{SO}R_{\mathbf{k}}$. From (M) and (N) with $\eta=-1$, $\mathbf{k}'=\mathbf{k}$:") + "</p>")
H.append(D(r"M(\mathbf{k}s|\mathbf{k}s')=\mathbf{A}\cdot\mathbf{V}_{ss'}\left(u_{\mathbf{k}s}u_{\mathbf{k}s'}+v_{\mathbf{k}s}v_{\mathbf{k}s'}\right),\qquad N(\mathbf{k}s,-\mathbf{k}s')=\mathbf{A}\cdot\mathbf{V}_{ss'}\left(u_{\mathbf{k}s}v_{\mathbf{k}s'}-u_{\mathbf{k}s'}v_{\mathbf{k}s}\right)", label="16"))
H.append("<ul>"
         "<li>" + T(r"<b>Intraband</b> ($s'=s$): $M=\mathbf{A}\cdot\nabla\varepsilon_{\mathbf{k}s}$ (since $u^2+v^2=1$), but there is no energy transfer, so no absorption at $\omega\ne0$; and $N(\mathbf{k}s,-\mathbf{k}s)\equiv0$ &mdash; no pair breaking within one band at $q=0$ (BdG check: @nNi@).") + "</li>"
         "<li>" + T(r"<b>Interband</b> ($s'=-s$): scattering when $E_{\mathbf{k},-s}-E_{\mathbf{k}s}=\hbar\omega$, and pair creation above $E_{\mathbf{k}+}+E_{\mathbf{k}-}$. Both are generically non-zero and are driven by $V_{SO}$ through the interband vertex.") + "</li>"
         "</ul>")
H.append("<p>" + T(
    r"So at $q=0$ the clean Rashba superconductor absorbs only through interband channels, which a constant-$B$ treatment cannot represent. "
    r"The paper's double integral over independent $\mathbf{k},\mathbf{k}'$ corresponds instead to momentum-non-conserving scattering, where the weights of Sec. 3.1 apply.") + "</p>")

# ---- 4 relation to paper ------------------------------------------------------------------------------------
H.append("<h2>4. How this differs from the formulas in the paper</h2>")
H.append("<ol>"
         "<li>" + T(r"<b>Constant $B$.</b> The paper averages $B$ over angles and spin indices and takes it constant. In helicity space this replaces $|\langle\mathbf{k}s|\mathbf{k}'s'\rangle|^2$ by its uncorrelated-angle average $\frac{1}{2}$ in all four channels. "
                    r"That is reasonable only if $\mathbf{k}$ and $\mathbf{k}'$ are uncorrelated; for small $q$ it misplaces all the weight of the $s'=-s$ channels (table above, Fig. 1b).") + "</li>"
         "<li>" + T(r"<b>Sign of the $\eta$ term.</b> With $M=B(uu'-\eta vv')$, (10) gives $-\eta\bar\Delta\bar\Delta'/EE'$. The paper's intermediate result is the $\eta=+1$ case, $\frac{1}{2}(1+\xi\xi'/EE'-\Delta\Delta'/EE')$, "
                    r"but its final expressions for $W$ and $W/W_N$, as I read them in the extracted text, carry $+\eta\Delta\Delta'/EE'$; with $\eta=+1$ for ultrasound that exchanges the coherence factors of the two cases. Please check this against the typeset PDF.") + "</li>"
         "<li>" + T(r"<b>Signed gap.</b> $\bar\Delta_{\mathbf{k}s}$ changes sign at $\varepsilon_0=4t\Delta^0/\Delta^s=@eps0@\,t$ in the example below. In $\bar\Delta\bar\Delta'$ the sign must be kept; $|\Delta_{\mathbf{k}s}|$ is wrong for transitions across the node.") + "</li>"
         "<li>" + T(r"<b>Sign of $\xi$ in energy variables.</b> The paper writes $\varepsilon_{\mathbf{k}s}-\mu=+\sqrt{E^2-\Delta^2}$ in the coherence factor. At a given $E$ there are two roots $\varepsilon_\pm(E)$ (previous note) and $\xi$ can have either sign, so") + "</li>"
         "</ol>")
H.append(D(r"C_\eta(E,\sigma;E',\sigma')=\frac{1}{2}\left[1+\frac{\xi_\sigma(E)\,\xi_{\sigma'}(E')}{E\,E'}-\eta\,\frac{\Delta_\sigma(E)\,\Delta_{\sigma'}(E')}{E\,E'}\right],\qquad \xi_\sigma(E)=\varepsilon_\sigma(E)-\mu", label="17"))
H.append("<p class='lead'>" + T(
    r"with $\varepsilon_\sigma(E)$, $\Delta_\sigma(E)$ and the band test of the $\Delta(E)$ note. <b>Example</b> ($V_{SO}=@VSO@\,t$, $\Delta^0=@D0@\,t$, $\Delta^s=@Ds@\,t$, $\mu=@mu@\,t$, $W=@W@\,t$), pairs of states with $E\approx@E1@\,t$ and $E'\approx@E2@\,t$. "
    r"At $E$ the roots are $\varepsilon_+=@e1p@\,t$ ($\xi>0$) and $\varepsilon_-=@e1m@\,t$ ($\xi<0$); at $E'$ they are $\varepsilon_-=@e2m@\,t$ and $\varepsilon_+=@e2p@\,t>W$, which the band test discards. "
    r"I collected all pairs of grid states (300&times;300&times;2 grid, windows $\pm0.004\,t$) in these energy windows and computed their coherence factors directly:") + "</p>")


def cell(val, ob):
    return f"{fm(val)}<br><span class='small'>[{fm(ob[0])}, {fm(ob[1])}]</span>"


tab = ("<table class='compact'><tr><th>" + T(r"roots $(\sigma,\sigma')$") + "</th><th>" + T(r"sgn$(\xi\xi')$") + "</th><th>" + T(r"helicity of the<br>states on $\sigma$") + "</th>"
       "<th>" + T(r"$C_{\eta=+1}$ from (17)<br>[range on the grid]") + "</th><th>" + T(r"$C_{\eta=-1}$ from (17)<br>[range on the grid]") + "</th>"
       "<th>&ldquo;+&radic;&rdquo; rule<br>(&eta;=+1, &eta;=&minus;1)</th></tr>"
       f"<tr><td>(+, &minus;)</td><td>&minus;1</td><td>{T(r'$s=-1$ only<br>(@n_hel1_p@ states)')}</td><td>{cell(rp['C_I'], rp['ob_C_I'])}</td><td>{cell(rp['C_II'], rp['ob_C_II'])}</td>"
       f"<td><span class='no'>{fm(rp['paper_C_I'])}, {fm(rp['paper_C_II'])}</span></td></tr>"
       f"<tr><td>(&minus;, &minus;)</td><td>+1</td><td>{T(r'both: $s=+1$: {}, $s=-1$: {}'.format(rm_['hel1'][1], rm_['hel1'][-1]))}</td><td>{cell(rm_['C_I'], rm_['ob_C_I'])}</td><td>{cell(rm_['C_II'], rm_['ob_C_II'])}</td>"
       f"<td><span class='ok'>{fm(rm_['paper_C_I'])}, {fm(rm_['paper_C_II'])}</span></td></tr>"
       "<tr><td>(+, +)<br>(&minus;, +)</td><td>&ndash;</td><td colspan='3' class='l'>" + T(r"$\varepsilon_+(E')=@e2p@\,t>W$: discarded by the band test; no grid state with $E\approx E'$ lies on that root (0 found)") + "</td><td>&ndash;</td></tr></table>")
H.append(tab)
H.append("<p>" + T(
    r"The formula (17) reproduces the grid values; the &ldquo;+&radic;&rdquo; prescription is right only when $\xi\xi'>0$ and is off by $\approx0.29$ in the $(+,-)$ channel, where $\xi\xi'<0$. "
    r"Helicity refines the band test: the $s=+1$ band ends at $4t$ and the $s=-1$ band at $W$, so a root with $4t<\varepsilon\le W$ can only belong to $s=-1$. "
    r"Here the $+$ root at $E=@E1@\,t$ has $\varepsilon_+=@e1p@\,t>4t$ and indeed all its states have $s=-1$; the $+$ root hosts $s=+1$ states only for $E<E(4t)=@E4@\,t$.") + "</p>")

# ---- 5 verification ----------------------------------------------------------------------------------------------
H.append("<h2>5. Exact numerical check</h2>")
H.append("<p class='lead'>" + T(
    r"The check does not use any helicity argument. I build the $4\times4$ Bogoliubov&ndash;de Gennes matrix in <i>spin</i> space, in the Nambu basis $(c_{\mathbf{k}\uparrow},c_{\mathbf{k}\downarrow},c^{\dagger}_{-\mathbf{k}\uparrow},c^{\dagger}_{-\mathbf{k}\downarrow})$, "
    r"with the physical gap matrix (singlet + triplet along the Rashba vector):") + "</p>")
H.append(D(r"H_{BdG}(\mathbf{k})=\begin{pmatrix}h(\mathbf{k})&D(\mathbf{k})\\ D^{\dagger}(\mathbf{k})&-h^{T}(-\mathbf{k})\end{pmatrix},\qquad D(\mathbf{k})=\left[\Delta^0+\Delta^s s_{\mathbf{k}}+\Delta^{x-y}R_{\mathbf{k}}\,\hat{\mathbf{g}}_{\mathbf{k}}\cdot\boldsymbol{\sigma}\right]i\sigma_y", label="18"))
H.append("<p class='lead'>" + T(
    r"I diagonalise it numerically and read off the coefficients of $\gamma^{\dagger}\gamma$ and $\gamma^{\dagger}\gamma^{\dagger}$ in $\sum Bc^{\dagger}c$ from the Nambu form $\mathrm{diag}(B_{\mathbf{k},\mathbf{k}'},-B^{T}_{-\mathbf{k}',-\mathbf{k}})$. "
    r"These are compared with the helicity-space formulas for random $(\mathbf{k},\mathbf{k}',s,s')$, with the scalar vertex and with the current vertex (two field directions).") + "</p>")
tbl = "<table class='compact'><tr><th class='l'>check</th><th>max deviation</th></tr>"
for lab, val in (
        (T(r"$T|\mathbf{k}s\rangle=\rho_{\mathbf{k}s}|-\mathbf{k}s\rangle$, eq. (3)"), "@dT@"),
        (T(r"spin-space gap projected on helicity states $=\delta_{ss'}\rho_{-\mathbf{k}s}\bar\Delta_{\mathbf{k}s}$ (no interband gap), eq. (4)"), "@dG@"),
        (T(r"eigenvalues of the spin-space BdG matrix $=\pm E_{\mathbf{k}\pm}$"), "@dE@"),
        (T(r"overlap (11) and current vertices (14)&ndash;(15) against spinor sandwiches"), "@dF@, @dV@"),
        (T(r"intraband $M(\mathbf{k}s|\mathbf{k}s)=\mathbf{A}\cdot\nabla\varepsilon_{\mathbf{k}s}$ against BdG"), "@dMi@")):
    tbl += f"<tr><td class='l'>{lab}</td><td>{T(val)}</td></tr>"
for nm in names:
    tbl += f"<tr><td class='l'>{_html.escape(nm).replace('eta', '&eta;')}: |M|<sub>BdG</sub> &minus; |M|<sub>formula</sub>; |N|<sub>BdG</sub> &minus; |N|<sub>formula</sub></td><td>{sci(wc[nm][0])}; {sci(wc[nm][1])}</td></tr>"
tbl += "</table>"
H.append(tbl)
H.append("<p class='small'>" + T(r"@nM@ random $(\mathbf{k},\mathbf{k}',s,s')$ combinations per quantity; the largest $|M|^2$ is @scale@ in units $|B_q|=|\mathbf{A}|=1$. The BdG parameters are those of my solution of the gap equations ($U=90t$, $\Delta t=4.5t$, $n=1.875$, $k_BT=0.01t$, $V_{SO}=@VSO@\,t$); the identity does not depend on them. Repeating the whole check at two other parameter sets, $(V_{SO},\Delta^0,\Delta^s,\mu)/t=$ @a1@ and @a2@ "
    r"(strong spin-orbit coupling with a negative $\Delta^0$ and $\mu$ below the band centre; weak coupling with a negative $\Delta^s$), gives maximum deviations in $|M|$, $|N|$ of @alt1@ and @alt2@, and of at most @altg@ in all the other checks.") + "</p>")
H.append(img("fig5_helicity_matrix_element.png",
             r"<b>Fig. 1.</b> (a) Helicity overlap factor (11): intra-helicity $\frac{1}{2}(1+\cos\Delta\theta)$, inter-helicity $\frac{1}{2}(1-\cos\Delta\theta)$; a constant $B$ corresponds to the angular average $\frac{1}{2}$. "
             r"(b) Brillouin-zone average of the two weights against the momentum transfer $q$. "
             r"(c) $|M|^2$ and $|N|^2$ from the exact spin-space BdG diagonalisation against the analytic helicity formula (C), (N), for both $\eta$.", width="90%"))

# ---- 6 code --------------------------------------------------------------------------------------------------------
H.append("<h2>6. Minimal code and files</h2>")
CODE = open("helicity_listing.py").read()
ns = {}
exec(CODE, ns)
rng_t = np.random.default_rng(3)
dmax = 0.0
for _ in range(300):
    k, kp = hme.random_k(rng_t), hme.random_k(rng_t)
    s, sp = rng_t.choice([-1, 1]), rng_t.choice([-1, 1])
    Ma = ns["M_helicity"](k, s, kp, sp, ns["vertex_scalar"], +1, D0, Ds, mu, VSO)
    Mb = hme.M_analytic(k, s, kp, sp, P, hme.vertex_scalar, +1)[0]
    A = (np.cos(0.9), np.sin(0.9))
    Mc = ns["M_helicity"](k, s, kp, sp, lambda a, c: ns["vertex_current"](a, c, A, VSO), -1, D0, Ds, mu, VSO)
    Md = hme.M_analytic(k, s, kp, sp, P, hme.make_vertex_current(0.9, P), -1)[0]
    dmax = max(dmax, abs(Ma - Mb), abs(Mc - Md))
assert dmax < 1e-12, dmax
H.append("<pre>" + _html.escape(CODE) + "</pre>")
H.append("<p class='small'>" + T(r"The listing was executed against the module when this note was built and agrees with <code>helicity_matrix_element.py</code> on 300 random points for both vertices (max difference: " + ("exactly 0" if dmax == 0 else sci(dmax)) + ").") + "</p>")
H.append("<p class='small'><b>Scope.</b> " + T(
    r"One-body perturbation; single-quasiparticle and pair-creation amplitudes at the level of the paper (no vertex corrections, screening or impurity matrix elements). "
    r"$M$ is defined up to the phase convention of $|\mathbf{k}s\rangle$; only $|M|^2$ is physical. Energy conservation and the $\mathbf{k},\mathbf{k}'$ sums that turn $|M|^2$ into the absorbed power $W$ are not part of this note.") + "</p>")
H.append("<p class='small'><b>Files</b> (<code>/home/user/deltaE/</code>): <code>Helicity_matrix_element.pdf</code>/<code>.html</code>, <code>helicity_matrix_element.py</code>, "
         "<code>helicity_listing.py</code>, <code>fig5_helicity_matrix_element.png</code>, <code>build_matrix_doc.py</code> + <code>doclib.py</code> (rebuild).</p>")

doclib.build(H, "Matrix element in helicity space", "Helicity_matrix_element", "Matrix element in helicity space (arXiv:2607.02812)")
print("built; math pieces:", sum(h.count("<math") for h in H))
