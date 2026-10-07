"""Build  Delta_of_E_derivation.html / .pdf  (MathML typeset by headless Chromium)."""
import base64, re, html as _html
import numpy as np
import latex2mathml.converter as L
from solve_gap import Model
from delta_of_E import (delta_of_E, branch_windows, band_halfwidth, brute_force_states,
                        stress_test_windows, helicity_band_edges)

# ------------------------------------------------------------------------------------------
# numbers used in the text
# ------------------------------------------------------------------------------------------
def fm(x, fmt=".4f"):
    return format(x, fmt).replace("-", "\u2212")


U, dt, n, VSO, kT = 90.0, 4.5, 1.875, 0.5, 0.01
m = Model(U=U, dt=dt, n=n, VSO=VSO, Nk=1600)
Tc, x, ok, res = m.solve_low_T(kT)
D0, Ds, mu = x
r = delta_of_E(np.array([2.0]), D0, Ds, mu, VSO=VSO)
b, DF, Emin, es, Dst = r["b"], r["Delta_F"], r["Emin"], r["eps_star"], r["Delta_star"]
W = band_halfwidth(VSO)
bw = branch_windows(D0, Ds, mu, VSO)
EpW, EmW = bw["E_plusW"], bw["E_minusW"]
hb = helicity_band_edges(VSO)
eps0 = 4 * D0 / Ds
E0 = abs(eps0 - mu)
beta = (es - mu) / Emin
eps_bf, Dbar_bf, E_bf, hel_bf = brute_force_states(D0, Ds, mu, VSO)
plus_bf = eps_bf > es
bf_p = (E_bf[plus_bf].min(), E_bf[plus_bf].max())
bf_m = (E_bf[~plus_bf].min(), E_bf[~plus_bf].max())
dev = 0.0
rr_all = delta_of_E(E_bf, D0, Ds, mu, VSO=VSO)
Dform = np.where(plus_bf, rr_all["Delta_plus"], rr_all["Delta_minus"])
dev = float(np.max(np.abs(Dbar_bf - Dform)))
st = stress_test_windows()

def sci(x):
    mant, ex = f"{x:.0e}".split("e")
    return f"{int(mant)}&times;10<sup>&minus;{abs(int(ex))}</sup>"


NUM = dict(Emin5=fm(Emin, ".5f"), sp0=fm(hb[1][0]), sp1=fm(hb[1][1]), sm0=fm(hb[-1][0]), sm1=fm(hb[-1][1]),
           ratio_pm=fm((1 + beta) / (1 - beta), ".3f"), ratio_inv=fm((1 - beta) / (1 + beta), ".2f"),
           W=fm(W), EpW=fm(EpW, ".3f"), EmW=fm(EmW, ".3f"), Emin=fm(Emin), eps_star=fm(es), Dst=fm(Dst), DF=fm(DF),
           b=fm(b), D0=fm(D0), Ds=fm(Ds), mu=fm(mu), Tc=fm(Tc, ".3f"), eps0=fm(eps0, ".3f"), E0=fm(E0, ".2f"),
           bfp0=fm(bf_p[0]), bfp1=fm(bf_p[1]), bfm0=fm(bf_m[0]), bfm1=fm(bf_m[1]),
           nsets=str(st["nsets"]), nout=str(st["n_eps_star_outside"]), gap=sci(float(st["max_upper_gap"])),
           beta=fm(beta, ".3f"), dev=sci(dev))

paper_Tc = {0.05: 0.33, 0.1: 0.33, 0.2: 0.35, 0.3: 0.38, 0.4: 0.42, 0.5: 0.47, 0.6: 0.53}
paper_Em = {0.05: 0.57, 0.1: 0.58, 0.2: 0.62, 0.3: 0.68, 0.4: 0.75, 0.5: 0.85, 0.6: 1.03}
rows = []
for v in paper_Tc:
    mm = Model(U=U, dt=dt, n=n, VSO=v, Nk=1200)
    Tc_v, x_v, ok_v, _ = mm.solve_low_T(kT)
    b_v = x_v[1] / 4
    DF_v = x_v[0] - b_v * x_v[2]
    E_v = abs(DF_v) / np.sqrt(1 + b_v ** 2)
    rows.append((v, Tc_v, paper_Tc[v], E_v, paper_Em[v], 2 * E_v / Tc_v, b_v, np.sign(DF_v) * b_v / np.sqrt(1 + b_v ** 2)))


# ------------------------------------------------------------------------------------------
# helpers: LaTeX -> MathML
# ------------------------------------------------------------------------------------------
UNARY = re.compile(r'(<mo[^>]*>(?!&#x00029;|&#x0005D;|&#x0007D;|&#x0007C;)[^<]*</mo>)<mo>(&#x02212;|&#x0002B;)</mo>')


def post(s):
    # latex2mathml emits numeric character references, e.g. &#x00394; for the Greek capital Delta
    for ch in ("\u0394", "&#x00394;"):
        s = s.replace(f"<mi>{ch}</mi>", '<mi mathvariant="normal">\u0394</mi>')
    s = s.replace("<mo>&#x0002F;</mo>", '<mo lspace="0" rspace="0">/</mo>')
    s = s.replace('<mo stretchy="false">&#x0007C;</mo>', '<mo stretchy="false" lspace="0" rspace="0">|</mo>')
    # unary minus / plus (after a relation, comma, opening bracket, ...) must use the prefix form, otherwise it is spaced like a binary operator
    s = UNARY.sub(r'\1<mo form="prefix">\2</mo>', s)
    return s


def fill(s):
    for k, v in NUM.items():
        s = s.replace(f"@{k}@", v)
    return s


BAD = []


def conv(tex, display=False):
    try:
        out = L.convert(tex, display="block") if display else L.convert(tex)
    except Exception as e:                      # report unsupported LaTeX instead of silently dropping it
        BAD.append((tex, repr(e)))
        return f"<span style='color:red'>[math error: {_html.escape(tex)}]</span>"
    if "\\" in re.sub(r"&#x[0-9A-Fa-f]+;", "", out):
        BAD.append((tex, "literal backslash in output"))
    return post(out)


def M(tex):
    return conv(fill(tex))


def T(text):
    """text with $...$ inline math and @key@ numeric placeholders"""
    parts = re.split(r"\$(.+?)\$", fill(text))
    return "".join(conv(p) if i % 2 else p for i, p in enumerate(parts))


eq_counter = [0]


def D(tex, label=None, cls=""):
    if label is None:
        eq_counter[0] += 1
        num = f"({eq_counter[0]})"
    elif label == "":
        num = ""
    else:
        num = f"({label})"
    return (f'<div class="eq {cls}"><div class="m">{conv(fill(tex), True)}</div>'
            f'<div class="n">{num}</div></div>')


def img(path, caption, width="100%"):
    data = base64.b64encode(open(path, "rb").read()).decode()
    return (f'<figure><img src="data:image/png;base64,{data}" style="width:{width}">'
            f'<figcaption>{T(caption)}</figcaption></figure>')


CSS = """
@page { size: Letter; margin: 0.7in 0.8in 0.7in 0.8in; }
html { font-size: 10.8pt; }
body { font-family: 'STIX Two Text','STIXGeneral','Latin Modern Roman','Liberation Serif',serif; line-height: 1.36; color:#111; }
math { font-family: 'Latin Modern Math','STIX Two Math','STIX Math','TeX Gyre Termes Math'; font-size: 1.06em; }
h1 { font-size: 21pt; margin: 0 0 2pt 0; line-height: 1.15; }
.sub { font-size: 11.5pt; color:#333; margin-bottom: 6pt; }
.ref { font-size: 9.6pt; color:#444; border-top: 1px solid #999; border-bottom: 1px solid #999; padding: 4pt 0; margin: 6pt 0 10pt 0; }
h2 { font-size: 13.2pt; margin: 15pt 0 4pt 0; border-bottom: 1px solid #bbb; padding-bottom: 1pt; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 9pt 0 2pt 0; page-break-after: avoid; }
p { margin: 3.5pt 0; text-align: left; }
p.lead { page-break-after: avoid; break-after: avoid; }
.eq { display: grid; grid-template-columns: 1fr auto; align-items: center; margin: 4pt 0; page-break-inside: avoid; }
.eq .m { text-align: center; }
.eq .n { padding-left: 10pt; color:#444; font-size: 0.95em; }
.eq.res { background:#eef4fb; border: 1px solid #8fb0d9; border-radius: 3pt; padding: 5pt 7pt; margin: 6pt 0; }
.eq.warn { background:#fdf1ee; border: 1px solid #d9998f; border-radius: 3pt; padding: 5pt 7pt; margin: 6pt 0; }
.box { background:#f6f6f2; border-left: 3pt solid #8a8a6a; padding: 4pt 9pt; margin: 6pt 0; page-break-inside: avoid; }
.box.warn { background:#fdf1ee; border-left-color:#c0594a; }
.step { margin-top: 8pt; font-weight: bold; page-break-after: avoid; }
table { border-collapse: collapse; margin: 6pt auto; font-size: 0.93em; page-break-inside: avoid; break-inside: avoid; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #aaa; padding: 2.5pt 7pt; text-align: center; }
th { background:#eee; }
td.l, th.l { text-align: left; }
table.compact th, table.compact td { padding: 1pt 6pt; }
.ok { color:#1b7f3b; font-weight: bold; } .no { color:#b3261e; font-weight: bold; }
figure { margin: 8pt 0; page-break-inside: avoid; }
figcaption { font-size: 9.4pt; color:#333; margin-top: 2pt; text-align: left; }
ul, ol { margin: 3pt 0 3pt 16pt; padding: 0; } li { margin: 2pt 0; text-align: left; }
code, pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.4pt; }
pre { background:#f3f3f3; border:1px solid #ddd; padding: 5pt 7pt; white-space: pre-wrap; page-break-inside: avoid; }
.small { font-size: 9.4pt; color:#333; }
ul.small li { margin: 1pt 0; }
pre { font-size: 8.1pt; }
"""

H = []
H.append("<h1>From \u0394(\u03b5) to \u0394(E)</h1>")
H.append("<div class='sub'>The gap function of the Rashba + correlated-hopping model as a function of the quasiparticle energy &mdash; "
         "step-by-step derivation, including the band test that discards unphysical roots</div>")
H.append("<div class='ref'>Paper: Joel Hutchinson, J. E. Hirsch, Frank Marsiglio, <i>Enhancement of Superconducting T<sub>c</sub> due to the Spin-orbit Interaction</i>, "
         "Phys. Rev. B <b>97</b>, 184513 (2018), arXiv:1801.08988. &ldquo;Paper Eq. n&rdquo; always refers to that article; the equations of this note are numbered (1), (2), &hellip;</div>")

# ---- result box -----------------------------------------------------------------------------
H.append("<div class='box'><b>Result.</b> " + T(
    "Let $b\\equiv\\Delta^s/4t$, $\\Delta_F\\equiv\\Delta^0-\\mu b$, $E_{\\min}\\equiv|\\Delta_F|/\\sqrt{1+b^2}$, $\\varepsilon^*\\equiv(\\mu+b\\Delta^0)/(1+b^2)$ "
    "and $W\\equiv4t\\sqrt{1+V_{SO}^2/2t^2}$ (half-width of the Rashba band). Solving $E(\\varepsilon)=E$ gives two <i>mathematical</i> branches, "
    "&ldquo;+&rdquo; with $\\varepsilon>\\varepsilon^*$ and &ldquo;&minus;&rdquo; with $\\varepsilon<\\varepsilon^*$, defined for $E\\ge E_{\\min}$:") + "</div>")
H.append(D(r"\Delta_\pm(E)=\frac{\Delta_F}{1+b^2}\mp\frac{b}{\sqrt{1+b^2}}\sqrt{E^2-E_{\min}^2}", label="R", cls="res"))
H.append(D(r"\varepsilon_\pm(E)=\varepsilon^*\pm\frac{\sqrt{E^2-E_{\min}^2}}{\sqrt{1+b^2}}", label="R'", cls="res"))
H.append("<div class='box warn'>" + T(
    "<b>Only branches whose band energy lies inside the Rashba band are real states.</b> The test, applied to each sign separately, is") + "</div>")
H.append(D(r"-W\le\varepsilon_\pm(E)\le+W", label="A", cls="warn"))
H.append("<div class='box warn'>" + T(
    "<b>A branch with $|\\varepsilon_\\pm(E)|>W$ is discarded:</b> no $(\\mathbf{k},s)$ state has quasiparticle energy $E$ on it, so $\\Delta_\\pm(E)$ does not exist there. "
    "For $\\varepsilon^*$ inside the band, $\\Delta(E)$ is therefore two-valued only for $E_{\\min}\\le E\\le\\min(E_{+W},E_{-W})$, single-valued up to $\\max(E_{+W},E_{-W})$ and undefined above that, "
    "where $E_{\\pm W}\\equiv E(\\varepsilon{=}\\pm W)$ is the quasiparticle energy at the band edge (Sec. 3; Sec. 3.3 gives the case of $\\varepsilon^*$ outside the band).") + "</div>")

# ---- 0. which energy ---------------------------------------------------------------------------
H.append("<h2>0. Which energy is which</h2>")
H.append("<p>" + T("Four energies appear:") + "</p>")
H.append("<table><tr><th>symbol</th><th class='l'>meaning</th><th>where</th></tr>"
         "<tr><td>" + M(r"\varepsilon\equiv\varepsilon_{\mathbf{k}s}=\varepsilon_{\mathbf{k}}-2sV_{SO}R_{\mathbf{k}}") + "</td>"
         "<td class='l'>normal-state Rashba band energy, " + T("$s=\\pm1$ helicity, $R_{\\mathbf{k}}\\equiv\\sqrt{\\sin^2k_x+\\sin^2k_y}$") + "</td><td>paper Eq. 15</td></tr>"
         "<tr><td>" + M(r"\xi\equiv\varepsilon-\mu") + "</td><td class='l'>band energy measured from the chemical potential</td><td>&ndash;</td></tr>"
         "<tr><td>" + M(r"E\equiv E_{\mathbf{k}s}=\sqrt{(\varepsilon_{\mathbf{k}s}-\mu)^2+|\Delta_{\mathbf{k}s}|^2}") + "</td>"
         "<td class='l'>Bogoliubov quasiparticle energy: the argument of the Fermi function in the gap equation (paper Eq. 30) and the energy a tunnelling bias measures (<i>eV</i>&nbsp;=&nbsp;<i>E</i>)</td><td>paper Eq. 26</td></tr>"
         "<tr><td>" + M(r"W\equiv4t\sqrt{1+V_{SO}^2/2t^2}") + "</td>"
         "<td class='l'>half-width of the Rashba band: every state has " + T("$-W\\le\\varepsilon_{\\mathbf{k}s}\\le W$") + "</td><td>Sec. 3</td></tr></table>")
H.append("<p>" + T(
    "The paper's result is $\\Delta(\\varepsilon)=\\Delta^0-\\Delta^s\\varepsilon/4t$ (paper Eq. 37). <b>&ldquo;$\\Delta(E)$&rdquo; is taken to mean the gap of the state whose quasiparticle energy is $E$</b>. "
    "If you instead meant $E\\equiv\\varepsilon-\\mu$, nothing needs inverting: (6) below is already $\\Delta(E)=\\Delta_F-bE$ (restricted, as always, to the band, $-W\\le E+\\mu\\le W$).") + "</p>")
H.append("<p class='small'>" + T(
    "Notation clashes in the paper: $g(E)$ is the density of states in paper Eq. 18 (there $E$ is a <i>band</i> energy) and $g(E)\\equiv(1-2f(E))/2E$ after paper Eq. 34 (there $E$ is the <i>quasiparticle</i> energy); "
    "$E_{\\min}$ is the bottom of the free Rashba band in Sec. II and the minimum excitation energy in paper Eq. 40. Here $N(\\varepsilon)$ is the normal-state density of states "
    "($\\int N\\,d\\varepsilon=2$, both helicities), the band bottom is $-W$, and $E_{\\min}$ is the minimum excitation energy of paper Eq. 40.") + "</p>")

# ---- 1. starting point -------------------------------------------------------------------------
H.append("<h2>1. Starting point: why the gap depends on (k, s) only through \u03b5</h2>")
H.append("<p>" + T("The paper's ansatz (paper Eq. 31), with $s_{\\mathbf{k}}=\\frac{1}{2}(\\cos k_x+\\cos k_y)$, is") + "</p>")
H.append(D(r"\bar\Delta_{\mathbf{k}s}=\Delta^0+\Delta^s s_{\mathbf{k}}+\Delta^{x-y}\,s\,R_{\mathbf{k}}"))
H.append("<p>" + T("Paper Eqs. 33 and 34 have the same $\\mathbf{k}'$-sum on the right-hand side, so their ratio gives (paper Eq. 35)") + "</p>")
H.append(D(r"\frac{\Delta^{x-y}}{\Delta^s}=\frac{4(\Delta t/t)\,V_{SO}}{8\,\Delta t}=\frac{V_{SO}}{2t}"))
H.append("<p>" + T("Now use $\\varepsilon_{\\mathbf{k}}=-2t(\\cos k_x+\\cos k_y)$, i.e. $s_{\\mathbf{k}}=-\\varepsilon_{\\mathbf{k}}/4t$, and paper Eq. 15, i.e. $sV_{SO}R_{\\mathbf{k}}=-(\\varepsilon_{\\mathbf{k}s}-\\varepsilon_{\\mathbf{k}})/2$:") + "</p>")
H.append(D(r"s_{\mathbf{k}}+\frac{sV_{SO}}{2t}R_{\mathbf{k}}=-\frac{\varepsilon_{\mathbf{k}}}{4t}-\frac{\varepsilon_{\mathbf{k}s}-\varepsilon_{\mathbf{k}}}{4t}=-\frac{\varepsilon_{\mathbf{k}s}}{4t}"))
H.append("<p>" + T("Inserting (2) and (3) into (1) gives paper Eq. 37:") + "</p>")
H.append(D(r"\bar\Delta_{\mathbf{k}s}=\Delta^0-\frac{\Delta^s}{4t}\,\varepsilon_{\mathbf{k}s}"))
H.append("<p>" + T(
    "Since $\\Delta_{\\mathbf{k}s}=s\\,e^{-i\\theta_{\\mathbf{k}}}\\bar\\Delta_{\\mathbf{k}s}$ (paper, text after Eq. 22) differs from $\\bar\\Delta_{\\mathbf{k}s}$ only by a phase, $|\\Delta_{\\mathbf{k}s}|=|\\bar\\Delta_{\\mathbf{k}s}|$ and the quasiparticle energy depends on $(\\mathbf{k},s)$ "
    "<i>only through</i> $\\varepsilon_{\\mathbf{k}s}$:") + "</p>")
H.append(D(r"E(\varepsilon)=\sqrt{(\varepsilon-\mu)^2+\left(\Delta^0-\frac{\Delta^s}{4t}\,\varepsilon\right)^2}"))
H.append("<p>" + T("All dependence on $\\mathbf{k}$, $s$ and $V_{SO}$ therefore sits in $\\varepsilon_{\\mathbf{k}s}$, in the density of states $N(\\varepsilon)$, in the band edges $\\pm W$ (Sec. 3) and in the self-consistent numbers $\\Delta^0,\\Delta^s,\\mu$. "
                   "The inversion $\\varepsilon\\to E$ is a one-variable problem.") + "</p>")

# ---- 2. inversion ------------------------------------------------------------------------------------
H.append("<h2>2. The inversion: two mathematical roots (Steps 1&ndash;5)</h2>")
H.append("<p>" + T("In Steps 1&ndash;5, $\\varepsilon$ is treated as an arbitrary real number. They give the two <i>mathematical</i> solutions of $E(\\varepsilon)=E$; "
                   "<b>Step 6 (Sec. 3) then decides which of them are physical.</b>") + "</p>")
H.append("<div class='step'>" + T("Step 1 &mdash; measure energies from the Fermi level.") + "</div>")
H.append("<p>" + T("Put $\\xi=\\varepsilon-\\mu$, $b\\equiv\\Delta^s/4t$ and $\\Delta_F\\equiv\\Delta(\\varepsilon{=}\\mu)=\\Delta^0-b\\mu$. Then") + "</p>")
H.append(D(r"\Delta(\xi)=\Delta^0-b(\mu+\xi)=\Delta_F-b\,\xi"))
H.append("<div class='step'>" + T("Step 2 &mdash; write $E^2$ as a polynomial in $\\xi$.") + "</div>")
H.append(D(r"E^2=\xi^2+(\Delta_F-b\xi)^2=(1+b^2)\,\xi^2-2b\Delta_F\,\xi+\Delta_F^2"))
H.append("<p>" + T("This is a convex parabola in $\\xi$ (leading coefficient $1+b^2>0$): $E$ has exactly one minimum, and every larger $E$ is reached at two values of $\\xi$.") + "</p>")
H.append("<div class='step'>" + T("Step 3 &mdash; complete the square.") + "</div>")
H.append(D(r"E^2=E_{\min}^2+(1+b^2)(\xi-\xi^*)^2,\qquad \xi^*=\frac{b\Delta_F}{1+b^2},\quad E_{\min}=\frac{|\Delta_F|}{\sqrt{1+b^2}}"))
H.append("<p class='small'>" + T("Check: $(1+b^2)(\\xi-\\xi^*)^2=(1+b^2)\\xi^2-2b\\Delta_F\\xi+b^2\\Delta_F^2/(1+b^2)$, and adding $E_{\\min}^2=\\Delta_F^2/(1+b^2)$ gives the constant $\\Delta_F^2$, as in (7).") + "</p>")
H.append("<p>" + T("In the paper's variables, $\\varepsilon^*=\\mu+\\xi^*$ and $E_{\\min}$ read") + "</p>")
H.append(D(r"\varepsilon^*=\frac{\mu+\Delta^0\,\Delta^s/4t}{1+(\Delta^s/4t)^2},\qquad E_{\min}=\frac{\left|\Delta^0-\mu\,\Delta^s/4t\right|}{\sqrt{1+(\Delta^s/4t)^2}}"))
H.append("<p>" + T("which are exactly paper Eqs. 39 and 40: a first consistency check. Because $\\xi^*\\neq0$ the minimum of $E(\\varepsilon)$ is <i>not</i> at the Fermi level.") + "</p>")
H.append("<div class='step'>" + T("Step 4 &mdash; solve for $\\xi$.") + "</div>")
H.append(D(r"\xi_\pm(E)=\xi^*\pm\frac{\sqrt{E^2-E_{\min}^2}}{\sqrt{1+b^2}}=\frac{b\Delta_F\pm\sqrt{(1+b^2)E^2-\Delta_F^2}}{1+b^2},\qquad E\ge E_{\min}"))
H.append("<p>" + T("There is no real root for $E<E_{\\min}$. The two roots, $\\varepsilon_\\pm=\\mu+\\xi_\\pm$, lie above and below $\\varepsilon^*$. "
                   "<b>They are only mathematical roots: whether a root corresponds to real states is decided in Step 6.</b>") + "</p>")
H.append("<div class='step'>" + T("Step 5 &mdash; insert $\\xi_\\pm(E)$ into $\\Delta(\\xi)=\\Delta_F-b\\xi$.") + "</div>")
H.append(D(r"\Delta_\pm(E)=\Delta_F-b\,\xi_\pm(E)=\Delta_F-\frac{b^2\Delta_F}{1+b^2}\mp\frac{b\sqrt{(1+b^2)E^2-\Delta_F^2}}{1+b^2}"))
H.append("<p>" + T("Using $\\Delta_F-b^2\\Delta_F/(1+b^2)=\\Delta_F/(1+b^2)$ and $(1+b^2)E^2-\\Delta_F^2=(1+b^2)(E^2-E_{\\min}^2)$ this simplifies to") + "</p>")
H.append(D(r"\Delta_\pm(E)=\frac{\Delta_F}{1+b^2}\mp\frac{b}{\sqrt{1+b^2}}\sqrt{E^2-E_{\min}^2},\qquad E\ge E_{\min}", label="R", cls="res"))
H.append(D(r"\varepsilon_\pm(E)=\mu+\xi_\pm(E)=\varepsilon^*\pm\frac{\sqrt{E^2-E_{\min}^2}}{\sqrt{1+b^2}}", label="R'", cls="res"))
H.append("<p>" + T("Written out with the paper's $\\Delta^0,\\Delta^s,\\mu$:") + "</p>")
H.append(D(r"\Delta_\pm(E)=\frac{\Delta^0-\mu\frac{\Delta^s}{4t}\mp\frac{\Delta^s}{4t}\sqrt{\left[1+\left(\frac{\Delta^s}{4t}\right)^2\right]E^2-\left(\Delta^0-\mu\frac{\Delta^s}{4t}\right)^2}}{1+\left(\frac{\Delta^s}{4t}\right)^2}"))
H.append("<p class='small'>" + T("Equivalent form with $b=\\tan\\varphi$: $\\Delta_\\pm(E)=\\Delta_F\\cos^2\\varphi\\mp\\sin\\varphi\\sqrt{E^2-\\Delta_F^2\\cos^2\\varphi}$.") + "</p>")
H.append("<h3>Geometric picture</h3>")
H.append("<p>" + T("In the $(\\xi,\\Delta)$ plane, (6) is a straight line and $E=\\sqrt{\\xi^2+\\Delta^2}$ is the distance from the origin (Fig. 1b). "
                   "$E_{\\min}=|\\Delta_F|/\\sqrt{1+b^2}$ is the perpendicular distance from the origin to the line, $(\\xi^*,\\Delta^*)$ is the foot of the perpendicular, and a circle of radius $E>E_{\\min}$ cuts the line in two points &mdash; "
                   "the two mathematical branches. Eq. (R) says: walk a distance $\\sqrt{E^2-E_{\\min}^2}$ along the line, in either direction, from the foot of the perpendicular. "
                   "Only the part of the line that lies inside the band is physical (solid in Fig. 1b); intersections on the dotted part are the discarded roots.") + "</p>")
H.append(img("fig1_geometry.png",
             "<b>Fig. 1.</b> (a) The gap $\\Delta(\\varepsilon)$ is linear, but the quasiparticle energy $E(\\varepsilon)$ has a minimum at $\\varepsilon^*\\neq\\mu$, so each $E$ is reached at two values of $\\varepsilon$. "
             "The shaded regions $|\\varepsilon|>W$ contain no states: of the two roots at $E=2t$ only the filled one is a real state, the open one is discarded (Step 6). "
             "(b) The same relation in the $(\\xi,\\Delta)$ plane: a circle of radius $E$ meets the gap line at two points (filled: root inside the band; open: root outside the band, discarded). Parameters as in Fig. 3."))

# ---- 3. band test -----------------------------------------------------------------------------------------
H.append("<h2>3. Step 6 &mdash; keep only the roots that are real states (band test)</h2>")
H.append("<p>" + T("A root $\\varepsilon_\\pm(E)$ describes a physical quasiparticle only if some $(\\mathbf{k},s)$ actually has that band energy, i.e. only if $\\varepsilon_\\pm(E)$ lies inside the Rashba band. "
                   "This single question decides which branches survive.") + "</p>")
H.append("<h3>3.1 The allowed band</h3>")
H.append(D(r"-W\le\varepsilon_{\mathbf{k}s}\le+W,\qquad W=4t\sqrt{1+\frac{V_{SO}^2}{2t^2}}"))
H.append("<p class='lead'>" + T(
    "<i>Proof.</i> For $s=-1$, $\\varepsilon_{\\mathbf{k}s}=-2t(\\cos k_x+\\cos k_y)+2V_{SO}R_{\\mathbf{k}}$. With $u=\\frac{1}{2}(\\cos k_x+\\cos k_y)\\in[-1,1]$ one has "
    "$R_{\\mathbf{k}}^2=2-\\cos^2k_x-\\cos^2k_y\\le2(1-u^2)$, hence, by the Cauchy&ndash;Schwarz inequality,") + "</p>")
H.append(D(r"\varepsilon_{\mathbf{k}s}\le-4tu+2\sqrt2\,V_{SO}\sqrt{1-u^2}\le\sqrt{(4t)^2+8V_{SO}^2}=W", label=""))
H.append("<p>" + T(
    "Equality holds on the diagonals $|k_x|=|k_y|$, so the maximum of $\\varepsilon_{\\mathbf{k}s}$ is exactly $W$; the $s=+1$ band is the mirror image, with minimum $-W$. "
    "Separately, the helicity bands are $s=+1$: $[-W,\\,4t]$ and $s=-1$: $[-4t,\\,W]$ (the points $(\\pi,\\pi)$ and $(0,0)$ give the $\\pm4t$ ends). "
    "$\\varepsilon_{\\mathbf{k}s}$ is continuous on the Brillouin zone, so every energy in $[-W,W]$ is realised and none outside. "
    "Because $\\bar\\Delta$ and $E$ depend on $(\\mathbf{k},s)$ only through $\\varepsilon_{\\mathbf{k}s}$ (Sec. 1), the helicity label plays no role in the test below.") + "</p>")
H.append("<h3>3.2 The admissibility rule</h3>")
H.append(D(r"-W\le\varepsilon_\sigma(E)\le+W,\qquad \varepsilon_\sigma(E)=\varepsilon^*+\sigma\,\frac{\sqrt{E^2-E_{\min}^2}}{\sqrt{1+b^2}},\quad\sigma=\pm1", label="A", cls="warn"))
H.append("<p>" + T(
    "<b>Discard rule.</b> If $|\\varepsilon_\\sigma(E)|>W$, no $(\\mathbf{k},s)$ state has $E_{\\mathbf{k}s}=E$ with $\\varepsilon_{\\mathbf{k}s}=\\varepsilon_\\sigma(E)$. "
    "That root is an artefact of continuing the parabola (7) to band energies that do not exist; it is discarded and $\\Delta_\\sigma(E)$ is <i>undefined</i> at that $E$ "
    "(not zero, not small). It must not enter any plot, average, sum or integral. In Fig. 1 these are the open circles in (b) and the dashed parts of the curves in (a); in Figs. 2 and 3 the dashed lines.") + "</p>")
H.append(img("fig2_band_test.png",
             "<b>Fig. 2.</b> The band test. Band energies of the two roots, $\\varepsilon_\\pm(E)$, against $E$. Solid: root inside the Rashba band (real states). Dashed: root outside the band (discarded). "
             "Black dots: $\\varepsilon_{\\mathbf{k}s}$ against $E_{\\mathbf{k}s}$ for every $(\\mathbf{k},s)$ state &mdash; they lie only on the solid parts. "
             "The bottom strip gives the number of real branches: 0 below $E_{\\min}$, 2 up to $E_{+W}$, 1 up to $E_{-W}$, 0 above.", width="95%"))

H.append("<h3>3.3 The rule written as a condition on E</h3>")
H.append("<p>" + T("By (8), $E$ grows monotonically with $|\\varepsilon-\\varepsilon^*|$. A root therefore reaches a band edge exactly when $E$ equals the quasiparticle energy at that edge,") + "</p>")
H.append(D(r"E_{\pm W}\equiv E(\varepsilon{=}\pm W)=\sqrt{E_{\min}^2+(1+b^2)(\pm W-\varepsilon^*)^2}=\sqrt{(\pm W-\mu)^2+(\Delta^0\mp bW)^2}"))
H.append("<p>" + T("For $\\varepsilon^*$ inside the band ($|\\varepsilon^*|\\le W$, the usual case) the test (A) is therefore equivalent to a simple energy window for each branch:") + "</p>")
H.append("<table><tr><th>branch</th><th>band energy of the root</th><th>real (passes (A)) for</th></tr>"
         "<tr><td>" + M(r"+") + "</td><td>" + M(r"\varepsilon_+\ge\varepsilon^*") + "</td><td>" + M(r"E_{\min}\le E\le E_{+W}") + "</td></tr>"
         "<tr><td>" + M(r"-") + "</td><td>" + M(r"\varepsilon_-\le\varepsilon^*") + "</td><td>" + M(r"E_{\min}\le E\le E_{-W}") + "</td></tr></table>")
H.append("<p>" + T(
    "Consequently: (i) $\\Delta(E)$ is <b>two-valued only</b> for $E_{\\min}\\le E\\le\\min(E_{+W},E_{-W})$; (ii) for $\\min(E_{+W},E_{-W})<E\\le\\max(E_{+W},E_{-W})$ exactly one branch survives and $\\Delta(E)$ is "
    "<b>single-valued</b>; (iii) above $E_{\\max}\\equiv\\max(E_{+W},E_{-W})$ there are no states and $\\Delta(E)$ does not exist ($E_{\\max}$ is the largest quasiparticle energy in the whole zone).") + "</p>")
H.append("<p>" + T(
    "<b>If $\\varepsilon^*$ lies outside the band</b> (e.g. $\\mu$ near a band edge together with large $b$), the $E_{\\min}$ of paper Eq. 40 is not reached by any state and the physical threshold moves to a band-edge energy. "
    "The general windows are") + "</p>")
H.append("<table><tr><th>branch</th><th>real for</th><th>never real if</th></tr>"
         "<tr><td>" + M(r"+") + "</td><td>" + M(r"E(\max(\varepsilon^*,-W))\le E\le E_{+W}") + "</td><td>" + M(r"\varepsilon^*>W") + "</td></tr>"
         "<tr><td>" + M(r"-") + "</td><td>" + M(r"E(\min(\varepsilon^*,W))\le E\le E_{-W}") + "</td><td>" + M(r"\varepsilon^*<-W") + "</td></tr></table>")
H.append("<p class='small'>" + T("with $E(\\varepsilon)$ from (5); for $|\\varepsilon^*|\\le W$ they reduce to the previous table.") + "</p>")
# ---- 4. checks -----------------------------------------------------------------------------------------------
H.append("<h2>4. Checks and properties</h2>")
H.append("<ul>"
         "<li>" + T("<b>BCS limit</b> $b\\to0$ (energy-independent gap): $\\Delta_\\pm(E)\\to\\Delta_F$, $E_{\\min}\\to|\\Delta_F|$, $\\varepsilon_\\pm\\to\\mu\\pm\\sqrt{E^2-\\Delta_F^2}$.") + "</li>"
         "<li>" + T("<b>Threshold</b> $E=E_{\\min}$ (for $|\\varepsilon^*|\\le W$): the root vanishes, both branches meet at $\\Delta^*=\\Delta_F/(1+b^2)=\\Delta(\\varepsilon^*)$.") + "</li>"
         "<li>" + T("<b>High energy</b>: $\\Delta_\\pm(E)\\to\\Delta^*\\mp\\frac{b}{\\sqrt{1+b^2}}E$ (there $E\\approx|\\xi|\\sqrt{1+b^2}$): each formula branch grows linearly with $E$ &mdash; within the band, up to $E_{\\pm W}$.") + "</li>"
         "<li>" + T("<b>Branch splitting</b> (particle&ndash;hole asymmetry), where both branches are real: $\\Delta_-(E)-\\Delta_+(E)=\\frac{2b}{\\sqrt{1+b^2}}\\sqrt{E^2-E_{\\min}^2}$, zero for $b=0$.") + "</li>"
         "<li>" + T("<b>Back-substitution</b>: $\\xi_\\pm^2+\\Delta_\\pm^2=E^2$ identically (checked symbolically with sympy, and numerically for every $(\\mathbf{k},s)$ state, Fig. 3).") + "</li>"
         "<li>" + T("$E_{\\min}=0$ only if $\\Delta_F=0$, i.e. if the node $\\varepsilon_0=4t\\Delta^0/\\Delta^s$ of the gap function lies exactly at $\\mu$.") + "</li>"
         "<li>" + T("<b>Band test, numerically.</b> At $V_{SO}=0.5t$ the helicity bands are $s=+1$: [@sp0@, @sp1@]$\\,t$ and $s=-1$: [@sm0@, @sm1@]$\\,t$, so $W=@W@\\,t$. "
                    "Over all 28,800 states, branch $+$ occupies exactly $E\\in$ [@bfp0@, @bfp1@]$\\,t$ and branch $-$ exactly [@bfm0@, @bfm1@]$\\,t$: the windows of the table above ($E_{\\min}=@Emin@\\,t$, $E_{+W}=@EpW@\\,t$, $E_{-W}=@EmW@\\,t$); "
                    "no state sits on a discarded root. "
                    "Stress test: @nsets@ random sets of $(\\Delta^0,\\Delta^s,\\mu,V_{SO})$, @nout@ of them with $\\varepsilon^*$ outside the band, brute force on a 300&times;300&times;2 grid &mdash; "
                    "no state lies outside the predicted window of its branch, and the upper end of every window (branch reaching a band edge) is filled to within @gap@&nbsp;$t$ (grid resolution).") + "</li>"
         "</ul>")

# ---- 5. recipe -----------------------------------------------------------------------------------------------------
H.append("<h2>5. Recipe for computing &Delta;(E)</h2>")
H.append("<ol>"
         "<li>" + T("Get $\\Delta^0,\\Delta^s,\\mu$ for your $T,U,\\Delta t,n,V_{SO}$ from the self-consistent solution of paper Eqs. 32, 33, 38 ($\\Delta^{x-y}=V_{SO}\\Delta^s/2t$ then follows). "
                    "$T$, $U$, $\\Delta t$ and $n$ enter only through these three numbers; $V_{SO}$ also enters $W$.") + "</li>"
         "<li>" + T("Compute $b=\\Delta^s/4t$, $\\Delta_F=\\Delta^0-\\mu b$, $E_{\\min}=|\\Delta_F|/\\sqrt{1+b^2}$, $\\varepsilon^*=(\\mu+b\\Delta^0)/(1+b^2)$, $\\Delta^*=\\Delta_F/(1+b^2)$ and $W=4t\\sqrt{1+V_{SO}^2/2t^2}$.") + "</li>"
         "<li>" + T("For a given $E$: if $E<E_{\\min}$ there is no state (and none if $E>E_{\\max}$).") + "</li>"
         "<li>" + T("Otherwise put $\\ell=\\sqrt{E^2-E_{\\min}^2}$, $\\varepsilon_\\pm=\\varepsilon^*\\pm\\ell/\\sqrt{1+b^2}$ and $\\Delta_\\pm=\\Delta^*\\mp b\\ell/\\sqrt{1+b^2}$ &mdash; the two mathematical branches.") + "</li>"
         "<li>" + T("<b>Band test:</b> keep a branch only if $|\\varepsilon_\\pm|\\le W$; <b>discard every branch with $|\\varepsilon_\\pm|>W$</b>.") + "</li>"
         "<li>" + T("What remains (zero, one or two values) is $\\Delta(E)$. It keeps the sign of $\\bar\\Delta$: it vanishes at $\\varepsilon_0=4t\\Delta^0/\\Delta^s$, i.e. at $E_0=|\\varepsilon_0-\\mu|$ on the branch that contains $\\varepsilon_0$.") + "</li>"
         "</ol>")

# ---- 6. by-products ------------------------------------------------------------------------------------------------------

H.append("<h2>6. By-products you will probably need next</h2>")
H.append("<p class='lead'>" + T(
    "Differentiating (10) gives the Jacobian, which turns a $\\mathbf{k}$-sum into an energy integral. "
    "With $S(E)=\\{\\sigma:\\,|\\varepsilon_\\sigma(E)|\\le W\\}$, the set of branches that pass the band test (A):") + "</p>")
H.append(D(r"\left|\frac{d\varepsilon_\pm}{dE}\right|=\frac{E}{\sqrt{(1+b^2)E^2-\Delta_F^2}}=\frac{E}{\sqrt{1+b^2}\,\sqrt{E^2-E_{\min}^2}}"))
H.append("<p class='lead'>" + T(
    "so the quasiparticle density of states has a BCS-like inverse-square-root edge at $E_{\\min}$, and the sum runs <i>only over admissible branches</i>:") + "</p>")
H.append(D(r"N_s(E)=\sum_{\sigma\in S(E)}N\left(\varepsilon_\sigma(E)\right)\left|\frac{d\varepsilon_\sigma}{dE}\right|"))
H.append("<p class='lead'>" + T(
    "With the BCS coherence factors $v^2=\\frac{1}{2}(1-\\xi/E)$ (paper Eq. 24) and $u^2=1-v^2$, the $T=0$ tunnelling density of states "
    "(momentum-independent matrix element, spin summed; $\\eta=+1$ adds an electron, weight $u^2$, $\\eta=-1$ removes one, weight $v^2$) is") + "</p>")
H.append(D(r"N_t(\eta E)=\sum_{\sigma\in S(E)}N\left(\varepsilon_\sigma(E)\right)\frac{E+\eta\,\xi_\sigma(E)}{2\sqrt{(1+b^2)E^2-\Delta_F^2}}"))
H.append("<p class='lead'>" + T(
    "$N(\\varepsilon)$ vanishes outside the band, but the closed forms above are finite on a discarded root, so in code the test (A) must be applied explicitly. "
    "At the gap edge (for $|\\varepsilon^*|\\le W$, so that both branches are present) $\\xi_\\pm\\to\\xi^*$ and the two coherence-peak weights are in the ratio") + "</p>")
H.append(D(r"\frac{N_t(+E_{\min})}{N_t(-E_{\min})}=\frac{1+\beta}{1-\beta},\qquad \beta=\frac{\xi^*}{E_{\min}}=\mathrm{sgn}(\Delta_F)\,\frac{b}{\sqrt{1+b^2}}"))
H.append("<p>" + T(
    "This is, in closed form, the density-of-states-independent asymmetry between bias polarities that the paper points out after Eq. 37. "
    "For the example below $\\beta=@beta@$, the ratio is @ratio_pm@, i.e. the $\\eta=-1$ edge is @ratio_inv@ times stronger than the $\\eta=+1$ edge.") + "</p>")

# ---- 7. numerics -----------------------------------------------------------------------------------------------------------------------
H.append("<h2>7. Numerical illustration</h2>")
H.append("<p class='lead'>" + T(
    "Parameters as in the paper's Fig. 8 ($U=90t$, $\\Delta t=4.5t$, $n=1.875$, $k_BT=0.01t$) with $V_{SO}=0.5t$. $\\Delta^0$, $\\Delta^s$, $\\mu$ and $T_c$ come from my own solver of paper Eqs. 32, 33, 38 "
    "(<code>solve_gap.py</code>); everything else follows from the formulas above.") + "</p>")


def table(heads, vals):
    return ("<table><tr>" + "".join(f"<th>{h}</th>" for h in heads) + "</tr><tr>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr></table>")


H.append(table([M(h) for h in (r"\Delta^0/t", r"\Delta^s/t", r"\mu/t", r"k_BT_c/t", r"b", r"\Delta_F/t", r"\varepsilon^*/t", r"\Delta^*/t", r"E_{\min}/t")],
               [NUM["D0"], NUM["Ds"], NUM["mu"], NUM["Tc"], NUM["b"], NUM["DF"], NUM["eps_star"], NUM["Dst"], NUM["Emin"]]))
H.append(table([M(r"W/t"), M(r"E_{+W}/t"), M(r"E_{-W}/t"), M(r"|\varepsilon^*|\le W\,?"), T("$\\varepsilon_0/t$ (node of $\\Delta$)"), M(r"E_0/t")],
               [NUM["W"], NUM["EpW"], NUM["EmW"], "yes", NUM["eps0"], NUM["E0"]]))
H.append("<p class='lead'>" + T(
    "<b>Which branches exist where</b> (this is the band test of Sec. 3 applied to the example):") + "</p>")
H.append("<table><tr><th class='l'>energy range</th><th class='l'>real branches</th><th class='l'>" + M(r"\Delta(E)") + "</th></tr>"
         "<tr><td class='l'>" + T("$E<E_{\\min}=@Emin@\\,t$") + "</td><td class='l'>none</td><td class='l'>undefined</td></tr>"
         "<tr><td class='l'>" + T("$E_{\\min}\\le E\\le E_{+W}=@EpW@\\,t$") + "</td><td class='l'>" + T("$+$ and $-$") + "</td><td class='l'>two-valued</td></tr>"
         "<tr><td class='l'>" + T("$E_{+W}<E\\le E_{-W}=@EmW@\\,t$") + "</td><td class='l'>" + T("$-$ only ($\\varepsilon_+>W$ is discarded)") + "</td><td class='l'>" + T("single-valued, $\\Delta_-(E)$") + "</td></tr>"
         "<tr><td class='l'>" + T("$E>E_{-W}$") + "</td><td class='l'>" + T("none ($\\varepsilon_-<-W$ as well)") + "</td><td class='l'>undefined</td></tr></table>")
H.append(img("fig3_delta_of_E.png",
             "<b>Fig. 3.</b> $\\Delta(E)$ for the example. Solid: the real branches $\\Delta_+$ (red, up to $E_{+W}$) and $\\Delta_-$ (teal, up to $E_{-W}$) from (R). "
             "Black dots: $\\bar\\Delta_{\\mathbf{k}s}$ against $E_{\\mathbf{k}s}$ for every one of the 28,800 states, built directly from paper Eqs. 15, 31, 35 and 26 without any inversion. "
             "Dashed: the same formulas where the root lies outside the band &mdash; these continuations are discarded, no state exists there. "
             "Diamonds mark where a real branch ends; the branches meet at $(E_{\\min},\\Delta^*)$.", width="93%"))
H.append("<p class='lead'>" + T("<b>Worked examples</b> &mdash; each root is tested against $W=@W@\\,t$ before $\\Delta$ is accepted:") + "</p>")
ex_rows = []
for E in (1.0, 2.0, 5.0, 8.0, 8.5):
    rr = delta_of_E(np.array([float(E)]), D0, Ds, mu, VSO=VSO)
    cells = [fm(E, ".1f")]
    for sg in ("plus", "minus"):
        e_, d_, ok_ = float(rr["eps_" + sg][0]), float(rr["Delta_" + sg][0]), bool(rr["ok_" + sg][0])
        cells += [fm(e_, ".3f"),
                  "<span class='ok'>kept</span>" if ok_ else "<span class='no'>discarded</span>",
                  fm(d_, ".3f") if ok_ else "&ndash;"]
    ex_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
H.append("<table><tr><th rowspan='2'>" + M(r"E/t") + "</th><th colspan='3'>" + T("$+$ branch ($\\varepsilon>\\varepsilon^*$)") + "</th><th colspan='3'>" + T("$-$ branch ($\\varepsilon<\\varepsilon^*$)") + "</th></tr>"
         "<tr><th>" + M(r"\varepsilon_+/t") + "</th><th>" + M(r"|\varepsilon_+|\le W\,?") + "</th><th>" + M(r"\Delta_+/t") + "</th>"
         "<th>" + M(r"\varepsilon_-/t") + "</th><th>" + M(r"|\varepsilon_-|\le W\,?") + "</th><th>" + M(r"\Delta_-/t") + "</th></tr>" + "".join(ex_rows) + "</table>")
H.append("<p>" + T(
    "At $E=2t$ the $+$ root sits at $\\varepsilon_+=5.138\\,t>W$: it is discarded, and $\\Delta(2t)=\\Delta_-(2t)=-0.289\\,t$ is single-valued. "
    "At $E=8.5t$ both roots are outside the band: no state has that energy. "
    "Because the gap function changes sign at $\\varepsilon_0=@eps0@\\,t$ (inside the band), $\\Delta_-(E)$ crosses zero at $E_0=|\\varepsilon_0-\\mu|=@E0@\\,t$.") + "</p>")
H.append("<ul>"
         "<li>" + T("<b>Inversion:</b> over all 28,800 $(\\mathbf{k},s)$ states $\\max|\\bar\\Delta_{\\mathbf{k}s}-\\Delta_\\pm(E_{\\mathbf{k}s})|$ = @dev@&nbsp;$t$ (using, for each state, the branch it belongs to); "
                    "the smallest $E_{\\mathbf{k}s}$ in the zone is 0.89914$\\,t$ against the formula value @Emin5@$\\,t$ (paper Eq. 40).") + "</li>"
         "<li>" + T("<b>By-products</b> (test function $F(E)=e^{-E/2}(1+0.3\\cos E)$; direct state sum vs. the energy integrals with the band-masked branch sums; the small differences are histogram binning): "
                    "$N_s$: 0.44651 vs 0.44646; $N_t(+E)$: 0.07996 vs 0.07994; $N_t(-E)$: 0.36655 vs 0.36652.") + "</li>"
         "</ul>")

# comparison with paper Fig. 8
dTc = max(abs(rw[1] / rw[2] - 1) for rw in rows) * 100
dEm = max(abs(rw[3] / rw[4] - 1) for rw in rows) * 100
H.append("<p class='lead'>" + T(
    "<b>Cross-check of my solver against the paper's Fig. 8</b> (same $U,\\Delta t,n,k_BT$; paper values read off the figure, so approximate). "
    "The largest deviations are @dTc@% in $T_c$ and @dEm@% in $E_{\\min}$.".replace("@dTc@", f"{dTc:.0f}").replace("@dEm@", f"{dEm:.0f}")) + "</p>")
tb = ["<table class='compact'><tr><th>" + M(r"V_{SO}/t") + "</th><th>" + M(r"k_BT_c/t") + "</th><th>paper</th><th>" + M(r"E_{\min}/t") + "</th><th>paper</th><th>"
      + M(r"2E_{\min}/k_BT_c") + "</th><th>" + M(r"b") + "</th><th>" + M(r"\beta") + "</th></tr>"]
for (v, Tc_v, Tc_p, E_v, E_p, rat, b_v, be_v) in rows:
    tb.append(f"<tr><td>{v:g}</td><td>{fm(Tc_v, '.3f')}</td><td>{Tc_p:.2f}</td><td>{fm(E_v, '.3f')}</td><td>{E_p:.2f}</td>"
              f"<td>{fm(rat, '.2f')}</td><td>{fm(b_v, '.3f')}</td><td>{fm(be_v, '.3f')}</td></tr>")
tb.append("</table>")
H.append("".join(tb))

# ---- 8. code ---------------------------------------------------------------------------------------------------------------------------------------
H.append("<h2>8. Minimal code and files</h2>")
CODE = '''import numpy as np

def delta_of_E(E, D0, Ds, mu, VSO, t=1.0):
    """Physical branches of Delta(E): (Delta_plus, Delta_minus).
    NaN for E < Emin and where the root eps_pm lies outside the band (|eps| > W)."""
    E = np.asarray(E, float)
    b     = Ds / (4*t)
    DF    = D0 - b*mu                               # Delta at the Fermi level
    s     = np.sqrt(1 + b*b)
    W     = 4*t*np.sqrt(1 + VSO**2/(2*t**2))        # Rashba band:  -W <= eps_ks <= W
    Emin  = abs(DF)/s                               # paper Eq. 40
    estar = (mu + b*D0)/(1 + b*b)                   # paper Eq. 39
    Dstar = DF/(1 + b*b)                            # Delta(eps*)
    with np.errstate(invalid="ignore"):
        ell   = np.sqrt(E**2 - Emin**2)             # NaN below the threshold
        eps_p = estar + ell/s                       # band energy of the "+" root
        eps_m = estar - ell/s                       # band energy of the "-" root
        Dp = np.where(np.abs(eps_p) <= W, Dstar - b*ell/s, np.nan)   # band test (A)
        Dm = np.where(np.abs(eps_m) <= W, Dstar + b*ell/s, np.nan)   # discarded -> NaN
    return Dp, Dm
'''
ns = {}
exec(CODE, ns)
Ec = np.linspace(0.3, 9.5, 4001)
Dp_c, Dm_c = ns["delta_of_E"](Ec, D0, Ds, mu, VSO)
ref = delta_of_E(Ec, D0, Ds, mu, VSO=VSO)
assert np.allclose(Dp_c, ref["Delta_plus_phys"], equal_nan=True) and np.allclose(Dm_c, ref["Delta_minus_phys"], equal_nan=True), "listing differs from module"
assert np.array_equal(np.isnan(Dp_c), ~ref["ok_plus"] | np.isnan(ref["Delta_plus"])), "NaN pattern"
H.append("<pre>" + _html.escape(CODE) + "</pre>")
H.append("<p class='small'>" + T("The listing was executed against the full module when this note was built: it reproduces the band-masked branches of <code>delta_of_E.py</code> on 4001 energies, NaN pattern included.") + "</p>")
H.append("<p class='small'><b>Files</b> (in <code>/home/user/deltaE/</code>): <code>Delta_of_E_derivation.pdf</code> / <code>.html</code> (this note); "
         "<code>delta_of_E.py</code> (formulas, band test, brute-force check over all states, 300-set stress test of the band rule, by-product check, figures); "
         "<code>solve_gap.py</code> (self-consistent solver of paper Eqs. 32, 33, 38; prints the table above); <code>verify_symbolic.py</code> (sympy proof of Steps 1&ndash;5); "
         "<code>fig1_geometry.png</code>, <code>fig2_band_test.png</code>, <code>fig3_delta_of_E.png</code>; <code>build_doc.py</code> (rebuilds this note).</p>")

# ------------------------------------------------------------------------------------------------------------------------------------------------------
if BAD:
    for tex, why in BAD:
        print("MATH PROBLEM:", why, "::", tex)
    raise SystemExit("fix the math above")

HTML = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>From \u0394(\u03b5) to \u0394(E)</title><style>" + CSS + "</style></head><body>"
        + "\n".join(H) + "</body></html>")
open("Delta_of_E_derivation.html", "w", encoding="utf-8").write(HTML)

from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_page()
    pg.goto("file:///home/user/deltaE/Delta_of_E_derivation.html")
    pg.wait_for_timeout(600)
    pg.pdf(path="Delta_of_E_derivation.pdf", format="Letter", print_background=True, prefer_css_page_size=True,
           display_header_footer=True, header_template="<span></span>",
           footer_template="<div style=\"width:100%;font-size:8px;text-align:center;color:#666;font-family:serif\">"
                           "From \u0394(\u03b5) to \u0394(E) &mdash; page <span class='pageNumber'></span> / <span class='totalPages'></span></div>")
    br.close()
print("built; equations numbered:", eq_counter[0], "; <math> count:", HTML.count("<math"))
