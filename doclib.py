"""Typesetting helpers (LaTeX -> MathML -> HTML -> PDF via headless Chromium), shared by the notes in this folder."""
import base64, re, html as _html
import latex2mathml.converter as L

NUM = {}          # numeric placeholders  @key@  used in the text

UNARY = re.compile(r'(<mo[^>]*>(?!&#x00029;|&#x0005D;|&#x0007D;|&#x0007C;)[^<]*</mo>)<mo>(&#x02212;|&#x0002B;)</mo>')


def post(s):
    # latex2mathml emits numeric character references, e.g. &#x00394; for the Greek capital Delta
    for ch in ("\u0394", "&#x00394;"):
        s = s.replace(f"<mi>{ch}</mi>", '<mi mathvariant="normal">\u0394</mi>')
    s = s.replace("<mi>&#x027E8;</mi>", '<mo stretchy="false" lspace="0" rspace="0">\u27e8</mo>')
    s = s.replace("<mi>&#x027E9;</mi>", '<mo stretchy="false" lspace="0" rspace="0">\u27e9</mo>')
    s = s.replace("<mi>&#x02032;</mi>", '<mo lspace="0" rspace="0">&#x02032;</mo>')
    # thin space after function names (cos, sin, ...), but never inside <msup>/<msub> (the base must stay the only first child)
    s = re.sub(r'(?<!<msup>)(?<!<msub>)(?<!<msubsup>)(<mi>(?:cos|sin|tan|exp|ln|log)</mi>)(?!<mo[^>]*>&#x00028;</mo>)', r'\1<mspace width="0.17em"/>', s)
    s = re.sub(r'(<msup><mi>(?:cos|sin|tan)</mi><mn>\d+</mn></msup>)', r'\1<mspace width="0.17em"/>', s)
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


def table(heads, vals):
    return ("<table><tr>" + "".join(f"<th>{h}</th>" for h in heads) + "</tr><tr>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr></table>")


def build(H, title, basename, footer, extra_css=""):
    """Write <basename>.html and <basename>.pdf from the list of HTML blocks H."""
    if BAD:
        for tex, why in BAD:
            print("MATH PROBLEM:", why, "::", tex)
        raise SystemExit("fix the math above")
    html_text = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>" + title + "</title><style>" + CSS + extra_css + "</style></head><body>"
                 + "\n".join(H) + "</body></html>")
    open(basename + ".html", "w", encoding="utf-8").write(html_text)
    from playwright.sync_api import sync_playwright
    import os
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page()
        pg.goto("file://" + os.path.abspath(basename + ".html"))
        pg.wait_for_timeout(600)
        pg.pdf(path=basename + ".pdf", format="Letter", print_background=True, prefer_css_page_size=True,
               display_header_footer=True, header_template="<span></span>",
               footer_template="<div style=\"width:100%;font-size:8px;text-align:center;color:#666;font-family:serif\">"
                               + footer + " &mdash; page <span class='pageNumber'></span> / <span class='totalPages'></span></div>")
        br.close()
    return html_text
