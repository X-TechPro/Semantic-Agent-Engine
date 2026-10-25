# -*- coding: utf-8 -*-
"""
Semantic Agent Engine - architecture PDF generator (v3, French + English).

Usage:
    pip install "reportlab>=3.6"
    python build_both.py                    # writes out/..._FR.pdf and out/..._EN.pdf
    python build_both.py --lang en          # English only
    python build_both.py --out-dir pdfs     # choose the output folder
    python build_both.py --fallback-fonts   # force built-in fonts (no download)

Fonts: Carlito (body) + Poppins (titles) are looked up in ./fonts and in the usual system
font folders, then downloaded from Google Fonts into ./fonts. If that fails the script
falls back to Helvetica, so it always produces a PDF.
"""
import os
import sys
import argparse
import urllib.request
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Table, TableStyle, Flowable, PageBreak,
                                NextPageTemplate, KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase.pdfmetrics import stringWidth

# ---------------------------------------------------------------- fonts
HERE = os.path.dirname(os.path.abspath(__file__))
FONTS_UNICODE = True   # False when we fall back to built-in Helvetica

CARLITO = "https://github.com/google/fonts/raw/main/ofl/carlito/"
POPPINS = "https://github.com/google/fonts/raw/main/ofl/poppins/"
FONT_FILES = {                       # alias -> (file name, download base url)
    "Body":    ("Carlito-Regular.ttf", CARLITO),
    "Body-B":  ("Carlito-Bold.ttf", CARLITO),
    "Body-I":  ("Carlito-Italic.ttf", CARLITO),
    "Body-BI": ("Carlito-BoldItalic.ttf", CARLITO),
    "Head":    ("Poppins-Bold.ttf", POPPINS),
    "Head-M":  ("Poppins-Medium.ttf", POPPINS),
}
STD_FALLBACK = {                     # alias -> built-in PDF face
    "Body": "Helvetica", "Body-B": "Helvetica-Bold", "Body-I": "Helvetica-Oblique",
    "Body-BI": "Helvetica-BoldOblique", "Head": "Helvetica-Bold", "Head-M": "Helvetica-Bold",
}


def _font_roots():
    roots = [os.path.join(HERE, "fonts"), os.environ.get("SAE_FONT_DIR", ""),
             os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts"),
             os.path.expanduser("~/Library/Fonts"), "/Library/Fonts", "/System/Library/Fonts",
             "/usr/share/fonts", "/usr/local/share/fonts",
             os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
             os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts")]
    return [r for r in roots if r and os.path.isdir(r)]


def _find_font(fname):
    for root in _font_roots():
        direct = os.path.join(root, fname)
        if os.path.isfile(direct):
            return direct
    for root in _font_roots():
        for dirpath, _dirs, files in os.walk(root):
            if fname in files:
                return os.path.join(dirpath, fname)
    return None


def _download_font(fname, base):
    os.makedirs(os.path.join(HERE, "fonts"), exist_ok=True)
    dest = os.path.join(HERE, "fonts", fname)
    try:
        print("  downloading %s ..." % fname)
        urllib.request.urlretrieve(base + fname, dest)
        if os.path.getsize(dest) < 20000:
            raise IOError("file too small")
        return dest
    except Exception as exc:             # offline, blocked, etc.
        print("  could not download %s (%s)" % (fname, exc))
        if os.path.exists(dest):
            os.remove(dest)
        return None


def setup_fonts(force_fallback=False, allow_download=True):
    """Register Carlito (body) + Poppins (titles). Falls back to Helvetica if unavailable."""
    global FONTS_UNICODE
    paths = {}
    if not force_fallback:
        for alias, (fname, base) in FONT_FILES.items():
            p = _find_font(fname)
            if p is None and allow_download:
                p = _download_font(fname, base)
            if p is None:
                paths = None
                break
            paths[alias] = p
    if paths:
        try:
            for alias, p in paths.items():
                pdfmetrics.registerFont(TTFont(alias, p))
            FONTS_UNICODE = True
            print("fonts: Carlito + Poppins")
        except Exception as exc:
            print("fonts: could not load TTF files (%s)" % exc)
            paths = None
    if not paths:
        for alias, face in STD_FALLBACK.items():
            pdfmetrics.registerFont(pdfmetrics.Font(alias, face, "WinAnsiEncoding"))
        FONTS_UNICODE = False
        print("fonts: built-in Helvetica fallback (layout may differ slightly; arrows become '->')")
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-BI")
    for alias in ("Body-B", "Body-I", "Body-BI", "Head", "Head-M"):   # lets <font>/<b> tags resolve in every style
        pdfmetrics.registerFontFamily(alias, normal=alias, bold=alias, italic=alias, boldItalic=alias)


# ---------------------------------------------------------------- colors
INK = HexColor("#1E2140")
TEXT = HexColor("#2C3050")
MUTED = HexColor("#666B85")
LINE = HexColor("#DADDEA")
BG = HexColor("#F5F6FB")
PURPLE = HexColor("#6A5AE0")
PURPLE_L = HexColor("#EEEBFD")
PURPLE_D = HexColor("#43379B")
TEAL = HexColor("#12907A")
TEAL_L = HexColor("#E2F4F0")
TEAL_D = HexColor("#0B5F51")
AMBER = HexColor("#D98A1F")
AMBER_L = HexColor("#FFF2DA")
AMBER_D = HexColor("#8A5410")
CORAL = HexColor("#D4553E")
CORAL_L = HexColor("#FCE8E3")

LANG = "fr"


def set_lang(l):
    global LANG
    LANG = l


def tx(fr_, en_):
    return en_ if LANG == "en" else fr_


PW, PH = A4
LM = 18 * mm
CW = PW - 2 * LM  # content width


def fr(t):
    """French typography: non-breaking spaces before : ; ? ! and inside « »."""
    if not FONTS_UNICODE:
        t = t.replace("\u2192", "->")
    if LANG == "en":
        return t
    nb = "\u00a0"
    for p in (" :", " ;", " ?", " !", " »"):
        t = t.replace(p, nb + p[1:])
    t = t.replace("« ", "«" + nb)
    return t


# ---------------------------------------------------------------- styles
body = ParagraphStyle("body", fontName="Body", fontSize=10.5, leading=14.8, textColor=TEXT)
body_s = ParagraphStyle("body_s", parent=body, fontSize=9.7, leading=13.4)
cell = ParagraphStyle("cell", parent=body, fontSize=8.9, leading=11.6)
cell_b = ParagraphStyle("cell_b", parent=cell, fontName="Body-B", textColor=INK)
cell_h = ParagraphStyle("cell_h", parent=cell, fontName="Head-M", textColor=colors.white, fontSize=8.2, leading=10.5)
ctitle = ParagraphStyle("ctitle", fontName="Head", fontSize=10.4, leading=13.5, textColor=INK)
bullet = ParagraphStyle("bullet", parent=body_s, leftIndent=11, bulletIndent=1, spaceAfter=2.2)
quote = ParagraphStyle("quote", parent=body, fontName="Body-I", textColor=INK, fontSize=10.8, leading=15.2)
cap = ParagraphStyle("cap", parent=body, fontSize=8.6, leading=11.4, textColor=MUTED)


def P(t, st=body):
    return Paragraph(fr(t), st)


def B(t, st=bullet, color="#6A5AE0"):
    return Paragraph(fr(t), st, bulletText="\u2022")


# ---------------------------------------------------------------- flowables
class Box(Flowable):
    """Rounded card holding paragraphs (floats = vertical spacers)."""

    def __init__(self, items, width=None, bg=BG, stroke=LINE, pad=9, radius=7, lw=0.7):
        super().__init__()
        self.items, self._w = items, width
        self.bg, self.stroke, self.pad, self.radius, self.lw = bg, stroke, pad, radius, lw

    def wrap(self, aw, ah):
        self.w = self._w or aw
        inner = self.w - 2 * self.pad
        self.hs = []
        tot = 0
        for it in self.items:
            if isinstance(it, (int, float)):
                self.hs.append(it)
                tot += it
            else:
                _, h = it.wrap(inner, 10000)
                self.hs.append(h)
                tot += h
        self.h = max(tot + 2 * self.pad, getattr(self, 'minh', 0))
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(self.bg)
        c.setStrokeColor(self.stroke)
        c.setLineWidth(self.lw)
        c.roundRect(0, 0, self.w, self.h, self.radius, stroke=1, fill=1)
        y = self.h - self.pad
        for it, h in zip(self.items, self.hs):
            y -= h
            if not isinstance(it, (int, float)):
                it.drawOn(c, self.pad, y)
        c.restoreState()


class SectionHeader(Flowable):
    def __init__(self, num, title, color=PURPLE, sub=None):
        super().__init__()
        self.num, self.title, self.color, self.sub = num, title, color, sub

    def wrap(self, aw, ah):
        self.w = aw
        self.h = 13 * mm if not self.sub else 17 * mm
        return aw, self.h

    def draw(self):
        c = self.canv
        top = self.h - 1 * mm
        c.setFillColor(self.color)
        c.circle(4.6 * mm, top - 4.6 * mm, 4.6 * mm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Head", 11)
        c.drawCentredString(4.6 * mm, top - 4.6 * mm - 3.9, str(self.num))
        c.setFillColor(INK)
        c.setFont("Head", 15.5)
        c.drawString(12 * mm, top - 6.2 * mm, fr(self.title))
        if self.sub:
            c.setFillColor(MUTED)
            c.setFont("Body", 10)
            c.drawString(12 * mm, top - 11.4 * mm, fr(self.sub))


def equal(*boxes):
    hs = []
    for b in boxes:
        b.minh = 0
        b.wrap(b._w, 10000)
        hs.append(b.h)
    for b in boxes:
        b.minh = max(hs)
    return boxes


def wrap_text(text, font, size, width):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if stringWidth(t, font, size) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class Diagram(Flowable):
    W, H = CW, 104 * mm

    def wrap(self, aw, ah):
        return self.W, self.H

    def rbox(self, x, y, w, h, fill, stroke, title, sub, tcolor=INK, scolor=MUTED, tsize=9.2):
        c = self.canv
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.setLineWidth(0.9)
        c.roundRect(x * mm, y * mm, w * mm, h * mm, 3.2 * mm, stroke=1, fill=1)
        inner = (w - 7) * mm
        sl = wrap_text(fr(sub), "Body", 8.4, inner)
        block = 11 + 10.4 * len(sl)
        top = y * mm + h * mm / 2 + block / 2
        c.setFillColor(tcolor)
        c.setFont("Head-M", tsize)
        c.drawString((x + 3.5) * mm, top - 8, fr(title))
        c.setFillColor(scolor)
        c.setFont("Body", 8.4)
        yy = top - 8 - 12
        for ln in sl:
            c.drawString((x + 3.5) * mm, yy, ln)
            yy -= 10.4

    def arrow(self, x1, y1, x2, y2, color=MUTED, head=True):
        import math
        c = self.canv
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.3)
        c.line(x1 * mm, y1 * mm, x2 * mm, y2 * mm)
        if head:
            a = math.atan2((y2 - y1), (x2 - x1))
            L, Wd = 2.4 * mm, 1.3 * mm
            tx_, ty = x2 * mm, y2 * mm
            p = c.beginPath()
            p.moveTo(tx_, ty)
            p.lineTo(tx_ - L * math.cos(a) + Wd * math.sin(a), ty - L * math.sin(a) - Wd * math.cos(a))
            p.lineTo(tx_ - L * math.cos(a) - Wd * math.sin(a), ty - L * math.sin(a) + Wd * math.cos(a))
            p.close()
            c.drawPath(p, stroke=0, fill=1)

    def label(self, x, y, t, color=MUTED, size=8, font="Body-I"):
        c = self.canv
        c.setFillColor(color)
        c.setFont(font, size)
        c.drawString(x * mm, y * mm, fr(t))

    def draw(self):
        c = self.canv
        # Row 1: session -> read -> decision nodes
        self.rbox(0, 80, 52, 22, BG, LINE,
                  tx("1 · Réunion finie", "1 · Finished session"),
                  tx("source datée", "dated source"))
        self.rbox(61, 80, 52, 22, PURPLE_L, PURPLE,
                  tx("2 · Découper + lire", "2 · Split + read"),
                  tx("messages, nature, statut", "messages, kind, status"), PURPLE_D)
        self.rbox(122, 80, 52, 22, PURPLE_L, PURPLE,
                  tx("3 · Nœuds de décision", "3 · Decision nodes"),
                  tx("messages groupés ; latest = courant",
                     "grouped messages; latest = current"), PURPLE_D)
        self.arrow(52.6, 91, 60.4, 91)
        self.arrow(113.6, 91, 121.4, 91)
        # fork: detector + node view
        c.setStrokeColor(MUTED)
        c.setLineWidth(1.3)
        c.line(148 * mm, 80 * mm, 148 * mm, 73 * mm)
        c.line(148 * mm, 73 * mm, 26 * mm, 73 * mm)
        self.arrow(26, 73, 26, 66.4)
        self.arrow(87, 73, 87, 66.4)
        self.label(28.2, 68.2, tx("propose", "proposes"))
        self.label(89.2, 68.2, tx("lit", "reads"))
        # Row 2: detector -> node view -> answer
        self.rbox(0, 40, 52, 26, PURPLE_L, PURPLE,
                  tx("4 · Détecteur", "4 · Detector"),
                  tx("compare les messages du nœud ; propose le changement",
                     "compares the node's messages; proposes the change"), PURPLE_D)
        self.rbox(61, 40, 52, 26, PURPLE_L, PURPLE,
                  tx("5 · Vue du nœud", "5 · Node view"),
                  tx("current ; as_of(date) ; at_turn(i)",
                     "current; as_of(date); at_turn(i)"), PURPLE_D)
        self.rbox(122, 40, 52, 26, BG, LINE,
                  tx("6 · Répondre", "6 · Answer"),
                  tx("réponse avec preuves citées", "answer with cited evidence"))
        self.arrow(52.6, 53, 60.4, 53)
        self.arrow(113.6, 53, 121.4, 53)
        # Memory and guard rails
        self.rbox(0, 14, 113, 16, TEAL_L, TEAL,
                  tx("Mémoire immuable", "Immutable memory"),
                  tx("messages · dates · nœuds · empreintes",
                     "messages · dates · nodes · fingerprints"), TEAL_D)
        self.arrow(26, 39.6, 26, 30.6, color=TEAL)
        self.arrow(87, 30.4, 87, 39.4, color=TEAL)
        self.label(28.2, 33.4, tx("écrit", "writes"), TEAL_D)
        self.label(89.2, 33.4, tx("lit", "reads"), TEAL_D)
        self.rbox(122, 10, 52, 24, AMBER_L, AMBER,
                  tx("Garde-fous", "Safeguards"),
                  tx("Aucune réécriture. as_of relit le passé. Sans preuve : « je ne sais pas ».",
                     "No rewriting. as_of replays the past. No evidence: “I don't know”."),
                  AMBER_D, HexColor("#6E4A18"))
        # Legend
        items = [(PURPLE_L, PURPLE, tx("partie entraînée", "trained part")),
                 (TEAL_L, TEAL, tx("mémoire", "memory")),
                 (BG, LINE, tx("sans entraînement", "no training")),
                 (AMBER_L, AMBER, tx("sécurité", "safety"))]
        x = 0
        for fill, st, t in items:
            c.setFillColor(fill)
            c.setStrokeColor(st)
            c.setLineWidth(0.9)
            c.roundRect(x * mm, 1.2 * mm, 4.2 * mm, 3.2 * mm, 1 * mm, stroke=1, fill=1)
            c.setFillColor(MUTED)
            c.setFont("Body", 8.4)
            c.drawString((x + 5.6) * mm, 1.8 * mm, fr(t))
            x += 7 + stringWidth(fr(t), "Body", 8.4) / mm + 6



class Timeline(Flowable):
    """3-week plan strip."""

    def __init__(self, weeks):
        super().__init__()
        self.weeks = weeks

    def wrap(self, aw, ah):
        self.w = aw
        self.h = 44 * mm
        return aw, self.h

    def draw(self):
        c = self.canv
        n = len(self.weeks)
        gap = 5 * mm
        bw = (self.w - gap * (n - 1)) / n
        cols = [(PURPLE_L, PURPLE, PURPLE_D), (TEAL_L, TEAL, TEAL_D), (AMBER_L, AMBER, AMBER_D)]
        for i, (title, lines) in enumerate(self.weeks):
            fill, st, dk = cols[i % 3]
            x = i * (bw + gap)
            c.setFillColor(fill)
            c.setStrokeColor(st)
            c.setLineWidth(0.9)
            c.roundRect(x, 0, bw, self.h, 4 * mm, stroke=1, fill=1)
            c.setFillColor(dk)
            c.setFont("Head", 10)
            c.drawString(x + 4 * mm, self.h - 8 * mm, fr(title))
            c.setFillColor(TEXT)
            c.setFont("Body", 9.2)
            y = self.h - 14.5 * mm
            for ln in lines:
                for sub in wrap_text(fr(ln), "Body", 9.2, bw - 12 * mm):
                    c.drawString(x + 8 * mm if sub is not None else x, y, sub)
                    y -= 11.2
                # bullet dot
            # draw bullets
        # second pass for bullets
        for i, (title, lines) in enumerate(self.weeks):
            fill, st, dk = cols[i % 3]
            x = i * (bw + gap)
            y = self.h - 14.5 * mm
            for ln in lines:
                c.setFillColor(st)
                c.circle(x + 5.2 * mm, y + 2.6, 1.3, stroke=0, fill=1)
                n_l = len(wrap_text(fr(ln), "Body", 9.2, bw - 12 * mm))
                y -= 11.2 * n_l


# ---------------------------------------------------------------- page decorations
def on_first(c, doc):
    c.saveState()
    top = PH - 66 * mm
    c.setFillColor(INK)
    c.rect(0, top, PW, 66 * mm, stroke=0, fill=1)
    p = c.beginPath()
    p.rect(0, top, PW, 66 * mm)
    c.clipPath(p, stroke=0, fill=0)
    c.setFillColor(PURPLE)
    c.setFillAlpha(0.55)
    c.circle(PW - 28 * mm, PH - 18 * mm, 40 * mm, stroke=0, fill=1)
    c.setFillColor(TEAL)
    c.setFillAlpha(0.6)
    c.circle(PW - 6 * mm, PH - 58 * mm, 22 * mm, stroke=0, fill=1)
    c.setFillColor(AMBER)
    c.setFillAlpha(0.5)
    c.circle(PW - 74 * mm, PH - 6 * mm, 10 * mm, stroke=0, fill=1)
    c.setFillAlpha(1)
    c.setFillColor(HexColor("#B9B2F5"))
    c.setFont("Head-M", 8.3)
    c.drawString(LM, PH - 15 * mm, "PISCINE MAKERS  \u00b7  EMBEDDING LAB")
    c.setFillColor(colors.white)
    c.setFont("Head", 29)
    c.drawString(LM, PH - 30 * mm, "Semantic Agent Engine")
    c.setFont("Body", 14.5)
    c.drawString(LM, PH - 40.5 * mm, fr(tx("Une mémoire qui sait ce qui a été décidé,", "A memory that knows what was decided,")))
    c.drawString(LM, PH - 47.2 * mm, fr(tx("et ce qui est encore vrai.", "and what is still true.")))
    c.setFillColor(HexColor("#B9B2F5"))
    c.setFont("Body", 10)
    c.drawString(LM, PH - 59 * mm, fr(tx("Architecture v3  ·  Jonathan Chayna  ·  octobre 2026",
                                          "Architecture v3  ·  Jonathan Chayna  ·  October 2026")))
    c.restoreState()
    footer(c, doc)


def on_later(c, doc):
    c.saveState()
    c.setFillColor(PURPLE)
    c.rect(0, PH - 3.2 * mm, PW, 3.2 * mm, stroke=0, fill=1)
    c.setFillColor(TEAL)
    c.rect(PW * 0.62, PH - 3.2 * mm, PW * 0.24, 3.2 * mm, stroke=0, fill=1)
    c.setFillColor(AMBER)
    c.rect(PW * 0.86, PH - 3.2 * mm, PW * 0.14, 3.2 * mm, stroke=0, fill=1)
    c.restoreState()
    footer(c, doc)


def footer(c, doc):
    c.saveState()
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(LM, 13 * mm, PW - LM, 13 * mm)
    c.setFont("Body", 8.5)
    c.setFillColor(MUTED)
    c.drawString(LM, 8.6 * mm, "Semantic Agent Engine  \u00b7  Architecture v3  \u00b7  Piscine Makers")
    c.drawRightString(PW - LM, 8.6 * mm, tx("Page %d", "Page %d") % doc.page)
    c.restoreState()


X = tx


TS = lambda **kw: TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 0)])


def hexc(c):
    return c.hexval().replace("0x", "#")


def title_p(color, text):
    return Paragraph(fr('<font color="%s">%s</font>' % (hexc(color), text)), ctitle)


def row_of(boxes, widths, gapw=4 * mm):
    equal(*boxes)
    cells, cols = [], []
    for i, b in enumerate(boxes):
        if i:
            cells.append("")
            cols.append(gapw)
        cells.append(b)
        cols.append(widths)
    t = Table([cells], colWidths=cols)
    t.setStyle(TS())
    return t


def build(lang, out):
    set_lang(lang)
    story = []
    S = story.append

    def gap(h=4):
        S(Spacer(1, h * mm))

    half = (CW - 4 * mm) / 2
    cw3 = (CW - 8 * mm) / 3

    # ================= PAGE 1 =================
    S(P(X("<b>Architecture v3</b> — la réunion est immuable ; les messages d'une décision forment un nœud ; "
          "`as_of` relit l'état passé.",
          "<b>Architecture v3</b> — the meeting is immutable; one decision's messages form a node; "
          "`as_of` replays the past."), body_s))
    S(SectionHeader("•", X("L'idée en une minute", "The idea in one minute"), PURPLE))
    S(P(X("En réunion, tout n'est pas décidé tout de suite : on propose, on hésite, on change d'avis. "
          "Une recherche classique retrouve les phrases qui <b>se ressemblent</b>, mais pas forcément celle qui est "
          "<b>encore vraie aujourd'hui</b>. Ce moteur lit chaque échange, le range dans une mémoire organisée, "
          "regroupe les messages d'une même décision dans un <b>nœud</b> — le dernier message dit l'état courant — "
          "et peut relire l'état à une date passée sans jamais réécrire la réunion.",
          "In meetings, not everything is decided right away: people propose, hesitate, change their minds. "
          "A classic search finds sentences that <b>look alike</b>, but not necessarily the one that is "
          "<b>still true today</b>. This engine reads every exchange, files it in an organised memory, "
          "groups the messages of one decision into a <b>node</b> — the latest message is the current state — "
          "and can replay the state on a past date without ever rewriting the meeting.")))
    gap(4)
    cards = [
        (PURPLE_L, PURPLE, PURPLE_D, X("Il comprend", "It understands"),
         X("Une proposition n'est pas une décision. « On pourrait peut-être… » ne déclenche rien ; « c'est validé », oui.",
           "A proposal is not a decision. “We could maybe…” triggers nothing; “it's approved” does.")),
        (TEAL_L, TEAL, TEAL_D, X("Il se souvient", "It remembers"),
         X("Chaque information garde sa date, la personne qui l'a dite et la phrase exacte d'origine.",
           "Every piece of information keeps its date, the person who said it and the exact original sentence.")),
        (AMBER_L, AMBER, AMBER_D, X("Il propose, il n'exécute pas", "It proposes, it never executes"),
         X("E-mails, tâches, événements : toujours des brouillons à confirmer. Aucune adresse n'est inventée.",
           "Emails, tasks, events: always drafts to confirm. No address is ever invented.")),
    ]
    S(row_of([Box([title_p(dk, t), 3, P(d, body_s)], width=cw3, bg=bg, stroke=st, pad=9)
              for bg, st, dk, t, d in cards], cw3))
    gap(5)
    S(Box([title_p(AMBER_D, X("L'image à retenir : un atelier bien rangé", "The image to remember: a tidy workshop")), 3,
           P(X("On ne retourne pas tout l'atelier pour trouver une clé à molette : chaque outil est rangé au bon endroit "
               "<b>dès son arrivée</b>. Ici, chaque information est rangée à côté de celles qui lui sont liées, au moment où "
               "elle arrive. Retrouver devient rapide, et quand une information change, un nouveau message rejoint le nœud ; "
               "l'ancien reste exactement ce qui a été dit.",
               "You don't turn the whole workshop upside down to find a wrench: every tool is put in the right place "
               "<b>as soon as it arrives</b>. Here, every piece of information is filed next to the ones it relates to, "
               "the moment it arrives. Finding things becomes fast, and when something changes, a new message joins the node; "
               "the old one remains exactly what was said."), body_s)],
          width=CW, bg=AMBER_L, stroke=AMBER, pad=10))
    gap(6)
    S(SectionHeader("•", X("Un seul cœur, deux usages", "One core, two uses"), TEAL))
    two = [
        Box([title_p(TEAL_D, X("Réunions et messages", "Meetings and messages")), 3,
             P(X("Le sujet de la piscine : retrouver une décision, qui s'est engagé, ce qui a changé entre deux réunions, "
                 "préparer un dossier ou un brouillon d'e-mail.",
                 "The piscine's topic: find a decision, who committed, what changed between two meetings, "
                 "prepare a file or an email draft."), body_s)],
            width=half, bg=TEAL_L, stroke=TEAL),
        Box([title_p(PURPLE_D, X("Documents et code", "Documents and code")), 3,
             P(X("Le même moteur sert de recherche générale : seul le découpage change (un paragraphe cohérent, une fonction). "
                 "La mémoire, les liens et la recherche restent identiques.",
                 "The same engine works as a general search: only the splitting changes (a coherent paragraph, a function). "
                 "Memory, links and search stay identical."), body_s)],
            width=half, bg=PURPLE_L, stroke=PURPLE),
    ]
    S(row_of(two, half))
    S(PageBreak())

    # ================= PAGE 2 =================
    S(SectionHeader(1, X("Le parcours d'un message", "The journey of a message"), PURPLE))
    S(P(X("Voici ce qui se passe quand un message arrive. Exemple utilisé dans tout le document :",
          "Here is what happens when a message arrives. Example used throughout this document:")))
    gap(1.5)
    S(Box([P(X("« Pour le projet X, on repousse le lancement à décembre. Au fait, où en est le back-end ? »",
               "“For project X, we're pushing the launch to December. By the way, where are we on the back end?”"), quote)],
          width=CW, bg=PURPLE_L, stroke=PURPLE, pad=9))
    gap(5)
    S(Diagram())
    gap(3)
    S(Box([title_p(PURPLE_D, X("Sur l'exemple", "On the example")), 3,
           B(X("<b>Morceau A</b> : « on repousse le lancement à décembre ». C'est une information (une décision). Elle est "
               "rangée en mémoire et rejoint le nœud « lancement » ; l'ancienne date reste telle qu'elle a été dite.",
               "<b>Piece A</b>: “we're pushing the launch to December”. This is information (a decision). It is "
               "stored in memory and joins the “launch” node; the old date remains exactly as it was said.")),
           B(X("<b>Morceau B</b> : « où en est le back-end ? ». C'est une question : on cherche dans la mémoire, "
               "<b>après</b> avoir enregistré A, pour que la réponse tienne compte de ce qui vient d'être dit.",
               "<b>Piece B</b>: “where are we on the back end?”. This is a question: we search the memory "
               "<b>after</b> saving A, so the answer reflects what was just said."))],
          width=CW, bg=BG, stroke=LINE, pad=9))
    gap(4)
    S(SectionHeader(2, X("Découper par sujet", "Split by topic"), PURPLE,
                    X("Un morceau = la plus grande portion qui parle encore de la même chose",
                      "One piece = the largest portion still about the same thing")))
    chunk_cards = [
        (PURPLE_L, PURPLE, PURPLE_D, X("Documents", "Documents"),
         X("On regroupe les phrases voisines qui parlent de la même chose. Cinq phrases sur un même sujet : un seul morceau. "
           "Une phrase sur une voiture puis une sur un ballon : deux morceaux.",
           "Neighbouring sentences about the same thing are grouped. Five sentences on one subject: a single piece. "
           "One sentence about a car then one about a ball: two pieces.")),
        (TEAL_L, TEAL, TEAL_D, X("Conversations", "Conversations"),
         X("On coupe dès que le sujet ou l'intention change, même au milieu d'une prise de parole. Dans l'exemple, une "
           "décision puis une question : deux morceaux. Un « OK, validé » très court reste collé au message précédent.",
           "We cut as soon as the topic or intent changes, even mid-turn. In the example, a "
           "decision then a question: two pieces. A very short “OK, approved” stays attached to the previous message.")),
        (AMBER_L, AMBER, AMBER_D, X("Code", "Code"),
         X("Une fonction = un morceau. Ce sont de simples règles de lecture du code, sans intelligence artificielle.",
           "One function = one piece. These are simple code-reading rules, no artificial intelligence.")),
    ]
    S(row_of([Box([title_p(dk, t), 3, P(d, body_s)], width=cw3, bg=bg, stroke=st, pad=9)
              for bg, st, dk, t, d in chunk_cards], cw3))
    gap(2)
    S(P(X("<b>Toujours :</b> même quand un morceau regroupe plusieurs phrases, on garde chaque phrase d'origine comme "
          "<b>preuve</b>. Un gros morceau ne doit pas noyer un détail comme une date ou un nom.",
          "<b>Always:</b> even when a piece groups several sentences, each original sentence is kept as "
          "<b>proof</b>. A big piece must never drown a detail such as a date or a name."), body_s))
    S(PageBreak())

    # ================= PAGE 3 =================
    S(SectionHeader(3, X("La mémoire", "The memory"), TEAL,
                    X("Ce qu'on garde, et comment on le range pour aller vite",
                      "What we keep, and how it is filed for speed")))
    left = Box([title_p(TEAL_D, X("Pour chaque morceau, on garde", "For each piece, we keep")), 3,
                B(X("le <b>texte</b> et la <b>phrase exacte</b> d'origine (la preuve)",
                    "the <b>text</b> and the <b>exact original sentence</b> (the proof)")),
                B(X("sa <b>nature</b> : décision, action, engagement, question, réponse, fait, risque, idée",
                    "its <b>kind</b>: decision, action, commitment, question, answer, fact, risk, idea")),
                B(X("son <b>statut</b> : proposé, confirmé, annulé — et le <b>nœud</b> auquel il appartient",
                    "its <b>status</b>: proposed, confirmed, cancelled — and the <b>node</b> it belongs to")),
                B(X("la <b>date</b>, la <b>personne</b> qui parle, le <b>responsable</b> et l'<b>échéance</b>",
                    "the <b>date</b>, the <b>person</b> speaking, the <b>owner</b> and the <b>deadline</b>")),
                B(X("la <b>source</b> (réunion, document) et le <b>projet</b>",
                    "the <b>source</b> (meeting, document) and the <b>project</b>")),
                B(X("des <b>liens</b> vers les autres messages du même nœud : prolonge, confirme, répond à",
                    "<b>links</b> to the other messages of the same node: continues, confirms, answers")),
                B(X("une <b>empreinte de sens</b> : une suite de nombres qui représente ce que veut dire le morceau",
                    "a <b>meaning fingerprint</b>: a string of numbers that represents what the piece means"))],
               width=half, bg=TEAL_L, stroke=TEAL, pad=10)
    right = Box([title_p(PURPLE_D, X("Comment on le range", "How it is filed")), 3,
                 P(X("<b>Un cahier</b> (base SQLite). Tout y est écrit : texte, dates, personnes, statuts, liens. Filtrer par "
                     "projet ou par date est instantané, et la recherche par mots exacts est déjà intégrée.",
                     "<b>A notebook</b> (SQLite database). Everything is written there: text, dates, people, statuses, links. "
                     "Filtering by project or date is instant, and exact-word search is built in."), body_s), 3,
                 P(X("<b>Un index de sens</b> (FAISS). Deux textes de sens proche ont des empreintes proches ; l'index "
                     "retrouve les plus proches en quelques millisecondes. Chaque entrée renvoie au numéro du morceau dans le cahier.",
                     "<b>A meaning index</b> (FAISS). Two texts with close meaning have close fingerprints; the index "
                     "finds the closest in a few milliseconds. Each entry points to the piece's number in the notebook."), body_s), 3,
                 P(X("<b>Règle d'or : on n'efface jamais.</b> Un nœud garde tous ses messages ; `as_of(date)` relit "
                     "l'état d'une date passée. L'historique reste disponible pour comparer deux réunions.",
                     "<b>Golden rule: never erase.</b> A node keeps every message; `as_of(date)` replays "
                     "the state on a past date. The history stays available to compare two meetings."), body_s)],
                width=half, bg=PURPLE_L, stroke=PURPLE, pad=10)
    S(row_of([left, right], half))
    gap(3)
    S(P(X("<b>Pourquoi pas de simples fichiers JSON ?</b> Ils sont lents à filtrer et à mettre à jour quand la mémoire grandit. "
          "Le JSON reste utile <i>à l'intérieur</i> du cahier, pour les informations supplémentaires de forme libre. "
          "Chaque morceau a deux empreintes : une <b>stable</b> (le sens du morceau seul) et une <b>enrichie par le "
          "contexte</b>. <b>Vérifié :</b> l'empreinte enrichie perd sur données réelles → la stable reste la seule active.",
          "<b>Why not plain JSON files?</b> They are slow to filter and update as memory grows. "
          "JSON stays useful <i>inside</i> the notebook, for free-form extra information. "
          "Each piece has two fingerprints: a <b>stable</b> one (the meaning of the piece alone) and one <b>enriched by "
          "context</b>. <b>Verified:</b> the enriched fingerprint loses on real data → the stable one is the only one active."),
        body_s))
    gap(6)

    S(SectionHeader(4, X("Retrouver la bonne information", "Finding the right information"), PURPLE,
                    X("Six étapes, de la question à la réponse prouvée", "Six steps, from question to proven answer")))
    steps = [
        ("1", X("Comprendre la demande", "Understand the request"),
         X("Un petit modèle rapide devine ce que veut la personne : répondre, comparer, faire un dossier, écrire un e-mail, "
           "créer une tâche, planifier, ou demander une précision. Il repère aussi qui, quoi, pour quand.",
           "A small fast model guesses what the person wants: answer, compare, build a file, write an email, "
           "create a task, schedule, or ask for clarification. It also spots who, what, by when.")),
        ("2", X("Chercher de deux façons", "Search two ways"),
         X("Par le <b>sens</b> (retrouve les reformulations) et par les <b>mots exacts</b> (noms, dates, chiffres). "
           "Les deux listes sont fusionnées.",
           "By <b>meaning</b> (finds rephrasings) and by <b>exact words</b> (names, dates, numbers). "
           "The two lists are merged.")),
        ("3", X("Filtrer selon la demande", "Filter by request"),
         X("Réponse : seulement l'information en vigueur. Comparaison : l'ancienne et la nouvelle, classées par date. "
           "Dossier : tout, avec les messages et les changements proposés signalés.",
           "Answer: only the information in force. Comparison: old and new, sorted by date. "
           "File: everything, with messages and proposed changes flagged.")),
        ("4", X("Départager", "Break ties"),
         X("Un second modèle relit la question avec chaque candidat et les classe du plus au moins pertinent.",
           "A second model rereads the question with each candidate and ranks them from most to least relevant.")),
        ("5", X("Lire l'état du nœud", "Read the node's state"),
         X("Le nœud garde ses messages dans l'ordre. `current` est le dernier ; `as_of(date)` ou "
           "`at_turn(i)` relit l'état à ce moment-là, sans rien réécrire.",
           "The node keeps its messages in order. `current` is the latest; `as_of(date)` or "
           "`at_turn(i)` replays the state at that point, without rewriting anything.")),
        ("6", X("Répondre avec preuves", "Answer with evidence"),
         X("Par défaut, un gabarit rédige la réponse en citant chaque affirmation : <b>réponse par gabarit citable par "
           "défaut ; réécriture LLM optionnelle, validée par un vérificateur</b>, qui renvoie toute réécriture contenant "
           "un nom ou un nombre inventé. Pas assez de preuve : « je n'ai pas assez d'information ». Pour un e-mail, le "
           "destinataire n'est retenu que s'il est clair, récent et unique ; sinon on demande.",
           "By default, a template writes the answer and cites every claim: <b>citable template answer by "
           "default; optional LLM rewrite, checked by a verifier</b> that bounces any rewrite containing "
           "an invented name or number. Not enough evidence: “I don't have enough information”. For an email, the "
           "recipient is only kept if clear, recent and unique; otherwise we ask.")),
    ]
    rows = []
    for n, tt, d in steps:
        rows.append([Paragraph('<font name="Head" size="10.5" color="#FFFFFF">%s</font>' % n,
                               ParagraphStyle("n", alignment=TA_CENTER)),
                     [Paragraph(fr("<b>%s</b>" % tt), cell_b), Paragraph(fr(d), cell)]])
    t5 = Table(rows, colWidths=[8 * mm, CW - 8 * mm])
    sty = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (0, -1), 0), ("RIGHTPADDING", (0, 0), (0, -1), 0),
           ("LEFTPADDING", (1, 0), (1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
           ("LINEBELOW", (1, 0), (1, -2), 0.5, LINE)]
    for i in range(len(steps)):
        sty.append(("BACKGROUND", (0, i), (0, i), PURPLE))
    t5.setStyle(TableStyle(sty))
    S(t5)
    S(PageBreak())

    # ================= PAGE 4 =================
    S(SectionHeader(5, X("Mises à jour", "Updates"), TEAL,
                    X("Vite tout de suite, en profondeur plus tard", "Fast right away, in depth later")))
    now = Box([title_p(TEAL_D, X("Tout de suite (instantané)", "Right away (instant)")), 3,
               B(X("découper et comprendre le nouveau message", "split and understand the new message")),
               B(X("l'enregistrer dans la mémoire", "save it in the memory")),
               B(X("l'ajouter au nœud de sa décision", "append it to its decision node")),
               B(X("ajouter le message au nœud ; le détecteur propose le changement, il ne réécrit rien",
                     "append the message to the node; the detector proposes a change, it rewrites nothing"))],
              width=half, bg=TEAL_L, stroke=TEAL, pad=10)
    later = Box([title_p(PURPLE_D, X("En arrière-plan (par lots, chaque nuit)", "In the background (in batches, nightly)")), 3,
                 B(X("construire les nœuds de décision et la file de revue du détecteur",
                     "build the decision nodes and the detector's review queue")),
                 B(X("reconstruire l'index de sens", "rebuild the meaning index")),
                 B(X("<b>audit de calibration</b> : gare aux seuils périmés ; l'audit recalibre et propose, "
                     "sans jamais remplacer le modèle livré",
                     "<b>calibration audit</b>: guards against stale thresholds; it recalibrates and proposes, "
                     "and never replaces the shipped model")),
                 B(X("proposer un ré-entraînement quand assez de données ont été corrigées",
                     "propose retraining once enough data has been corrected"))],
                width=half, bg=PURPLE_L, stroke=PURPLE, pad=10)
    S(row_of([now, later], half))
    gap(2.5)
    S(P(X("C'est surtout utile pour les <b>messages</b>, qui arrivent en continu : la réaction est immédiate, "
          "le grand rangement se fait par lots, sans jamais ralentir la conversation.",
          "This matters most for <b>messages</b>, which arrive continuously: the reaction is immediate, "
          "the big tidy-up happens in batches, without ever slowing the conversation."), body_s))
    gap(3)

    S(SectionHeader(6, X("Ce qu'on entraîne, et sur quoi", "What we train, and on what"), PURPLE,
                    X("Chaque modèle a un rôle précis et se mesure séparément",
                      "Each model has a precise role and is measured separately")))
    hdr = [Paragraph(fr(x), cell_h) for x in
           [X("Rôle", "Role"), X("Point de départ", "Starting point"),
            X("On l'entraîne sur…", "Trained on…"), X("On mesure…", "We measure…")]]
    sub = lambda t: "<br/><font name='Body' size='8' color='#666B85'>%s</font>" % t
    rows_txt = [
        (X("Découpeur" + sub("où couper"), "Splitter" + sub("where to cut")),
         X("Règle de similarité entre phrases, puis petit modèle multilingue",
           "Sentence-similarity rule, then a small multilingual model"),
         X("Paires de phrases voisines « même sujet / sujet différent », fabriquées en collant des extraits de réunions différentes, + exemples annotés",
           "Pairs of neighbouring sentences “same topic / different topic”, built by gluing excerpts from different meetings, + annotated examples"),
         X("Qualité des coupures", "Cut quality")),
        (X("Lecteur" + sub("nature, statut, qui/quoi/quand, type de demande"),
           "Reader" + sub("kind, status, who/what/when, request type")),
         X("CamemBERT (moderncamembert-base), licence MIT vérifiée",
           "CamemBERT (moderncamembert-base), MIT licence verified"),
         X("Conversations françaises annotées : réunions publiques + dialogues synthétiques vérifiés à la main",
           "Annotated French conversations: public meetings + synthetic dialogues checked by hand"),
         X("F1 par catégorie ; recours à un LLM si « pas sûr »",
           "F1 per category; fall back to an LLM when “not sure”")),
        (X("Moteur de sens" + sub("empreintes"), "Meaning engine" + sub("fingerprints")),
         X("Base déterministe par hachage (aucun téléchargement, aucun GPU) ; e5-base pour les vérifications en français réel ; bge-m3 jamais mesuré",
           "Deterministic hash baseline (no download, no GPU); e5-base for real-French checks; bge-m3 was never measured"),
         X("Paires question / bon passage + « faux amis » (même sujet, mauvais statut, personne ou date)",
           "Question / right-passage pairs + “look-alikes” (same topic, wrong status, person or date)"),
         X("Recall@K, MRR, nDCG ; score sur les faux amis. <b>Version fine-tunée entraînée :</b> gagne ses propres tests, perd le test final → base gelée conservée",
           "Recall@K, MRR, nDCG; score on look-alikes. <b>Fine-tuned version trained:</b> wins its own tests, loses the final test → frozen base kept")),
        (X("Nœuds + détecteur" + sub("groupe, puis propose le changement"), "Nodes + detector" + sub("group, then propose the change")),
         X("Regroupement par sujet (déterministe) + juge 7B en vérificateur sur des paires candidates",
           "Topic grouping (deterministic) + 7B judge as a verifier on candidate pairs"),
         X("53 paires étiquetées à la main (024c, 047a) ; les quatre autres réunions restent à annoter",
           "53 hand-labelled pairs (024c, 047a); the other four meetings still need annotation"),
         X("Reach, pureté, précision/rappel des lignes et des courants",
           "Reach, purity, row and current precision/recall")),
        (X("Trieur final" + sub("classement"), "Final re-sorter" + sub("ranking")),
         X("bge-reranker-v2-m3 prêt à l'emploi", "bge-reranker-v2-m3, off the shelf"),
         X("Rien au départ. Entraîné seulement si les cas difficiles le justifient",
           "Nothing at first. Trained only if hard cases justify it"),
         X("Gain sur les cas difficiles", "Gain on hard cases")),
        (X("Couche « concepts »" + sub("testée, désactivée"), "“Concepts” layer" + sub("tested, switched off")),
         X("Petit réseau qui regarde tous les morceaux ensemble", "Small network that looks at all pieces together"),
         X("Flux simulés : une info arrive, une autre la remplace", "Simulated streams: one item arrives, another replaces it"),
         X("Mesurée sur données réelles en français : elle perd, donc elle reste désactivée. Preuve que les règles de jugement fonctionnent",
           "Measured on real French data: it loses, so it stays off. Proof that the judging rules work")),
        (X("Rédacteur" + sub("réponses, dossiers"), "Writer" + sub("answers, files")),
         X("Gabarit qui cite chaque affirmation (par défaut) ; réécriture LLM optionnelle (drapeau)",
           "Template that cites every claim (default); optional LLM rewrite (flag)"),
         X("Aucun entraînement", "No training"),
         X("Citations exactes ; vérificateur qui rejette tout nom ou nombre inventé",
           "Exact citations; verifier that rejects any invented name or number")),
    ]
    data = [hdr]
    for r in rows_txt:
        data.append([Paragraph(fr(r[0]), cell_b)] + [Paragraph(fr(x), cell) for x in r[1:]])
    t7 = Table(data, colWidths=[30 * mm, 42 * mm, 56 * mm, 46 * mm], repeatRows=1)
    st7 = [("BACKGROUND", (0, 0), (-1, 0), INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
           ("TOPPADDING", (0, 0), (-1, -1), 3.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
           ("LINEBELOW", (0, 1), (-1, -1), 0.5, LINE), ("BOX", (0, 0), (-1, -1), 0.7, LINE),
           ("ROUNDEDCORNERS", [6, 6, 6, 6])]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st7.append(("BACKGROUND", (0, i), (-1, i), BG))
    t7.setStyle(TableStyle(st7))
    S(t7)
    gap(3)
    S(Box([P(X("<b>Lecteur vérifié :</b> Laya écarté (modèle de dialogue, pas un encodeur) ; CamemBERT retenu, licence MIT vérifiée.",
               "<b>Reader verified:</b> Laya ruled out (a dialogue model, not an encoder); CamemBERT kept, MIT licence verified."), body_s)],
          width=CW, bg=AMBER_L, stroke=AMBER, pad=7))
    S(PageBreak())

    # ================= PAGE 5 =================
    S(SectionHeader(7, X("Comment cela répond au sujet", "How this answers the brief"), PURPLE,
                    X("Les quatre difficultés du sujet, et la pièce qui s'en occupe",
                      "The brief's four difficulties, and the part that handles each")))
    diff = [
        [Paragraph(fr(x), cell_h) for x in [X("Difficulté du sujet", "Brief's difficulty"), X("Ce qui s'en occupe", "What handles it")]],
        [Paragraph(fr(X("<b>Pertinence</b><br/>retrouver même avec d'autres mots", "<b>Relevance</b><br/>find it even with other words")), cell),
         Paragraph(fr(X("Moteur de sens + recherche par mots exacts + tri final.", "Meaning engine + exact-word search + final re-sort.")), cell)],
        [Paragraph(fr(X("<b>Précision sémantique</b><br/>proposition, décision, annulation…", "<b>Semantic precision</b><br/>proposal, decision, cancellation…")), cell),
         Paragraph(fr(X("Lecteur de statut + « faux amis » à l'entraînement (même sujet, mauvais statut).",
                        "Status reader + “look-alikes” in training (same topic, wrong status).")), cell)],
        [Paragraph(fr(X("<b>Contexte</b><br/>plusieurs réunions, dates, versions", "<b>Context</b><br/>several meetings, dates, versions")), cell),
         Paragraph(fr(X("Nœuds de décision + `as_of`/`at_turn` + détecteur de changement (revue) + mémoire datée et sourcée.",
                        "Decision nodes + `as_of`/`at_turn` + change detector (review) + dated, sourced memory.")), cell)],
        [Paragraph(fr(X("<b>Intention et destinataire</b><br/>quelle action, pour qui", "<b>Intent and recipient</b><br/>which action, for whom")), cell),
         Paragraph(fr(X("Lecteur de demandes + règle fixe pour le destinataire (clair, récent, unique) + droit de demander ou de dire « je ne sais pas ».",
                        "Request reader + fixed rule for the recipient (clear, recent, unique) + right to ask or to say “I don't know”.")), cell)],
    ]
    t8 = Table(diff, colWidths=[62 * mm, CW - 62 * mm])
    t8.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("LINEBELOW", (0, 1), (-1, -1), 0.5, LINE),
                            ("BOX", (0, 0), (-1, -1), 0.7, LINE), ("ROUNDEDCORNERS", [6, 6, 6, 6]),
                            ("BACKGROUND", (0, 2), (-1, 2), BG), ("BACKGROUND", (0, 4), (-1, 4), BG)]))
    S(t8)
    gap(6)

    S(SectionHeader(8, X("Comment on saura que ça marche", "How we know it works"), TEAL))
    S(B(X("<b>Retrouve-t-on la bonne information ?</b> Recall@K et MRR, comparés à une recherche classique par mots et à un "
          "modèle de sens non entraîné.",
          "<b>Do we find the right information?</b> Recall@K and MRR, compared with a classic word search and an "
          "untrained meaning model.")))
    S(B(X("<b>Distingue-t-on proposition et décision ?</b> Score par catégorie, avec une attention aux négations, aux dates et aux nombres.",
          "<b>Do we tell proposal from decision?</b> Score per category, with attention to negations, dates and numbers.")))
    S(B(X("<b>Le nœud retrace-t-il l'état à chaque tour ?</b> `at_turn` et `as_of` sur les paires d'or : "
          "40/40, et zéro message réécrit.",
          "<b>Does the node replay the state at every turn?</b> `at_turn` and `as_of` on the gold pairs: "
          "40/40, and zero messages rewritten.")))
    S(B(X("<b>Choisit-on le bon destinataire, ou demande-t-on ?</b> Précision, taux de faux destinataires, abstentions justifiées.",
          "<b>Do we pick the right recipient, or ask?</b> Precision, false-recipient rate, justified abstentions.")))
    S(B(X("<b>Les réponses sont-elles prouvées ?</b> Citations exactes et part d'affirmations sans preuve.",
          "<b>Are answers proven?</b> Exact citations and share of unsupported claims.")))
    gap(2)
    S(Box([title_p(TEAL_D, X("Le test décisif : la version en vigueur", "The decisive test: the version in force")), 3,
           P(X("On insère « lancement en décembre » (1<super>er</super> octobre), puis « finalement en janvier » (5 octobre), "
               "puis quelques messages sans rapport, et on demande : « quand est le lancement ? ». <b>Résultat mesuré :</b> "
               "la bonne version sort en premier dans <b>0,6616</b> des cas en moyenne (System B), contre <b>0,2915</b> "
               "pour la recherche classique, sur des projets jamais vus (+0,3629, 3 tirages sur 3). Le test "
               "décembre → janvier figure dans le rapport : la géométrie seule échoue, la couche de relations corrige. "
               "<b>En v3 :</b> la réunion reste la vérité de sa date ; `at_turn`/`as_of` relisent l'état passé "
               "(40/40 sur les paires d'or) et le détecteur propose les changements dans une file de revue. "
               "<b>Honnêtement :</b> l'objectif absolu de 0,90 n'est pas atteint ; le détecteur n'est étiqueté que "
               "sur 2 réunions (P 0,82–1,00, R 0,33–0,57) ; un scan ciblé sur les quatre autres ne retrouve que "
               "2/21 paires d'or, et l'étiquetage des sujets sur parole brute reste ouvert.",
               "We insert “launch in December” (October 1st), then “actually in January” (October 5th), "
               "then a few unrelated messages, and ask: “when is the launch?”. <b>Measured result:</b> "
               "the right version comes first in <b>0.6616</b> of cases on average (System B), versus <b>0.2915</b> "
               "for a classic search, on unseen projects (+0.3629, 3 of 3 seeds). The December → January "
               "test is in the report: geometry alone fails, the relation layer fixes it. "
               "<b>In v3:</b> the meeting remains the truth of its date; `at_turn`/`as_of` replay the past "
               "(40/40 on the gold pairs) and the detector proposes changes into a review queue. "
               "<b>To be honest:</b> the absolute 0.90 target is not met; the detector is labelled on only "
               "2 meetings (P 0.82–1.00, R 0.33–0.57); a targeted scan on the other four reaches only 2/21 "
               "gold changes, and topic tagging on raw speech remains open."), body_s)],
          width=CW, bg=TEAL_L, stroke=TEAL, pad=10))
    S(PageBreak())

    # ---- Preuves
    S(SectionHeader(9, X("Preuves", "Evidence"), TEAL,
                    X("Montré, pas promis", "Shown, not promised")))
    pcards = [
        (TEAL_L, TEAL, TEAL_D, X("Les démos passent", "The demos pass"),
         X("Les 7 démonstrations passent sur le corpus synthétique, et 10/10 sur trois vraies réunions françaises.",
           "The 7 demos pass on the synthetic corpus, and 10/10 on three real French meetings.")),
        (PURPLE_L, PURPLE, PURPLE_D, X("Tout se régénère", "Everything regenerates"),
         X("Chaque chiffre se régénère à partir des artefacts, avec une seule commande.",
           "Every number regenerates from the artefacts with a single command.")),
        (AMBER_L, AMBER, AMBER_D, X("Licences vérifiées", "Licences verified"),
         X("Tous les modèles et tous les jeux de données ont une licence vérifiée.",
           "All models and all datasets carry a verified licence.")),
    ]
    S(row_of([Box([title_p(dk, t), 3, P(d, body_s)], width=cw3, bg=bg, stroke=st, pad=9)
              for bg, st, dk, t, d in pcards], cw3))
    gap(3)
    meas = [
        [Paragraph(fr(x), cell_h) for x in [X("Mesure", "Measure"), X("Résultat", "Result")]],
        [Paragraph(fr(X("Version en vigueur en premier (projets jamais vus)", "Version in force first (unseen projects)")), cell_b),
         Paragraph(fr(X("0,2915 → 0,6616 (+0,3629 ; 3 tirages sur 3) face à la recherche classique",
                        "0.2915 → 0.6616 (+0.3629; 3 of 3 seeds) versus plain search")), cell)],
        [Paragraph(fr(X("v3 — messages rangés dans un nœud", "v3 — messages filed into a node")), cell_b),
         Paragraph(fr(X("99,5 % ; 25/40 paires d'or dans un nœud ; `at_turn(new)` = 40/40 ; 0 mutation",
                        "99.5%; 25/40 gold pairs in one node; `at_turn(new)` = 40/40; 0 mutation")), cell)],
        [Paragraph(fr(X("v3 — courant et détecteur (2 réunions étiquetées)", "v3 — current and detector (2 labelled meetings)")), cell_b),
         Paragraph(fr(X("courants de nœud P 0,90 / R 0,42 ; lignes P 0,82–1,00 / R 0,33–0,57",
                        "node currents P 0.90 / R 0.42; rows P 0.82–1.00 / R 0.33–0.57")), cell)],
        [Paragraph(fr(X("Objectif absolu 0,90", "Absolute 0.90 target")), cell_b),
         Paragraph(fr(X("Non atteint", "Not met")), cell)],
        [Paragraph(fr(X("7 démonstrations", "7 demos")), cell_b),
         Paragraph(fr(X("Réussies sur le corpus synthétique ; 10/10 sur trois vraies réunions françaises (sujets étiquetés à la main)",
                        "Passed on the synthetic corpus; 10/10 on three real French meetings (hand-labelled topics)")), cell)],
        [Paragraph(fr(X("Moteur de sens fine-tuné", "Fine-tuned meaning engine")), cell_b),
         Paragraph(fr(X("Gagne ses propres tests, perd le test final → base gelée conservée",
                        "Wins its own tests, loses the final test → frozen base kept")), cell)],
        [Paragraph(fr(X("Couche « concepts » / empreinte enrichie", "“Concepts” layer / enriched fingerprint")), cell_b),
         Paragraph(fr(X("Perd sur données réelles en français → désactivée", "Loses on real French data → switched off")), cell)],
    ]
    tm = Table(meas, colWidths=[64 * mm, CW - 64 * mm])
    stm = [("BACKGROUND", (0, 0), (-1, 0), INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
           ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("LINEBELOW", (0, 1), (-1, -1), 0.5, LINE),
           ("BOX", (0, 0), (-1, -1), 0.7, LINE), ("ROUNDEDCORNERS", [6, 6, 6, 6])]
    for i in (2, 4):
        stm.append(("BACKGROUND", (0, i), (-1, i), BG))
    tm.setStyle(TableStyle(stm))
    S(tm)
    gap(2.5)
    S(P(X("C'est l'exigence « protocole reproductible » du sujet, <b>montrée plutôt que promise</b>.",
          "This is the brief's “reproducible protocol” requirement, <b>shown rather than promised</b>."), body_s))
    gap(6)

    # ---- Plan
    S(SectionHeader(10, X("Plan et priorités", "Plan and priorities"), AMBER))
    S(Timeline([
        (X("Semaine 1", "Week 1"), [X("Données choisies et documentées", "Data chosen and documented"),
                                    X("Schéma d'annotation", "Annotation schema"),
                                    X("Recherche par mots + sens, sans entraînement : le point de départ chiffré",
                                      "Word + meaning search, no training: the measured starting point")]),
        (X("Semaine 2", "Week 2"), [X("Entraîner le lecteur et le détecteur", "Train the reader and the detector"),
                                    X("Affiner le moteur de sens", "Tune the meaning engine"),
                                    X("Au moins une ablation", "At least one ablation")]),
        (X("Semaine 3", "Week 3"), [X("Agent complet, dossiers, brouillons", "Full agent, files, drafts"),
                                    X("Démonstration en 7 scènes", "7-scene demo"),
                                    X("Rapport : résultats, erreurs, limites", "Report: results, errors, limits")]),
    ]))
    gap(3)
    S(P(X("<b>Si le temps manque, l'ordre est :</b> (1) mémoire + recherche hybride + lecteur + nœuds + détecteur ; (2) moteur de sens affiné ; "
          "(3) découpeur appris ; (4) couche « concepts » comme expérience comparée. Le cœur fonctionne sans la couche (4).",
          "<b>If time runs short, the order is:</b> (1) memory + hybrid search + reader + nodes + detector; (2) tuned meaning engine; "
          "(3) learned splitter; (4) “concepts” layer as a compared experiment. The core works without (4)."), body_s))
    S(PageBreak())

    # ---- Limites
    S(SectionHeader(11, X("Limites mesurées", "Measured limits"), CORAL,
                    X("Ce que les mesures montrent, pas ce qu'on craignait", "What the measurements show, not what we feared")))
    S(B(X("<b>Le détecteur n'est étiqueté que sur 2 réunions sur 6.</b> Rappel 0,42 (8/19 paires d'or) ; "
          "61 des 81 nœuds retombent sur le dernier message, qui peut être du bavardage.",
          "<b>The detector is labelled on only 2 of 6 meetings.</b> Recall 0.42 (8/19 gold pairs); "
          "61 of 81 nodes fall back to the latest message, which may be chatter.")))
    S(B(X("<b>Le courant d'un nœud est le dernier message, pas forcément le dernier changement.</b> C'est la règle demandée ; "
          "la vérité datée se lit avec `at_turn`/`as_of`.",
          "<b>A node's current is the latest message, not necessarily the latest change.</b> That is the requested rule; "
          "the dated truth is read with `at_turn`/`as_of`.")))
    S(B(X("<b>3 questions sans réponse sur 12 laissent encore filtrer une réponse</b> au lieu d'un « je ne sais pas ».",
          "<b>3 of 12 unanswerable questions still leak an answer</b> instead of an “I don't know”.")))
    S(B(X("<b>La relation « répond à » n'est jamais produite</b> par le relieur livré.",
          "<b>The “answers” relation is never produced</b> by the shipped linker.")))
    S(B(X("<b>L'étiquetage des sujets sur de la parole réelle brute</b> est le seul problème ouvert lié à la quantité de "
          "données : il faut plus de réunions annotées.",
          "<b>Topic tagging on raw real speech</b> is the one open problem tied to data scale: "
          "more annotated meetings are needed.")))
    gap(6)

    # ---- Lexique
    S(SectionHeader(12, X("Petit lexique", "Short glossary"), PURPLE, X("Les mots techniques, en clair", "Technical words, in plain language")))
    glos = [
        (X("Empreinte de sens (embedding)", "Meaning fingerprint (embedding)"),
         X("Une suite de nombres qui représente le sens d'un texte. Deux textes de sens proche ont des empreintes proches.",
           "A string of numbers that represents the meaning of a text. Two texts with close meaning have close fingerprints.")),
        (X("Recherche par mots (BM25)", "Word search (BM25)"),
         X("Retrouve les textes qui contiennent les mêmes mots que la question. Très bon pour les noms, dates et chiffres.",
           "Finds texts containing the same words as the question. Very good for names, dates and numbers.")),
        (X("Fusion (RRF)", "Fusion (RRF)"),
         X("Mélange deux classements (par sens et par mots) en un seul, sans avoir à régler de poids.",
           "Merges two rankings (by meaning and by words) into one, without having to tune weights.")),
        (X("Trieur (reranker)", "Re-sorter (reranker)"),
         X("Un modèle qui lit la question et un candidat ensemble pour mieux les classer.",
           "A model that reads the question and a candidate together to rank them better.")),
        (X("Nœud de décision (v3)", "Decision node (v3)"),
         X("Groupe les messages d'une même décision. `current` est le dernier ; `as_of`/`at_turn` relit l'état passé sans réécrire.",
           "Groups the messages of one decision. `current` is the latest; `as_of`/`at_turn` replays the past without rewriting.")),
        (X("Détecteur de changement", "Change detector"),
         X("Compare les messages d'un nœud et propose, dans une file de revue, lequel a changé la valeur.",
           "Compares the messages of a node and proposes, in a review queue, which one changed the value.")),
        (X("Faux amis (hard negatives)", "Look-alikes (hard negatives)"),
         X("Exemples qui se ressemblent mais ne veulent pas dire la même chose : même sujet, mauvais statut, mauvaise date.",
           "Examples that look alike but do not mean the same thing: same topic, wrong status, wrong date.")),
        (X("Entraîner (fine-tuning)", "Train (fine-tuning)"),
         X("Reprendre un modèle déjà appris et le spécialiser sur nos propres exemples.",
           "Take an already-trained model and specialise it on our own examples.")),
        (X("FAISS / SQLite", "FAISS / SQLite"),
         X("Deux outils de rangement : FAISS retrouve par proximité de sens, SQLite stocke tout le reste dans un cahier.",
           "Two filing tools: FAISS finds by closeness of meaning, SQLite stores everything else in a notebook.")),
        (X("LLM", "LLM"),
         X("Modèle de langage qui rédige du texte. Ici, il est optionnel et un vérificateur contrôle ce qu'il écrit.",
           "A language model that writes text. Here it is optional, and a verifier checks what it writes.")),
    ]
    gd = [[Paragraph(fr("<b>%s</b>" % a), cell_b), Paragraph(fr(b), cell)] for a, b in glos]
    tg = Table(gd, colWidths=[52 * mm, CW - 52 * mm])
    tg.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5), ("LEFTPADDING", (0, 0), (-1, -1), 7),
                            ("LINEBELOW", (0, 0), (-1, -2), 0.5, LINE), ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                            ("ROUNDEDCORNERS", [6, 6, 6, 6]), ("BACKGROUND", (0, 0), (0, -1), BG)]))
    S(tg)

    # ================= BUILD =================
    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=LM, rightMargin=LM, topMargin=18 * mm, bottomMargin=19 * mm,
                          title="Semantic Agent Engine - Architecture v3", author="Jonathan Chayna",
                          subject="Piscine Makers - Embedding Lab")
    fw, fh = CW, PH - 18 * mm - 19 * mm
    first = PageTemplate(id="first", frames=[Frame(LM, 19 * mm, fw, PH - 19 * mm - 76 * mm, id="f1", leftPadding=0,
                                                   rightPadding=0, topPadding=0, bottomPadding=0)],
                         onPage=on_first, autoNextPageTemplate="later")
    later = PageTemplate(id="later", frames=[Frame(LM, 19 * mm, fw, fh, id="f2", leftPadding=0, rightPadding=0,
                                                   topPadding=0, bottomPadding=0)], onPage=on_later)
    doc.addPageTemplates([first, later])
    doc.build(story)
    print("ok", out)


def main():
    try:
        import reportlab
        major_minor = tuple(int(x) for x in reportlab.Version.split(".")[:2])
        if major_minor < (3, 6):
            sys.exit("reportlab %s is too old (rounded table corners need >= 3.6): pip install -U reportlab"
                     % reportlab.Version)
    except ValueError:
        pass
    ap = argparse.ArgumentParser(description="Build the Semantic Agent Engine architecture PDFs.")
    ap.add_argument("--lang", choices=["fr", "en", "both"], default="both")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--fallback-fonts", action="store_true", help="use built-in Helvetica, skip font lookup/download")
    ap.add_argument("--no-download", action="store_true", help="never download fonts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_fonts(force_fallback=args.fallback_fonts, allow_download=not args.no_download)
    for lang in (["fr", "en"] if args.lang == "both" else [args.lang]):
        build(lang, os.path.join(args.out_dir, "semantic_agent_engine_architecture_%s.pdf" % lang.upper()))


if __name__ == "__main__":
    main()
