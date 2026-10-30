# -*- coding: utf-8 -*-
"""
Semantic Agent Engine - architecture PDF generator (v4 default, v3 rollback, French + English).

Usage:
    pip install "reportlab>=3.6"
    python build_both.py                    # writes v4 PDFs (default)
    python build_both.py --variant v3       # rebuild the v3 rollback PDFs
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
import shutil
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
VERSION = "v3"


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
    c.drawString(LM, PH - 59 * mm, fr(tx("Architecture %s  ·  Jonathan Chayna  ·  octobre 2026" % VERSION,
                                          "Architecture %s  ·  Jonathan Chayna  ·  October 2026" % VERSION)))
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
    c.drawString(LM, 8.6 * mm, "Semantic Agent Engine  \u00b7  Architecture %s  \u00b7  Piscine Makers" % VERSION)
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
                          title="Semantic Agent Engine - Architecture %s" % VERSION, author="Jonathan Chayna",
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


class V4Diagram(Flowable):
    """Seven named steps, 4 + 3, with the linker->memory write path in teal."""
    W, H = CW, 71 * mm
    R1_Y, R1_H, R2_Y, R2_H, GAP = 41, 26, 4, 28, 4

    def wrap(self, aw, ah):
        return self.W, self.H

    def rbox(self, x, y, w, h, fill, stroke, num, title, sub,
             tcolor=INK, scolor=MUTED):
        c = self.canv
        c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(0.9)
        c.roundRect(x * mm, y * mm, w * mm, h * mm, 3.4 * mm, stroke=1, fill=1)
        cx, cy = (x + 5.2) * mm, (y + h - 5.2) * mm
        c.setFillColor(stroke); c.circle(cx, cy, 3.0 * mm, stroke=0, fill=1)
        c.setFillColor(colors.white); c.setFont("Head", 8.2)
        c.drawCentredString(cx, cy - 2.9, str(num))
        c.setFillColor(tcolor); c.setFont("Head-M", 9.0)
        c.drawString((x + 10.4) * mm, cy - 3.2, fr(title))
        c.setFillColor(scolor); c.setFont("Body", 7.8)
        yy = (y + h - 11.8) * mm
        for ln in wrap_text(fr(sub), "Body", 7.8, (w - 12) * mm)[:3]:
            c.drawString((x + 10.4) * mm, yy, ln)
            yy -= 5.1 * mm

    def arrow(self, x1, y1, x2, y2, color=MUTED):
        import math
        c = self.canv
        c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(1.3)
        c.line(x1 * mm, y1 * mm, x2 * mm, y2 * mm)
        a = math.atan2((y2 - y1), (x2 - x1)); L, Wd = 2.2 * mm, 1.15 * mm
        tx_, ty = x2 * mm, y2 * mm
        pth = c.beginPath(); pth.moveTo(tx_, ty)
        pth.lineTo(tx_ - L * math.cos(a) + Wd * math.sin(a),
                   ty - L * math.sin(a) - Wd * math.cos(a))
        pth.lineTo(tx_ - L * math.cos(a) - Wd * math.sin(a),
                   ty - L * math.sin(a) + Wd * math.cos(a))
        pth.close(); c.drawPath(pth, stroke=0, fill=1)

    def draw(self):
        c = self.canv
        w1 = (self.W / mm - 3 * self.GAP) / 4
        w2 = (self.W / mm - 2 * self.GAP) / 3
        xs1 = [i * (w1 + self.GAP) for i in range(4)]
        xs2 = [i * (w2 + self.GAP) for i in range(3)]
        top = [
            (BG, LINE, 1, X("store", "store"),
             X("texte, auteur, date, id — jamais modifié",
               "text, speaker, date, id — never modified")),
            (BG, LINE, 2, X("chunker", "chunker"),
             X("phrases + offsets exacts ; « OK » rattaché",
               "sentences + exact offsets; “OK” attached")),
            (PURPLE_L, PURPLE, 3, X("tagger", "tagger"),
             X("nature, statut, point, valeur ; Laya",
               "kind, status, point, value; Laya")),
            (TEAL_L, TEAL, 4, X("linker", "linker"),
             X("même point, valeur différente → supersedes",
               "same point, different value → supersedes")),
        ]
        bottom = [
            (TEAL_L, TEAL, 5, X("memory", "memory"),
             X("état append-only ; courant = dernier confirmé",
               "append-only state; current = latest confirmed")),
            (PURPLE_L, PURPLE, 6, X("retriever", "retriever"),
             X("BM25 + dense + graphe ; Qwen3 rerank",
               "BM25 + dense + graph; Qwen3 rerank")),
            (AMBER_L, AMBER, 7, X("answerer", "answerer"),
             X("contexte + état → LLM → cité ou refus",
               "context + state → LLM → cited or refused")),
        ]
        for i, (fill, st, n, t, sub) in enumerate(top):
            self.rbox(xs1[i], self.R1_Y, w1, self.R1_H, fill, st, n, t, sub,
                      PURPLE_D if st is PURPLE else (TEAL_D if st is TEAL else INK))
            if i:
                self.arrow(xs1[i-1] + w1 + 0.8, self.R1_Y + self.R1_H / 2,
                           xs1[i] - 0.8, self.R1_Y + self.R1_H / 2)
        for i, (fill, st, n, t, sub) in enumerate(bottom):
            self.rbox(xs2[i], self.R2_Y, w2, self.R2_H, fill, st, n, t, sub,
                      PURPLE_D if st is PURPLE else (AMBER_D if st is AMBER else TEAL_D))
            if i:
                self.arrow(xs2[i-1] + w2 + 0.8, self.R2_Y + self.R2_H / 2,
                           xs2[i] - 0.8, self.R2_Y + self.R2_H / 2)
        # the one write path: linker -> memory, drawn as a teal elbow
        bx = xs1[3] + w1 / 2
        mid = self.R1_Y - 6.0
        c.setStrokeColor(TEAL); c.setLineWidth(1.6)
        c.line(bx * mm, self.R1_Y * mm, bx * mm, mid * mm)
        c.line(bx * mm, mid * mm, (xs2[0] + w2 / 2) * mm, mid * mm)
        self.arrow(xs2[0] + w2 / 2, mid, xs2[0] + w2 / 2, self.R2_Y + self.R2_H,
                   color=TEAL)
        c.setFillColor(TEAL_D); c.setFont("Body-I", 7.6)
        c.drawString(96 * mm, (mid + 1.2) * mm,
                     fr(X("ajoute liens + état, sans réécrire",
                          "appends links + state, never rewrites")))



class Legend(Flowable):
    """Four colour swatches, drawn (not typed): the font has no solid square."""

    def __init__(self, items):
        super().__init__()
        self.items = items

    def wrap(self, aw, ah):
        self.w, self.h = aw, 5.6 * mm
        return self.w, self.h

    def draw(self):
        c = self.canv
        x = 0.0
        for fill, stroke, label in self.items:
            c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(0.8)
            c.roundRect(x * mm, 1.0 * mm, 3.2 * mm, 3.2 * mm, 0.9 * mm, stroke=1, fill=1)
            c.setFillColor(MUTED); c.setFont("Body", 8.0)
            c.drawString((x + 4.6) * mm, 1.8 * mm, fr(label))
            x += 4.6 + stringWidth(fr(label), "Body", 8.0) / mm + 6.0


def _steps_table(rows):
    data = []
    for i, (name, desc) in enumerate(rows, 1):
        data.append([Paragraph('<font name="Head" size="10" color="#FFFFFF">%s</font>' % i,
                               ParagraphStyle("n7", alignment=TA_CENTER)),
                     [Paragraph(fr("<b>%s</b>" % name), cell_b),
                      Paragraph(fr(desc), cell)]])
    t = Table(data, colWidths=[8 * mm, CW - 8 * mm])
    sty = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
           ("LEFTPADDING", (0, 0), (0, -1), 0), ("RIGHTPADDING", (0, 0), (0, -1), 0),
           ("LEFTPADDING", (1, 0), (1, -1), 8),
           ("TOPPADDING", (0, 0), (-1, -1), 4.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
           ("LINEBELOW", (1, 0), (1, -2), 0.5, LINE)]
    for i in range(len(rows)):
        sty.append(("BACKGROUND", (0, i), (0, i), PURPLE))
    t.setStyle(TableStyle(sty))
    return t


def build_v4(lang, out):
    global VERSION
    VERSION = "v4"
    set_lang(lang)
    story = []
    S = story.append
    def gap(h=4): S(Spacer(1, h * mm))
    half = (CW - 4 * mm) / 2
    cw3 = (CW - 8 * mm) / 3

    # ================= PAGE 1 — l'idée =================
    S(P(X("<b>Architecture v4</b> — sept étapes nommées, du message qui arrive à la réponse citée. "
          "Chaque étape a une entrée, une sortie, un composant et une mesure.",
          "<b>Architecture v4</b> — seven named steps, from the arriving message to the cited answer. "
          "Every step has an input, an output, a component and a measure."), body_s))
    S(SectionHeader("•", X("L'idée en une minute", "The idea in one minute"), PURPLE))
    S(P(X("Une recherche classique retrouve les phrases qui se ressemblent, pas forcément celle qui est encore vraie. "
          "Ici, le message reste intact, le <b>chunker</b> en tire des phrases avec leurs offsets, le <b>tagger</b> "
          "lit nature, statut, point et valeur, le <b>linker</b> dit ce qui remplace quoi, la <b>memory</b> garde "
          "l'état courant et tout l'historique, le <b>retriever</b> mélange mots, sens et graphe, et l'<b>answerer</b> "
          "répond avec la phrase source — ou refuse.",
          "A classic search finds sentences that look alike, not necessarily the one that is still true. "
          "Here the message stays intact, the <b>chunker</b> extracts sentences with offsets, the <b>tagger</b> "
          "reads kind, status, point and value, the <b>linker</b> says what replaces what, <b>memory</b> keeps the "
          "current state and the whole history, the <b>retriever</b> blends words, meaning and graph, and the "
          "<b>answerer</b> replies with the source sentence — or refuses.")))
    gap(3)
    cards = [
        (PURPLE_L, PURPLE, PURPLE_D, X("Il comprend", "It understands"),
         X("Une proposition n'est pas une décision. Le statut est explicite : proposé, confirmé, annulé, réalisé.",
           "A proposal is not a decision. Status is explicit: proposed, confirmed, cancelled, completed.")),
        (TEAL_L, TEAL, TEAL_D, X("Il se souvient", "It remembers"),
         X("Message, phrase, auteur, date et preuve gardent leur identité. Rien n'est réécrit.",
           "Message, sentence, speaker, date and proof keep their identity. Nothing is rewritten.")),
        (AMBER_L, AMBER, AMBER_D, X("Il ne devine pas", "It does not guess"),
         X("Sans preuve : « je ne sais pas ». Aucun nom, nombre, date ou destinataire inventé.",
           "No evidence: “I don't know”. No invented name, number, date or recipient.")),
    ]
    S(row_of([Box([title_p(dk, t), 3, P(d, body_s)], width=cw3, bg=bg, stroke=st, pad=9)
              for bg, st, dk, t, d in cards], cw3))
    gap(5)
    S(SectionHeader("•", X("Les sept étapes, en une ligne chacune",
                           "The seven steps, one line each"), TEAL))
    for name, line in [
        ("store", X("le message entre tel quel ; la base refuse toute mise à jour",
                    "the message arrives as-is; the database refuses updates")),
        ("chunker", X("phrases et unités, offsets exacts, confirmations courtes rattachées",
                      "sentences and units, exact offsets, short confirmations attached")),
        ("tagger", X("nature, statut, intention, point, valeur — Laya fine-tuné",
                     "kind, status, intent, point, value — fine-tuned Laya")),
        ("linker", X("même point, valeur différente, tour posé → lien supersedes",
                     "same point, different value, settled turn → supersedes link")),
        ("memory", X("état append-only ; courant = dernier confirmé ; l'ancien reste",
                     "append-only state; current = latest confirmed; the old stays")),
        ("retriever", X("BM25 + dense + graphe, fusion, Qwen3-Reranker-0.6B",
                        "BM25 + dense + graph, fusion, Qwen3-Reranker-0.6B")),
        ("answerer", X("contexte + état → LLM ; réponse citée ou refus, chiffres vérifiés",
                       "context + state → LLM; cited answer or refusal, numbers checked")),
    ]:
        S(B("<b>%s</b> — %s" % (name, line)))
    gap(4)
    S(P(X("<b>La règle de fer :</b> le message et la phrase sont immuables ; les liens et les états sont ajoutés, jamais "
          "réécrits. L'état courant est une vue, pas une source de vérité.",
          "<b>Iron rule:</b> message and sentence are immutable; links and states are appended, never rewritten. "
          "Current state is a view, not a source of truth."), body_s))
    gap(3)
    S(Box([title_p(TEAL_D, X("Le chiffre", "The number")), 3,
           P(X("<b>0,778</b> de réponses justes sur 5 réunions écrites à la main et 36 questions ; la bonne phrase "
               "est dans le top 10 dans <b>96 %</b> des cas ; <b>7</b> questions pièges sur 10 sont refusées. "
               "Chiffres régénérés par <font name='Body-B'>make meetings</font>.",
               "<b>0.778</b> correct answers over 5 hand-written meetings and 36 questions; the right sentence is "
               "in the top 10 for <b>96%</b> of them; <b>7</b> of 10 tricky questions are refused. "
               "Regenerated by <font name='Body-B'>make meetings</font>."), body_s)],
          width=CW, bg=TEAL_L, stroke=TEAL, pad=9))
    S(PageBreak())

    # ================= PAGE 2 — le schéma =================
    S(SectionHeader("•", X("Le schéma", "The diagram"), PURPLE,
                    X("Sept composants nommés, un seul chemin : écrire à gauche, lire à droite",
                      "Seven named components, one path: write on the left, read on the right")))
    S(V4Diagram())
    gap(3)
    S(B(X("<b>Ligne du haut</b> — ce qui arrive : le message est stocké, découpé, tagué, puis relié aux tours qui "
          "portaient l'ancienne valeur.",
          "<b>Top row</b> — what arrives: the message is stored, split, tagged, then linked to the turns that carried "
          "the old value.")))
    S(B(X("<b>Ligne du bas</b> — ce qui se lit : la mémoire garde l'état courant, le retriever classe les tours, "
          "l'answerer écrit ou refuse. Le retriever ne modifie jamais la mémoire.",
          "<b>Bottom row</b> — what is read: memory holds the current state, the retriever ranks the turns, the "
          "answerer writes or refuses. The retriever never modifies memory.")))
    S(B(X("<b>Le connecteur</b> entre linker et memory est le seul chemin d'écriture d'état : rien d'autre ne "
          "change l'état courant.",
          "<b>The connector</b> between linker and memory is the only state-writing path: nothing else changes the "
          "current state.")))
    gap(3)
    S(Legend([(PURPLE_L, PURPLE, X("entraîné", "trained")),
              (TEAL_L, TEAL, X("déterministe ou mémoire", "deterministic or memory")),
              (BG, LINE, X("sans entraînement", "no training")),
              (AMBER_L, AMBER, X("contrat et refus", "contract and refusal"))]))
    gap(2)
    S(P(X("Deux étapes n'ont volontairement aucun entraînement : le linker et la mémoire.",
          "Two steps deliberately have no training at all: the linker and memory."), cap))
    gap(3)
    S(Box([title_p(AMBER_D, X("Le contrat, en clair", "The contract, in plain words")), 3,
           P(X("Un message ne change jamais ; une phrase garde ses offsets ; un remplacement est un lien ajouté ; "
               "un état est un événement ajouté ; une réponse cite une phrase ou refuse. C'est ce qui rend le "
               "système auditable : on peut toujours remonter de la réponse au texte exact.",
               "A message never changes; a sentence keeps its offsets; a replacement is an appended link; a state is "
               "an appended event; an answer cites a sentence or refuses. That is what makes the system auditable: "
               "you can always walk back from the answer to the exact text."), body_s)],
          width=CW, bg=AMBER_L, stroke=AMBER, pad=9))
    gap(4)
    S(Box([title_p(TEAL_D, X("Le chemin d'une décision", "The path of a decision")), 3,
           P(X("<b>« on part sur 4 500, c'est acté »</b> → <b>store</b> (t:17, 11/03) → <b>chunker</b> (1 unité) → "
               "<b>tagger</b> (point = prix, valeur = 4 500, confirmé) → <b>linker</b> (remplace les 4 800 du 03/03) → "
               "<b>memory</b> (courant = 4 500, les 4 800 restent) → <b>retriever</b> (top 1) → <b>answerer</b> "
               "(« 4 500, confirmé [traiteur:17:u0] »).",
               "<b>“we're going with 4,500, it's agreed”</b> → <b>store</b> (t:17, 11/03) → <b>chunker</b> (1 unit) → "
               "<b>tagger</b> (point = price, value = 4,500, confirmed) → <b>linker</b> (replaces the 4,800 of 03/03) → "
               "<b>memory</b> (current = 4,500, the 4,800 stays) → <b>retriever</b> (top 1) → <b>answerer</b> "
               "(“4,500, confirmed [traiteur:17:u0]”)."), body_s)],
          width=CW, bg=TEAL_L, stroke=TEAL, pad=9))
    gap(3)
    S(Box([title_p(PURPLE_D, X("Le chemin d'une question sans état", "The path of a question without a state")), 3,
           P(X("<b>« le traiteur c'est combien ? »</b> → <b>retriever</b> (le tour du prix remonte) → aucun état "
               "rattaché à cette unité → <b>answerer</b> cite le tour directement (« 4 500 [traiteur:17:u0] »). "
               "C'est le cas qui échouait avant : un fait vrai sans état n'était pas répondable.",
               "<b>“how much is the caterer?”</b> → <b>retriever</b> (the price turn comes up) → no state attached "
               "to that unit → <b>answerer</b> cites the turn directly (“4,500 [traiteur:17:u0]”). This is the case "
               "that failed before: a true fact with no state was not answerable."), body_s)],
          width=CW, bg=PURPLE_L, stroke=PURPLE, pad=9))
    gap(3)
    S(P(X("Les sept étapes se lisent comme une phrase : le message est stocké, découpé, tagué, relié, "
          "mémorisé, retrouvé, puis cité. Aucune ne réécrit la précédente.",
          "The seven steps read as one sentence: the message is stored, split, tagged, linked, memorised, "
          "retrieved, then cited. None of them rewrites the one before."), body_s))
    S(PageBreak())

    # ================= PAGE 3 — exemple + chunker =================
    S(SectionHeader(1, X("Sur l'exemple", "On the example"), PURPLE,
                    X("Un message, deux morceaux, deux preuves",
                      "One message, two pieces, two proofs")))
    S(Box([P(X("« Pour le projet X, on repousse le lancement à décembre. Au fait, où en est le back-end ? »",
               "“For project X, we're pushing the launch to December. By the way, where are we on the back end?”"), quote)],
          width=CW, bg=PURPLE_L, stroke=PURPLE, pad=9))
    gap(3)
    S(B(X("<b>Morceau A</b> — « on repousse le lancement à décembre » : décision. Le message reste intact ; un "
          "<font name='Body-B'>StateEvent</font> met le lancement à décembre, l'ancienne valeur reste dans l'historique.",
          "<b>Piece A</b> — “we're pushing the launch to December”: decision. The message stays intact; a "
          "<font name='Body-B'>StateEvent</font> sets the launch to December, the old value stays in history.")))
    S(B(X("<b>Morceau B</b> — « où en est le back-end ? » : question. On cherche après avoir enregistré A ; la réponse "
          "cite la phrase source.",
          "<b>Piece B</b> — “where are we on the back end?”: question. We search after storing A; the answer cites "
          "the source sentence.")))
    S(B(X("<b>Un « OK, validé » très court</b> reste attaché au message précédent ; il ne devient pas une phrase isolée.",
          "<b>A very short “OK, approved”</b> stays attached to the preceding message; it does not become an isolated sentence.")))
    gap(5)
    S(SectionHeader(2, X("2 · Chunker — découper sans perdre la preuve",
                         "2 · Chunker — split without losing the proof"), PURPLE))
    chunk_cards = [
        (PURPLE_L, PURPLE, PURPLE_D, X("Documents", "Documents"),
         X("On regroupe les phrases voisines qui parlent de la même chose. Cinq phrases sur un sujet : un seul morceau. "
           "Une phrase sur une voiture puis une sur un ballon : deux morceaux.",
           "Neighbouring sentences about the same thing are grouped. Five sentences on one subject: one piece. "
           "One sentence about a car then one about a ball: two pieces.")),
        (TEAL_L, TEAL, TEAL_D, X("Conversations", "Conversations"),
         X("On coupe dès que le sujet ou l'intention change, même au milieu d'une prise de parole. Un « OK, validé » "
           "court reste collé au message précédent.",
           "We cut as soon as the topic or intent changes, even mid-turn. A short “OK, approved” stays attached "
           "to the preceding message.")),
        (AMBER_L, AMBER, AMBER_D, X("Code", "Code"),
         X("Une fonction = un morceau. Ce sont de simples règles de lecture du code, sans intelligence artificielle.",
           "One function = one piece. Simple code-reading rules, no artificial intelligence.")),
    ]
    S(row_of([Box([title_p(dk, t), 3, P(d, body_s)], width=cw3, bg=bg, stroke=st, pad=9)
              for bg, st, dk, t, d in chunk_cards], cw3))
    gap(2)
    S(P(X("<b>Toujours :</b> même quand un morceau regroupe plusieurs phrases, chaque phrase d'origine garde ses offsets "
          "et reste la <b>preuve</b>. <font name='Body-B'>text[start:end] == span.text</font> sur les 103 tours : 0 écart.",
          "<b>Always:</b> even when a piece groups several sentences, each original sentence keeps its offsets and "
          "remains the <b>proof</b>. <font name='Body-B'>text[start:end] == span.text</font> on all 103 turns: 0 mismatch."), body_s))
    gap(3)
    cut_rows = [
        [Paragraph(fr(x), cell_h) for x in [X("Cas", "Case"), X("Règle", "Rule"), X("Exemple", "Example")]],
        [Paragraph(fr(X("deux phrases, même sujet", "two sentences, same topic")), cell_b), Paragraph(fr(X("un seul morceau", "a single piece")), cell), Paragraph(fr(X("budget proposé puis validé", "budget proposed then approved")), cell)],
        [Paragraph(fr(X("le sujet ou l'intention change", "topic or intent changes")), cell_b), Paragraph(fr(X("on coupe", "we cut")), cell), Paragraph(fr(X("décision puis question", "decision then question")), cell)],
        [Paragraph(fr(X("un « OK, validé » court", "a short “OK, approved”")), cell_b), Paragraph(fr(X("rattaché au message précédent", "attached to the previous message")), cell), Paragraph(fr(X("il ne devient pas une unité", "it does not become a unit")), cell)],
        [Paragraph(fr(X("du code", "code")), cell_b), Paragraph(fr(X("une fonction = un morceau", "one function = one piece")), cell), Paragraph(fr(X("règles de lecture, sans IA", "reading rules, no AI")), cell)],
    ]
    tcut = Table(cut_rows, colWidths=[46*mm, 58*mm, CW-104*mm], repeatRows=1)
    scut = [("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ("LEFTPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,1),(-1,-1),0.5,LINE),
            ("BOX",(0,0),(-1,-1),0.7,LINE),("ROUNDEDCORNERS",[5,5,5,5]),
            ("BACKGROUND",(0,2),(-1,2),BG),("BACKGROUND",(0,4),(-1,4),BG)]
    tcut.setStyle(TableStyle(scut))
    S(tcut)
    gap(4)
    S(P(X("<b>Le contrat immuable</b>", "<b>The immutable contract</b>"), cell_b))
    schema = [
        [Paragraph(fr(X("Objet", "Record")), cell_h), Paragraph(fr(X("Ce qui ne change jamais", "What never changes")), cell_h)],
        [Paragraph("Message", cell_b), Paragraph(fr(X("message_id, source, project, speaker, timestamp, text, hash",
                                                       "message_id, source, project, speaker, timestamp, text, hash")), cell)],
        [Paragraph("Chunk", cell_b), Paragraph(fr(X("chunk_id, message_id, start, end, text, sentence_index",
                                                     "chunk_id, message_id, start, end, text, sentence_index")), cell)],
        [Paragraph("Unit", cell_b), Paragraph(fr(X("unit_id, message_id, project, slot, point, value, status, kind, evidence",
                                                    "unit_id, message_id, project, slot, point, value, status, kind, evidence")), cell)],
        [Paragraph("LinkEvent", cell_b), Paragraph(fr(X("ajouté : from, to, relation, confidence, evidence",
                                                        "appended: from, to, relation, confidence, evidence")), cell)],
        [Paragraph("StateEvent", cell_b), Paragraph(fr(X("ajouté : event (projet:slot:point), valeur, statut, evidence, date",
                                                         "appended: event (project:slot:point), value, status, evidence, date")), cell)],
    ]
    ts = Table(schema, colWidths=[28*mm, CW-28*mm])
    ts.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),5),("LINEBELOW",(0,1),(-1,-1),0.5,LINE),
        ("BOX",(0,0),(-1,-1),0.7,LINE),("ROUNDEDCORNERS",[5,5,5,5]),
        ("BACKGROUND",(0,2),(-1,2),BG)]))
    S(ts)
    gap(2)
    S(P(X("Le <b>point</b> est le discriminateur qui manquait : sans lui, « clés », « fibre » et « bail » partageaient "
          "la même clé de date. La clé est <font name='Body-B'>(projet, slot, point)</font>.",
          "The <b>point</b> is the missing discriminator: without it, “keys”, “fibre” and “lease” shared one date key. "
          "The key is <font name='Body-B'>(project, slot, point)</font>."), body_s))
    S(PageBreak())

    # ================= PAGE 4 — les sept étapes =================
    S(SectionHeader(3, X("Les sept étapes", "The seven steps"), PURPLE,
                    X("Chacune a un nom, une entrée, une sortie et une mesure",
                      "Each has a name, an input, an output and a measure")))
    steps7 = [
        (X("store", "store"),
         X("Le message entre tel quel : identifiant, auteur, date, texte. La base refuse toute mise à jour ; "
           "ré-ingérer le même message ne fait rien. 0 doublon.",
           "The message arrives as-is: id, speaker, date, text. The database refuses updates; re-ingesting the same "
           "message does nothing. 0 duplicates.")),
        (X("chunker", "chunker"),
         X("Découpe en phrases et unités, avec offsets exacts. Les confirmations courtes restent rattachées. "
           "Sortie : des chunks qui pointent vers leur texte.",
           "Splits into sentences and units with exact offsets. Short confirmations stay attached. "
           "Output: chunks pointing at their text.")),
        (X("tagger", "tagger"),
         X("Laya multilingue fine-tuné lit chaque unité : nature, statut, intention, point et valeur. "
           "Rappel des changements 1,00 ; 32 valeurs sur 32 sont des valeurs et pas des bouts de phrase.",
           "Fine-tuned multilingual Laya reads each unit: kind, status, intent, point and value. "
           "Change recall 1.00; 32 of 32 values are values, not sentence fragments.")),
        (X("linker", "linker"),
         X("Règle déterministe : même projet, même slot, même point, valeur différente, tour nouveau et posé → "
           "un lien <font name='Body-B'>supersedes</font> vers le tour le plus récent qui portait l'ancienne valeur. "
           "4 remplacements sur 4 reliés.",
           "Deterministic rule: same project, slot and point, different value, settled new turn → a "
           "<font name='Body-B'>supersedes</font> link to the latest turn carrying the old value. 4 of 4 replacements linked.")),
        (X("memory", "memory"),
         X("État append-only : une proposition reste ouverte, une confirmation met à jour la vue courante, "
           "l'ancien reste dans l'historique. 12 valeurs courantes sur 12 ; 0 fuite de statut entre unités.",
           "Append-only state: a proposal stays open, a confirmation updates the current view, the old value stays in "
           "history. 12 of 12 current values; 0 status leak between units.")),
        (X("retriever", "retriever"),
         X("Entités et événements, puis BM25 + dense, expansion de graphe, fusion des classements et reranking "
           "Qwen3-Reranker-0.6B. hit@1 .577 · hit@3 .846 · hit@10 .962.",
           "Entities and events, then BM25 + dense, graph expansion, rank fusion and Qwen3-Reranker-0.6B. "
           "hit@1 .577 · hit@3 .846 · hit@10 .962.")),
        (X("answerer", "answerer"),
         X("La question, les tours retrouvés (id, date, auteur) et l'état courant partent au LLM ; la réponse n'est "
           "gardée que si elle cite un extrait fourni et que ses chiffres y figurent — sinon refus. "
           "Cite la bonne phrase .808 ; 7 pièges sur 10 refusés.",
           "The question, the retrieved turns (id, date, speaker) and the current state go to the LLM; the answer is "
           "kept only if it cites a supplied excerpt and its numbers appear there — otherwise refusal. "
           "Cites the right sentence .808; 7 of 10 tricky questions refused.")),
    ]
    S(_steps_table(steps7))
    gap(4)
    S(P(X("<b>Ce que chaque étape reçoit et rend</b>", "<b>What each step receives and returns</b>"), cell_b))
    io_rows = [
        [Paragraph(fr(x), cell_h) for x in [X("Étape", "Step"), X("Reçoit", "Receives"), X("Rend", "Returns")]],
        [Paragraph("store", cell_b), Paragraph(fr(X("message brut", "raw message")), cell), Paragraph(fr(X("ligne immuable", "immutable row")), cell)],
        [Paragraph("chunker", cell_b), Paragraph(fr(X("message", "message")), cell), Paragraph(fr(X("chunks + offsets", "chunks + offsets")), cell)],
        [Paragraph("tagger", cell_b), Paragraph(fr(X("chunk", "chunk")), cell), Paragraph(fr(X("nature, point, valeur, statut", "kind, point, value, status")), cell)],
        [Paragraph("linker", cell_b), Paragraph(fr(X("unité + tours antérieurs", "unit + older turns")), cell), Paragraph(fr(X("lien supersedes, ou rien", "supersedes link, or nothing")), cell)],
        [Paragraph("memory", cell_b), Paragraph(fr(X("liens + états", "links + states")), cell), Paragraph(fr(X("vue courante + historique", "current view + history")), cell)],
        [Paragraph("retriever", cell_b), Paragraph(fr(X("question", "question")), cell), Paragraph(fr(X("10 tours classés", "10 ranked turns")), cell)],
        [Paragraph("answerer", cell_b), Paragraph(fr(X("question + tours + état", "question + turns + state")), cell), Paragraph(fr(X("réponse citée ou refus", "cited answer or refusal")), cell)],
    ]
    tio = Table(io_rows, colWidths=[28*mm, 58*mm, CW-86*mm], repeatRows=1)
    sio = [("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
           ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
           ("LEFTPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,1),(-1,-1),0.5,LINE),
           ("BOX",(0,0),(-1,-1),0.7,LINE),("ROUNDEDCORNERS",[5,5,5,5])]
    for i in (2, 4, 6):
        sio.append(("BACKGROUND",(0,i),(-1,i),BG))
    tio.setStyle(TableStyle(sio))
    S(tio)
    gap(3)
    S(Box([title_p(PURPLE_D, X("Ce qui change par rapport à v2/v3", "What changes from v2/v3")), 3,
           B(X("<b>v2</b> avait le bon parcours, mais « remplacer » pouvait réécrire l'ancien message.",
               "<b>v2</b> had the right journey, but “replace” could rewrite the old message.")),
           B(X("<b>v3</b> rendait la réunion immuable, mais le courant venait du dernier message, pas d'un état confirmé.",
               "<b>v3</b> made the meeting immutable, but current came from the latest message, not a confirmed state.")),
           B(X("<b>v4</b> nomme chaque étape, ajoute les offsets, le point, les liens typés et l'état append-only ; "
               "le courant est le dernier état confirmé, pas le dernier bavardage.",
               "<b>v4</b> names every step, adds offsets, the point, typed links and append-only state; "
               "current is the latest confirmed state, not the latest chatter."))],
          width=CW, bg=BG, stroke=LINE, pad=8))
    S(PageBreak())

    # ================= PAGE 5 — retriever =================
    S(SectionHeader(4, X("6 · Retriever — retrouver, puis lire l'état",
                         "6 · Retriever — find it, then read the state"), TEAL,
                    X("Le cœur s'arrête ici : la réponse est une génération sur ce contexte",
                      "The core stops here: the answer is a generation over this context")))
    layers = [
        (X("Interpréter la question", "Interpret the query"),
         X("Intention, slot demandé, personnes, dates ; résoudre « il », « elle », « ça », « le », « lui », « alors ».",
           "Intent, requested slot, people, dates; resolve “it”, “he”, “she”, “that”, “him”, “then”.")),
        (X("Entités et événements", "Entities and events"),
         X("Index séparés : personnes, projets, objets, décisions, actions, événements. « Le payer lui » retrouve le "
           "paiement et le prestataire.",
           "Separate indexes: people, projects, objects, decisions, actions, events. “Pay him” finds the payment and "
           "the supplier.")),
        (X("BM25 + dense", "BM25 + dense"),
         X("Mots exacts pour noms, dates, chiffres ; sens pour les reformulations ; fusion RRF.",
           "Exact words for names, dates, numbers; meaning for paraphrases; RRF fusion.")),
        (X("Expansion de graphe", "Graph expansion"),
         X("Parcourir <font name='Body-B'>same_event</font>, <font name='Body-B'>answers</font>, "
           "<font name='Body-B'>creates_action</font>, <font name='Body-B'>supports</font>, "
           "<font name='Body-B'>supersedes</font> pour retrouver les continuations distantes.",
           "Traverse <font name='Body-B'>same_event</font>, <font name='Body-B'>answers</font>, "
           "<font name='Body-B'>creates_action</font>, <font name='Body-B'>supports</font>, "
           "<font name='Body-B'>supersedes</font> to recover distant continuations.")),
        (X("Reranking contextuel", "Contextual reranking"),
         X("Qwen3-Reranker-0.6B relit la question avec chaque candidat ; le classement du reranker est fusionné "
           "avec celui du pipeline, pas substitué.",
           "Qwen3-Reranker-0.6B rereads the question with each candidate; its ranking is fused with the pipeline's, "
           "not substituted for it.")),
    ]
    for t, d in layers:
        S(B("<b>%s</b> — %s" % (t, d)))
    gap(3)
    S(P(X("<b>Le reranker, mesuré sur 26 questions répondables</b>",
          "<b>The reranker, measured on 26 answerable questions</b>"), cell_b))
    rr = [
        [Paragraph(fr(x), cell_h) for x in [X("Ordre", "Ordering"), "hit@1", "hit@3", "hit@5"]],
        [Paragraph(fr(X("ordre du pipeline", "pipeline order")), cell_b), Paragraph(".346", cell), Paragraph(".808", cell), Paragraph(".846", cell)],
        [Paragraph(fr(X("reranker, consigne par défaut", "reranker, default instruction")), cell_b), Paragraph(".500", cell), Paragraph(".692", cell), Paragraph(".846", cell)],
        [Paragraph(fr(X("reranker, consigne française", "reranker, French instruction")), cell_b), Paragraph(".423", cell), Paragraph(".885", cell), Paragraph(".885", cell)],
        [Paragraph(fr(X("fusion + consigne française", "fusion + French instruction")), cell_b), Paragraph("<b>.577</b>", cell), Paragraph("<b>.846</b>", cell), Paragraph(".885", cell)],
    ]
    trr = Table(rr, colWidths=[CW-72*mm, 24*mm, 24*mm, 24*mm])
    srr = [("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
           ("TOPPADDING",(0,0),(-1,-1),3.4),("BOTTOMPADDING",(0,0),(-1,-1),3.4),
           ("LEFTPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,1),(-1,-1),0.5,LINE),
           ("BOX",(0,0),(-1,-1),0.7,LINE),("ROUNDEDCORNERS",[5,5,5,5]),
           ("BACKGROUND",(0,4),(-1,4),TEAL_L),("ALIGN",(1,0),(-1,-1),"CENTER")]
    trr.setStyle(TableStyle(srr))
    S(trr)
    gap(2)
    S(P(X("Le reranking seul gagne le top 1 et perd le top 3 ; la fusion garde les deux. Aucun fine-tune : "
          "26 questions suffiraient à le faire gagner sur son propre jeu et à le perdre ailleurs.",
          "Pure reranking wins top 1 and loses top 3; fusion keeps both. No fine-tune: 26 questions would be enough "
          "to win its own set and lose elsewhere."), body_s))
    gap(3)
    S(P(X("<b>Le choix de l'encodeur, mesuré</b>", "<b>The encoder choice, measured</b>"), cell_b))
    emb_rows = [
        [Paragraph(fr(x), cell_h) for x in [X("Encodeur", "Encoder"), X("Ce qu'on a mesuré", "What we measured"), X("Décision", "Decision")]],
        [Paragraph("e5-base", cell_b), Paragraph(fr(X("la référence sur les 36 questions", "the reference on all 36 questions")), cell), Paragraph(fr(X("livré", "shipped")), cell)],
        [Paragraph("jina v3", cell_b), Paragraph(fr(X("même niveau, parfois devant", "same level, sometimes ahead")), cell), Paragraph(fr(X("rien à gagner", "nothing to gain")), cell)],
        [Paragraph("jina v5", cell_b), Paragraph(fr(X("même niveau que v3", "same level as v3")), cell), Paragraph(fr(X("rien à gagner", "nothing to gain")), cell)],
        [Paragraph(fr(X("late chunking", "late chunking")), cell_b), Paragraph(fr(X("gain net si la requête est contextualisée", "clear gain when the query is contextual")), cell), Paragraph(fr(X("gardé derrière --long-split", "kept behind --long-split")), cell)],
    ]
    temb = Table(emb_rows, colWidths=[34*mm, 78*mm, CW-112*mm], repeatRows=1)
    semb = [("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ("LEFTPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,1),(-1,-1),0.5,LINE),
            ("BOX",(0,0),(-1,-1),0.7,LINE),("ROUNDEDCORNERS",[5,5,5,5]),
            ("BACKGROUND",(0,2),(-1,2),BG),("BACKGROUND",(0,4),(-1,4),BG)]
    temb.setStyle(TableStyle(semb))
    S(temb)
    gap(4)
    S(Box([title_p(AMBER_D, X("Ce que la recherche ne récupère pas encore", "What retrieval still misses")), 3,
           B(X("Cinq questions répondables dont le point n'apparaît dans aucun tour : « un maintenant, un au troisième "
               "trimestre », « les 3 000 de différence », « la cause de l'incident », deux réponses anaphoriques « c'est moi ».",
               "Five answerable questions whose point appears in no turn: “one now, one in Q3”, “the 3,000 gap”, "
               "“the cause of the incident”, two anaphoric “it's me” answers.")),
           B(X("Les embeddings ne changent rien de mesurable : jina v3 et v5 font jeu égal avec e5 ; e5-base reste livré "
               "pour la reproductibilité. Le gain venait du late chunking, pas du modèle.",
               "Embeddings change nothing measurable: jina v3 and v5 tie with e5; e5-base ships for reproducibility. "
               "The gain came from late chunking, not the model.")),
           B(X("Le late chunking n'aide que si la requête est encodée avec son contexte : des vecteurs calculés à "
               "l'index avec une requête hors contexte inversent le signal (9/40).",
               "Late chunking only helps when the query is encoded with its context: index-time vectors with a "
               "context-free query invert the signal (9/40)."))],
          width=CW, bg=AMBER_L, stroke=AMBER, pad=9))
    S(PageBreak())

    # ================= PAGE 6 — linker + memory =================
    S(SectionHeader(5, X("4 · Linker et 5 · Memory", "4 · Linker and 5 · Memory"), TEAL,
                    X("Décider ce qui remplace quoi, sans jamais réécrire",
                      "Decide what replaces what, without ever rewriting")))
    left = Box([title_p(TEAL_D, X("La règle du linker", "The linker's rule")), 3,
                B(X("même <b>projet</b>, même <b>slot</b>, même <b>point</b>",
                    "same <b>project</b>, same <b>slot</b>, same <b>point</b>")),
                B(X("valeur <b>différente</b> et <b>tour nouveau</b>, déjà posé (pas une question, pas une hésitation)",
                    "a <b>different value</b> in a <b>new, settled</b> turn (not a question, not a hesitation)")),
                B(X("le lien pointe vers le tour le plus récent qui portait l'ancienne valeur",
                    "the link points at the latest turn carrying the old value")),
                B(X("rien n'est réécrit : « remplacé » est un lien, pas une modification",
                    "nothing is rewritten: “replaced” is a link, not an edit"))],
               width=half, bg=TEAL_L, stroke=TEAL, pad=10)
    right = Box([title_p(PURPLE_D, X("La clé d'état", "The state key")), 3,
                 P(X("<font name='Body-B'>(projet, slot, point)</font> — le point distingue « clés », « fibre », "
                     "« bail », « salaire », « acompte ».", 
                     "<font name='Body-B'>(project, slot, point)</font> — the point separates “keys”, “fibre”, "
                     "“lease”, “salary”, “deposit”."), body_s), 3,
                 P(X("Une proposition reste <b>ouverte</b> ; confirmé / annulé / réalisé met à jour la vue courante ; "
                     "l'ancien reste dans l'historique.",
                     "A proposal stays <b>open</b>; confirmed / cancelled / completed updates the current view; "
                     "the old value stays in history."), body_s), 3,
                 P(X("Une unité ne reçoit un statut que si elle est la preuve de cet état : 76 fuites sur 103 unités "
                     "avant la règle, <b>0</b> après.",
                     "A unit only gets a status if it is that state's evidence: 76 leaks out of 103 units before the "
                     "rule, <b>0</b> after."), body_s)],
                width=half, bg=PURPLE_L, stroke=PURPLE, pad=10)
    S(row_of([left, right], half))
    gap(4)
    S(P(X("<b>Cinq clés réelles, telles que la mémoire les écrit</b>",
          "<b>Five real keys, exactly as memory writes them</b>"), cell_b))
    keys = [
        [Paragraph(fr(x), cell_h) for x in [X("Projet", "Project"), "slot", "point", X("Valeur", "Value"), X("Statut", "Status")]],
        [Paragraph("nova", cell_b), Paragraph("date", cell), Paragraph("lancement", cell), Paragraph("5 avril", cell), Paragraph(fr(X("confirmé", "confirmed")), cell)],
        [Paragraph("hiring", cell_b), Paragraph("budget", cell), Paragraph("salaire", cell), Paragraph("52 000", cell), Paragraph(fr(X("confirmé", "confirmed")), cell)],
        [Paragraph("traiteur", cell_b), Paragraph("budget", cell), Paragraph("acompte", cell), Paragraph("1 350", cell), Paragraph(fr(X("proposé", "proposed")), cell)],
        [Paragraph("demenagement", cell_b), Paragraph("date", cell), Paragraph("clés", cell), Paragraph("1er septembre", cell), Paragraph(fr(X("confirmé", "confirmed")), cell)],
        [Paragraph("incident", cell_b), Paragraph("budget", cell), Paragraph("utilisateurs", cell), Paragraph("12 000", cell), Paragraph(fr(X("confirmé", "confirmed")), cell)],
    ]
    tk = Table(keys, colWidths=[34*mm, 26*mm, 34*mm, 40*mm, CW-134*mm], repeatRows=1)
    sk = [("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
          ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
          ("LEFTPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,1),(-1,-1),0.5,LINE),
          ("BOX",(0,0),(-1,-1),0.7,LINE),("ROUNDEDCORNERS",[5,5,5,5])]
    for i in (2, 4):
        sk.append(("BACKGROUND",(0,i),(-1,i),BG))
    tk.setStyle(TableStyle(sk))
    S(tk)
    gap(4)
    S(P(X("<b>Qui devient « courant » ?</b>", "<b>What becomes “current”?</b>"), cell_b))
    cur_rows = [
        [Paragraph(fr(x), cell_h) for x in [X("Unité", "Unit"), X("Statut", "Status"), X("Courant ?", "Current?"), X("Pourquoi", "Why")]],
        [Paragraph(fr(X("« on pourrait remonter à 1 500 »", "“we could raise it to 1,500”")), cell_b), Paragraph(fr(X("proposé", "proposed")), cell), Paragraph(fr(X("non", "no")), cell), Paragraph(fr(X("une proposition ne remplace pas une décision", "a proposal never replaces a decision")), cell)],
        [Paragraph(fr(X("« on valide 800 »", "“we approve 800”")), cell_b), Paragraph(fr(X("confirmé", "confirmed")), cell), Paragraph(fr(X("oui", "yes")), cell), Paragraph(fr(X("valeur confirmée, datée, avec preuve", "confirmed, dated, with proof")), cell)],
        [Paragraph(fr(X("« finalement on annule »", "“actually we cancel”")), cell_b), Paragraph(fr(X("annulé", "cancelled")), cell), Paragraph(fr(X("oui (annulé)", "yes (cancelled)")), cell), Paragraph(fr(X("l'annulation est un état, pas un silence", "a cancellation is a state, not a silence")), cell)],
        [Paragraph(fr(X("« il a dit qu'il verrait »", "“he said he'd see”")), cell_b), Paragraph(fr(X("inconnu", "unknown")), cell), Paragraph(fr(X("non", "no")), cell), Paragraph(fr(X("aucun engagement pris", "no commitment made")), cell)],
    ]
    tcur = Table(cur_rows, colWidths=[58*mm, 24*mm, 26*mm, CW-108*mm], repeatRows=1)
    scur = [("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ("LEFTPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,1),(-1,-1),0.5,LINE),
            ("BOX",(0,0),(-1,-1),0.7,LINE),("ROUNDEDCORNERS",[5,5,5,5]),
            ("BACKGROUND",(0,2),(-1,2),BG),("BACKGROUND",(0,4),(-1,4),BG)]
    tcur.setStyle(TableStyle(scur))
    S(tcur)
    gap(4)
    S(Box([title_p(TEAL_D, X("Un état, trois messages", "One state, three messages")), 3,
           B(X("<b>1er octobre</b> — « le budget est proposé à 1 200 € » : proposition, état ouvert.",
               "<b>October 1</b> — “the budget is proposed at €1,200”: proposal, open state.")),
           B(X("<b>8 octobre</b> — « on valide 800 € » : confirmation ; le courant devient 800 €, 1 200 € reste dans l'historique.",
               "<b>October 8</b> — “we approve €800”: confirmation; current becomes €800, €1,200 stays in history.")),
           B(X("<b>15 octobre</b> — « finalement on annule » : annulation ; le courant devient annulé, les deux valeurs restent visibles.",
               "<b>October 15</b> — “actually we cancel”: cancellation; current becomes cancelled, both values remain visible.")),
           B(X("<b>« On pourrait remonter à 1 500 € » ne remplace pas un 800 € confirmé.</b> Il reste ouvert.",
               "<b>“We could raise it to €1,500” does not replace a confirmed €800.</b> It stays open."))],
          width=CW, bg=TEAL_L, stroke=TEAL, pad=9))
    gap(3)
    now = Box([title_p(TEAL_D, X("Tout de suite", "Right away")), 3,
               B(X("découper, tagger, indexer le message", "split, tag, index the message")),
               B(X("chercher les candidats dans l'historique", "retrieve candidates from history")),
               B(X("ajouter liens et état, sans réécrire", "append links and state, without rewriting"))],
              width=half, bg=TEAL_L, stroke=TEAL, pad=9)
    later = Box([title_p(PURPLE_D, X("En arrière-plan", "In the background")), 3,
                 B(X("recalculer les index d'entités et d'événements", "rebuild entity/event indexes")),
                 B(X("auditer le reranker et les seuils", "audit the reranker and thresholds")),
                 B(X("proposer un ré-entraînement sur données corrigées", "propose retraining on corrected data"))],
                width=half, bg=PURPLE_L, stroke=PURPLE, pad=9)
    S(row_of([now, later], half))
    S(PageBreak())

    # ================= PAGE 7 — résultats + entraînement =================
    S(SectionHeader(6, X("Les résultats, étape par étape", "Results, step by step"), PURPLE,
                    X("5 réunions écrites à la main · 103 tours · 36 questions",
                      "5 hand-written meetings · 103 turns · 36 questions")))
    res = [
        [Paragraph(fr(x), cell_h) for x in [X("Étape", "Step"), X("Mesure", "Measure"), X("Résultat", "Result")]],
        [Paragraph(fr(X("1 · store", "1 · store")), cell_b), Paragraph(fr(X("édition refusée ; ré-ingestion sans effet", "edit refused; re-ingest no-op")), cell), Paragraph(fr(X("refusée ; 0 doublon", "refused; 0 duplicates")), cell)],
        [Paragraph(fr(X("2 · chunker", "2 · chunker")), cell_b), Paragraph(fr(X("<font name='Body-B'>text[start:end] == span.text</font>", "<font name='Body-B'>text[start:end] == span.text</font>")), cell), Paragraph(fr(X("0 écart sur 103 tours", "0 mismatch on 103 turns")), cell)],
        [Paragraph(fr(X("3 · tagger", "3 · tagger")), cell_b), Paragraph(fr(X("rappel des changements ; valeurs qui sont des valeurs", "change recall; values that are values")), cell), Paragraph(fr(X("1,00 ; 32 sur 32", "1.00; 32 of 32")), cell)],
        [Paragraph(fr(X("4 · linker", "4 · linker")), cell_b), Paragraph(fr(X("tour remplacé relié", "replaced turn linked")), cell), Paragraph(fr(X("4 sur 4", "4 of 4")), cell)],
        [Paragraph(fr(X("5 · memory", "5 · memory")), cell_b), Paragraph(fr(X("valeur courante ; proposition ouverte ; fuites", "current value; open proposal; leaks")), cell), Paragraph(fr(X("12 sur 12 ; 1 sur 1 ; 0", "12 of 12; 1 of 1; 0")), cell)],
        [Paragraph(fr(X("6 · retriever", "6 · retriever")), cell_b), Paragraph(fr(X("hit@1 / hit@3 / hit@10", "hit@1 / hit@3 / hit@10")), cell), Paragraph(fr(X(".577 / .846 / .962", ".577 / .846 / .962")), cell)],
        [Paragraph(fr(X("7 · answerer", "7 · answerer")), cell_b), Paragraph(fr(X("cite la bonne phrase ; pièges refusés", "cites the right sentence; tricky refused")), cell), Paragraph(fr(X(".808 ; 7 sur 10", ".808; 7 of 10")), cell)],
    ]
    tr = Table(res, colWidths=[30*mm, 62*mm, CW-92*mm])
    str_ = [("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),3.6),("BOTTOMPADDING",(0,0),(-1,-1),3.6),
            ("LEFTPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,1),(-1,-1),0.5,LINE),
            ("BOX",(0,0),(-1,-1),0.7,LINE),("ROUNDEDCORNERS",[5,5,5,5])]
    for i in (2, 4, 6):
        str_.append(("BACKGROUND",(0,i),(-1,i),BG))
    tr.setStyle(TableStyle(str_))
    S(tr)
    gap(3)
    S(P(X("<b>Bout en bout : 0,778</b> — formel 0,833, désordonné 0,750, piège 0,700. Même jeu, trois rédacteurs :",
          "<b>End to end: 0.778</b> — formal 0.833, messy 0.750, tricky 0.700. Same set, three answerers:"), cell_b))
    ans_rows = [
        [Paragraph(fr(x), cell_h) for x in [X("Rédacteur", "Answerer"), X("Réponses justes", "Correct"), X("Cite la bonne phrase", "Cites right turn"), X("Pièges refusés", "Tricky refused")]],
        [Paragraph(fr(X("gabarit déterministe", "deterministic template")), cell_b), Paragraph(".694", cell), Paragraph(".731", cell), Paragraph("9/10", cell)],
        [Paragraph(fr(X("LLM seul (7B)", "LLM alone (7B)")), cell_b), Paragraph(".583", cell), Paragraph(".577", cell), Paragraph("7/10", cell)],
        [Paragraph(fr(X("hybride (livré)", "hybrid (shipped)")), cell_b), Paragraph("<b>.778</b>", cell), Paragraph("<b>.808</b>", cell), Paragraph("7/10", cell)],
        [Paragraph(fr(X("LLM 4B", "4B LLM")), cell_b), Paragraph(".278", cell), Paragraph("0", cell), Paragraph("10/10", cell)],
    ]
    ta = Table(ans_rows, colWidths=[CW-108*mm, 36*mm, 36*mm, 36*mm], repeatRows=1)
    sa = [("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
          ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
          ("LEFTPADDING",(0,0),(-1,-1),6),("LINEBELOW",(0,1),(-1,-1),0.5,LINE),
          ("BOX",(0,0),(-1,-1),0.7,LINE),("ROUNDEDCORNERS",[5,5,5,5]),
          ("BACKGROUND",(0,3),(-1,3),PURPLE_L),("ALIGN",(1,0),(-1,-1),"CENTER")]
    ta.setStyle(TableStyle(sa))
    S(ta)
    gap(2)
    S(P(X("Le 4B refuse tout : sur 8 Go, le 7B est le plancher utilisable. La mémoire répond quand elle a la preuve ; "
          "le LLM ne prend que le reste.",
          "The 4B refuses everything: on 8 GB, the 7B is the usable floor. Memory answers when it has the proof; "
          "the LLM only takes the rest."), body_s))
    gap(4)
    S(SectionHeader(7, X("Ce qu'on entraîne", "What we train"), PURPLE,
                    X("Chaque modèle a un rôle et une mesure ; deux étapes n'ont aucun entraînement",
                      "Each model has a role and a measure; two steps have no training at all")))
    train_rows = [
        [Paragraph(fr(x), cell_h) for x in [X("Étape", "Step"), X("Modèle", "Model"), X("Données", "Data"), X("Mesure", "Measure")]],
        [Paragraph(fr(X("3 · tagger", "3 · tagger")), cell_b), Paragraph("Laya multilingue, fine-tuné", cell),
         Paragraph(fr(X("12 500 / 2 000 / 4 200 exemples synthétiques, familles disjointes", "12,500 / 2,000 / 4,200 synthetic examples, family-disjoint")), cell),
         Paragraph(fr(X("rappel 1,00 ; 32/32", "recall 1.00; 32/32")), cell)],
        [Paragraph(fr(X("6 · retriever (sens)", "6 · retriever (meaning)")), cell_b), Paragraph("e5-base gelé", cell),
         Paragraph(fr(X("paires réelles + synthétiques", "real + synthetic pairs")), cell), Paragraph(fr(X("hit@10 .962", "hit@10 .962")), cell)],
        [Paragraph(fr(X("6 · retriever (rerank)", "6 · retriever (rerank)")), cell_b), Paragraph("Qwen3-Reranker-0.6B, zéro-shot", cell),
         Paragraph(fr(X("aucune", "none")), cell), Paragraph(fr(X("+.231 hit@1 par fusion", "+.231 hit@1 by fusion")), cell)],
        [Paragraph(fr(X("4 · linker", "4 · linker")), cell_b), Paragraph(fr(X("règle déterministe", "deterministic rule")), cell),
         Paragraph(fr(X("aucune", "none")), cell), Paragraph(fr(X("4/4", "4/4")), cell)],
        [Paragraph(fr(X("5 · memory", "5 · memory")), cell_b), Paragraph(fr(X("règle append-only", "append-only rule")), cell),
         Paragraph(fr(X("aucune", "none")), cell), Paragraph(fr(X("12/12 ; 0 fuite", "12/12; 0 leak")), cell)],
        [Paragraph(fr(X("7 · answerer", "7 · answerer")), cell_b), Paragraph("Qwen3.5-4B + vérificateur", cell),
         Paragraph(fr(X("aucune", "none")), cell), Paragraph(fr(X("0,778 hybride", "0.778 hybrid")), cell)],
    ]
    tt = Table(train_rows, colWidths=[32*mm, 48*mm, 58*mm, CW-138*mm], repeatRows=1)
    stt = [("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"TOP"),
           ("TOPPADDING",(0,0),(-1,-1),3.2),("BOTTOMPADDING",(0,0),(-1,-1),3.2),
           ("LEFTPADDING",(0,0),(-1,-1),5),("LINEBELOW",(0,1),(-1,-1),0.5,LINE),
           ("BOX",(0,0),(-1,-1),0.7,LINE),("ROUNDEDCORNERS",[5,5,5,5])]
    for i in (2, 4, 6):
        stt.append(("BACKGROUND",(0,i),(-1,i),BG))
    tt.setStyle(TableStyle(stt))
    S(tt)
    gap(2)
    S(P(X("Le fine-tune du reranker a été écarté : 26 questions suffisent à le faire gagner sur son propre jeu et "
          "à le perdre ailleurs. Le tagger, lui, a 18 700 exemples et gagne pour de vrai.",
          "Reranker fine-tuning was ruled out: 26 questions are enough to win its own set and lose elsewhere. "
          "The tagger has 18,700 examples and genuinely wins."), body_s))
    S(PageBreak())

    # ================= PAGE 8 — limites + lexique =================
    S(SectionHeader(8, X("Limites mesurées", "Measured limits"), CORAL,
                    X("Ce que les chiffres disent, sans les arrondir",
                      "What the numbers say, unrounded")))
    S(B(X("<b>Synthétique uniquement.</b> Cinq réunions, 36 questions, un or écrit par l'auteur des réunions. "
          "Aucune réunion tenue à l'écart : c'est un point de départ solide, pas un résultat de concours.",
          "<b>Synthetic only.</b> Five meetings, 36 questions, gold written by the author of the meetings. "
          "No held-out meeting: a solid baseline, not a leaderboard result.")))
    S(B(X("<b>Les règles sont ajustées à ces cinq réunions</b> : statut, point, valeur. La prochaine preuve honnête "
          "est une réunion que les règles n'ont jamais vue.",
          "<b>The rules are fitted to these five meetings</b>: status, point, value. The next honest test is a "
          "meeting the rules have never seen.")))
    S(B(X("<b>hit@1 = .577.</b> La bonne phrase est dans le top 3 pour 85 % des questions, dans le top 10 pour 96 % — "
          "mais pas toujours première. Le reranker gagne le top 1 et le perd parfois au top 3 ; la fusion limite la casse.",
          "<b>hit@1 = .577.</b> The right sentence is in the top 3 for 85% of questions, in the top 10 for 96% — "
          "but not always first. The reranker wins top 1 and sometimes loses top 3; fusion limits the damage.")))
    S(B(X("<b>3 questions pièges sur 10 reçoivent encore une réponse.</b> Le 7B conclut d'une intention qu'une action "
          "a eu lieu (« on va lui envoyer l'offre » → « il a accepté »). Un modèle plus grand est la variable manquante.",
          "<b>3 of 10 tricky questions still get an answer.</b> The 7B concludes from an intention that an action "
          "happened (“we'll send the offer” → “he accepted”). A larger model is the missing variable.")))
    S(B(X("<b>Deux clés pour un même point.</b> Un tour sans point écrit <font name='Body-B'>projet:slot</font> à côté "
          "de <font name='Body-B'>projet:slot:point</font>. Le linker devrait hériter du point quand il est unique.",
          "<b>Two keys for one point.</b> A turn without a point writes <font name='Body-B'>project:slot</font> next to "
          "<font name='Body-B'>project:slot:point</font>. The linker should inherit the point when it is unambiguous.")))
    gap(4)
    S(Box([title_p(CORAL, X("Ce qui changerait le verdict", "What would change the verdict")), 3,
           B(X("une réunion tenue à l'écart, que ni les règles ni les seuils n'ont vue",
               "a held-out meeting that neither the rules nor the thresholds have seen")),
           B(X("un answerer plus grand (14B ou hébergé) : les trois pièges restants sont une limite de modèle",
               "a larger answerer (14B or hosted): the three remaining tricky answers are a model limit")),
           B(X("hériter du point quand il est unique, pour supprimer la double clé",
               "inherit the point when it is unambiguous, to remove the double key")),
           B(X("vingt réunions de plus pour séparer une règle qui marche d'une règle qui a de la chance",
               "twenty more meetings to separate a rule that works from a rule that got lucky"))],
          width=CW, bg=HexColor("#FDEDEA"), stroke=CORAL, pad=9))
    gap(4)
    S(SectionHeader(9, X("Petit lexique", "Short glossary"), PURPLE,
                    X("Les mots techniques, en clair", "Technical words in plain language")))
    glos = [
        (X("store", "store"), X("Le message d'origine, tel quel : texte, auteur, date, identifiant. Jamais modifié.",
                                "The original message, as-is: text, speaker, date, id. Never modified.")),
        (X("chunker", "chunker"), X("Découpe le message en phrases et unités ; chaque phrase garde ses offsets exacts.",
                                    "Splits the message into sentences and units; every sentence keeps exact offsets.")),
        (X("tagger", "tagger"), X("Lit chaque unité : nature, statut, point, valeur. C'est Laya, fine-tuné.",
                                  "Reads each unit: kind, status, point, value. This is fine-tuned Laya.")),
        (X("linker", "linker"), X("Règle qui dit quel tour remplace quel autre. Elle ajoute un lien, elle ne modifie rien.",
                                  "Rule that says which turn replaces which. It appends a link, it changes nothing.")),
        (X("memory", "memory"), X("L'état courant est une vue sur des événements ajoutés ; l'historique reste intact.",
                                  "Current state is a view over appended events; history stays intact.")),
        (X("retriever", "retriever"), X("Mots exacts + sens + graphe, puis reranker : il trouve les tours candidats.",
                                        "Exact words + meaning + graph, then a reranker: it finds candidate turns.")),
        (X("answerer", "answerer"), X("Question + tours retrouvés + état → LLM. La réponse cite ou refuse ; le vérificateur "
                                      "rejette tout chiffre absent des preuves.",
                                      "Question + retrieved turns + state → LLM. The answer cites or refuses; the verifier "
                                      "rejects any number absent from the proof.")),
        (X("abstention", "abstention"), X("Sans preuve suffisante, le moteur dit « je ne sais pas » au lieu d'inventer.",
                                          "Without enough evidence, the engine says “I don't know” instead of inventing.")),
    ]
    gd = [[Paragraph(fr("<b>%s</b>" % a), cell_b), Paragraph(fr(b), cell)] for a,b in glos]
    tg = Table(gd, colWidths=[30*mm, CW-30*mm])
    tg.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),("LEFTPADDING",(0,0),(-1,-1),6),
        ("LINEBELOW",(0,0),(-1,-2),0.5,LINE),("BOX",(0,0),(-1,-1),0.7,LINE),
        ("ROUNDEDCORNERS",[5,5,5,5]),("BACKGROUND",(0,0),(0,-1),BG)]))
    S(tg)
    gap(2)
    S(P(X("Tout se régénère : <font name='Body-B'>make meetings</font> (bout en bout), "
          "<font name='Body-B'>make steps</font> (tableau ci-dessus), <font name='Body-B'>make ask4</font> "
          "(une question, à la main).",
          "Everything regenerates: <font name='Body-B'>make meetings</font> (end to end), "
          "<font name='Body-B'>make steps</font> (the table above), <font name='Body-B'>make ask4</font> "
          "(one question, by hand)."), cap))

    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=LM, rightMargin=LM, topMargin=18 * mm, bottomMargin=19 * mm,
                          title="Semantic Agent Engine - Architecture v4", author="Jonathan Chayna",
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
    ap.add_argument("--variant", choices=["v3", "v4", "both"], default="v4",
                    help="v4 is the new architecture; v3 is kept for rollback")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--fallback-fonts", action="store_true", help="use built-in Helvetica, skip font lookup/download")
    ap.add_argument("--no-download", action="store_true", help="never download fonts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_fonts(force_fallback=args.fallback_fonts, allow_download=not args.no_download)
    langs = ["fr", "en"] if args.lang == "both" else [args.lang]
    variants = ["v3", "v4"] if args.variant == "both" else [args.variant]
    for variant in variants:
        for lang in langs:
            if variant == "v4":
                name = "semantic_agent_engine_architecture_v4_%s.pdf" % lang.upper()
                out = os.path.join(args.out_dir, name)
                build_v4(lang, out)
                root = os.path.join(HERE, name)
                shutil.copyfile(out, root)
            else:
                name = "semantic_agent_engine_architecture_%s.pdf" % lang.upper()
                build(lang, os.path.join(args.out_dir, name))


if __name__ == "__main__":
    main()
