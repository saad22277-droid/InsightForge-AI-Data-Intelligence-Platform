"""Convert reports/Technical_Report.md into a styled PDF (reports/Technical_Report.pdf).

Supports the Markdown subset used in the report: headings, paragraphs, bullet/numbered lists,
pipe tables, code fences, images, **bold**, `code`.  Usage: python scripts/build_technical_report.py
"""
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "Technical_Report.md"
OUT = ROOT / "reports" / "Technical_Report.pdf"
ACCENT, INK, MUTED, LINE = colors.HexColor("#4338ca"), colors.HexColor("#1f2937"), colors.HexColor("#6b7280"), colors.HexColor("#d1d5db")

base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=base["Title"], fontSize=24, leading=29, textColor=INK, alignment=0, spaceAfter=10),
    "h2": ParagraphStyle("h2", parent=base["Heading1"], fontSize=15, leading=19, textColor=ACCENT, spaceBefore=14, spaceAfter=6, keepWithNext=1),
    "h3": ParagraphStyle("h3", parent=base["Heading2"], fontSize=11.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3, keepWithNext=1),
    "p": ParagraphStyle("p", parent=base["BodyText"], fontSize=9.6, leading=14, textColor=INK, spaceAfter=5),
    "li": ParagraphStyle("li", parent=base["BodyText"], fontSize=9.6, leading=13.5, textColor=INK, leftIndent=14, bulletIndent=3, spaceAfter=2),
    "cell": ParagraphStyle("c", parent=base["BodyText"], fontSize=8.2, leading=10.4, spaceAfter=0),
    "cellh": ParagraphStyle("ch", parent=base["BodyText"], fontSize=8.2, leading=10.4, fontName="Helvetica-Bold", textColor=colors.white),
    "code": ParagraphStyle("code", parent=base["Code"], fontSize=7.6, leading=9.5, backColor=colors.HexColor("#f3f4f6"), borderPadding=5),
}


def inline(text: str) -> str:
    text = text.encode("cp1252", "replace").decode("cp1252")
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8.5" color="#4338ca">\1</font>', text)
    return text


def image(path: str, alt: str, max_h=15 * cm):
    p = (SRC.parent / path).resolve()
    reader = ImageReader(str(p))
    w, h = reader.getSize()
    width = 16 * cm
    height = width * h / w
    if height > max_h:
        height, width = max_h, max_h * w / h
    return Image(str(p), width=width, height=height)


def table(rows: list[list[str]]):
    header, body = rows[0], rows[1:]
    ncols = len(header)
    data = [[Paragraph(inline(c), S["cellh"]) for c in header]] + [
        [Paragraph(inline(c), S["cell"]) for c in (r + [""] * ncols)[:ncols]] for r in body]
    first = 5.2 * cm if ncols > 2 else 6.5 * cm
    widths = [first] + [(17 * cm - first) / (ncols - 1)] * (ncols - 1)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def build() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    story, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            story += [Spacer(1, 2 * cm), Paragraph(inline(line[2:]), S["title"])]
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), S["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), S["h3"]))
        elif line.startswith("```"):
            block = []
            i += 1
            while not lines[i].startswith("```"):
                block.append(lines[i]); i += 1
            story.append(Preformatted("\n".join(block), S["code"]))
        elif line.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            story += [image(m.group(2), m.group(1)), Spacer(1, 6)]
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            story += [table(rows), Spacer(1, 8)]
            continue
        elif re.match(r"^\s*[-*] ", line):
            story.append(Paragraph(inline(re.sub(r"^\s*[-*] ", "", line)), S["li"], bulletText="\u2022"))
        elif re.match(r"^\d+\. ", line):
            n = re.match(r"^(\d+)\. ", line).group(1)
            story.append(Paragraph(inline(re.sub(r"^\d+\. ", "", line)), S["li"], bulletText=f"{n}."))
        elif line.strip():
            para = [line.strip()]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||!\[|```|\s*[-*] |\d+\. )", lines[i + 1]):
                i += 1; para.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(para)), S["p"]))
        i += 1

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.2 * cm, "AI Data Intelligence Platform - Technical Report")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}"); canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title="AI Data Intelligence Platform - Technical Report",
                            author="AI Data Intelligence Platform")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(build())
