"""Professional PDF analytical report built from real, computed results."""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = colors.HexColor("#4338ca")
INK = colors.HexColor("#1f2937")
MUTED = colors.HexColor("#6b7280")
LINE = colors.HexColor("#d1d5db")
PLOT_COLOR = "#4f46e5"
PLOT_ALT = "#f59e0b"


def _clean(text) -> str:
    """Escape for reportlab and drop characters Helvetica cannot render."""
    text = str(text if text is not None else "")
    text = text.encode("cp1252", "replace").decode("cp1252")
    return escape(text)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:.{digits}g}"
    return str(value)


class _Styles:
    def __init__(self):
        base = getSampleStyleSheet()
        self.title = ParagraphStyle("t", parent=base["Title"], fontSize=24, leading=28,
                                    textColor=INK, alignment=TA_LEFT, spaceAfter=6)
        self.subtitle = ParagraphStyle("st", parent=base["Normal"], fontSize=11,
                                       textColor=MUTED, spaceAfter=4)
        self.h1 = ParagraphStyle("h1", parent=base["Heading1"], fontSize=15, leading=18,
                                 textColor=ACCENT, spaceBefore=14, spaceAfter=6, keepWithNext=1)
        self.h2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, leading=14,
                                 textColor=INK, spaceBefore=8, spaceAfter=3, keepWithNext=1)
        self.body = ParagraphStyle("b", parent=base["BodyText"], fontSize=9.5, leading=13.5,
                                   textColor=INK, spaceAfter=4)
        self.bullet = ParagraphStyle("bl", parent=self.body, leftIndent=12, bulletIndent=2)
        self.small = ParagraphStyle("s", parent=self.body, fontSize=8, leading=10.5,
                                    textColor=MUTED)
        self.cell = ParagraphStyle("c", parent=self.body, fontSize=8, leading=10, spaceAfter=0)
        self.cellb = ParagraphStyle("cb", parent=self.cell, fontName="Helvetica-Bold",
                                    textColor=colors.white)
        self.note = ParagraphStyle("n", parent=self.body, fontSize=8.5, textColor=MUTED,
                                   backColor=colors.HexColor("#f3f4f6"), borderPadding=5,
                                   spaceBefore=4, spaceAfter=8)


def _table(S: _Styles, header: list[str], rows: list[list], widths: list[float] | None = None):
    data = [[Paragraph(_clean(h), S.cellb) for h in header]]
    for row in rows:
        data.append([Paragraph(_clean(c), S.cell) for c in row])
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _md(S: _Styles, text: str) -> list:
    """Render the small markdown subset Gemini produces (bold, bullets, paragraphs)."""
    flow = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _clean(line))
        html = re.sub(r"(?<!\*)\*(?!\*|\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", html)
        if re.match(r"^(\*|-|\u2022)\s+", line):
            html = re.sub(r"^(\*|-|\u2022)\s+", "", html)
            flow.append(Paragraph(html, S.bullet, bulletText="\u2022"))
        else:
            html = re.sub(r"^#+\s*", "", html)
            flow.append(Paragraph(html, S.body))
    return flow


# ----------------------------------------------------------------------------- charts
def _png(fig) -> io.BytesIO:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.title.set_fontsize(9)


def chart_missing(eda: dict):
    items = [c for c in eda["missing"]["columns"] if c["missing"] > 0][:10]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh([c["column"][:25] for c in items][::-1], [c["percentage"] for c in items][::-1],
            color=PLOT_ALT)
    ax.set_xlabel("% missing", fontsize=7)
    ax.set_title("Missing values by column")
    _style_axes(ax)
    return _png(fig)


def chart_histograms(df: pd.DataFrame, columns: list[str]):
    columns = columns[:3]
    if not columns:
        return None
    fig, axes = plt.subplots(1, len(columns), figsize=(6.4, 2.3))
    axes = np.atleast_1d(axes)
    for ax, column in zip(axes, columns):
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        ax.hist(values, bins=15, color=PLOT_COLOR, edgecolor="white")
        ax.set_title(column[:22])
        _style_axes(ax)
    fig.tight_layout()
    return _png(fig)


def chart_heatmap(eda: dict):
    corr = eda["correlations"]
    cols = corr["columns"][:12]
    if len(cols) < 2:
        return None
    matrix = np.array([[np.nan if v is None else v for v in row[:len(cols)]]
                       for row in corr["matrix"][:len(cols)]], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), [c[:12] for c in cols], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(cols)), [c[:12] for c in cols], fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(matrix[i, j]) > 0.5 else "#222222")
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix (Pearson)")
    return _png(fig)


def chart_categories(eda: dict):
    for column, stats in eda["categorical_statistics"].items():
        top = stats["top_values"][:8]
        if 2 <= stats["unique"] <= 60 and top:
            fig, ax = plt.subplots(figsize=(6, 2.6))
            ax.bar([t["value"][:14] for t in top], [t["count"] for t in top], color=PLOT_COLOR)
            ax.set_title(f"Most frequent values: {column[:30]}")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            _style_axes(ax)
            return _png(fig)
    return None


def chart_importance(run: dict):
    items = (run.get("details") or {}).get("feature_importance") or []
    if not items:
        return None
    items = items[:10]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh([i["feature"][:25] for i in items][::-1], [i["importance"] for i in items][::-1],
            color=PLOT_COLOR)
    ax.set_title("Feature importance (share of total)")
    _style_axes(ax)
    return _png(fig)


def chart_actual_vs_pred(run: dict):
    sample = (run.get("details") or {}).get("predictions_sample") or []
    if not sample:
        return None
    a = [s["actual"] for s in sample]
    p = [s["predicted"] for s in sample]
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.scatter(a, p, s=10, alpha=0.6, color=PLOT_COLOR)
    low, high = min(a + p), max(a + p)
    ax.plot([low, high], [low, high], color=PLOT_ALT, linewidth=1)
    ax.set_xlabel("Actual", fontsize=7)
    ax.set_ylabel("Predicted", fontsize=7)
    ax.set_title("Predicted vs actual (test set)")
    _style_axes(ax)
    return _png(fig)


def chart_confusion(run: dict):
    details = run.get("details") or {}
    matrix, classes = details.get("confusion_matrix"), details.get("classes")
    if not matrix or not classes or len(classes) > 12:
        return None
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(classes)), [str(c)[:10] for c in classes], rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), [str(c)[:10] for c in classes])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)
    ax.set_title("Confusion matrix (test set)")
    return _png(fig)


def _image(buffer, width_cm=15.5):
    if buffer is None:
        return None
    from reportlab.lib.utils import ImageReader

    reader = ImageReader(buffer)
    w, h = reader.getSize()
    buffer.seek(0)
    width = width_cm * cm
    return Image(buffer, width=width, height=width * h / w)


# --------------------------------------------------------------------------- the report
def build_report_pdf(
    out_path: Path,
    *,
    title: str,
    dataset: dict,
    df: pd.DataFrame,
    eda: dict,
    cleaning_recs: list[dict],
    runs: list[dict],
    ai_sections: list[dict],
    ai_notes: list[str],
    include_ai: bool,
) -> dict:
    S = _Styles()
    story: list = []
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    overview = eda["overview"]
    quality = eda["quality"]
    sections: list[str] = []

    def h1(text):
        sections.append(text)
        story.append(Paragraph(_clean(text), S.h1))

    # ---- title block
    story += [Spacer(1, 1.2 * cm), Paragraph(_clean(title), S.title),
              Paragraph(_clean(f"Dataset: {dataset['filename']}"), S.subtitle),
              Paragraph(f"Generated {generated} by AI Data Intelligence Platform", S.subtitle),
              Spacer(1, 0.4 * cm)]

    # ---- 1. project & dataset information
    h1("1. Project and dataset information")
    story.append(_table(S, ["Property", "Value"], [
        ["File name", dataset["filename"]],
        ["Dataset ID / active version", f"{dataset['id']} / v{dataset.get('version', '?')}"],
        ["Rows x columns", f"{overview['row_count']:,} x {overview['column_count']}"],
        ["Numeric / categorical columns",
         f"{len(overview['numeric_columns'])} / {len(overview['categorical_columns'])}"],
        ["Memory footprint", f"{overview['memory_bytes'] / 1024:,.1f} KB"],
        ["File SHA-256", (dataset.get("sha256") or "-")[:32] + "..."],
        ["Uploaded", dataset.get("created_at") or "-"],
    ], [5.5 * cm, 11 * cm]))

    # ---- 2. executive summary (deterministic)
    h1("2. Executive summary")
    missing = eda["missing"]
    facts = [
        f"The dataset has {overview['row_count']:,} rows and {overview['column_count']} columns.",
        f"Overall data quality score: {quality['quality_score']} / 100.",
        f"{missing['total_missing_cells']:,} cells are missing "
        f"({missing['rows_with_missing']:,} rows contain at least one missing value).",
        f"{eda['duplicates']['duplicate_rows']:,} duplicate rows were found.",
        f"{eda['outliers']['total_outliers']:,} IQR-based outliers were detected across numeric columns.",
    ]
    if eda["correlations"]["top_pairs"]:
        p = eda["correlations"]["top_pairs"][0]
        facts.append(f"Strongest numeric relationship: {p['column_a']} and {p['column_b']} "
                     f"(r = {p['correlation']}, {p['strength']} {p['direction']}).")
    if runs:
        best = max(runs, key=lambda r: (r.get("primary_metric_value") is not None,
                                        r.get("primary_metric_value") or 0))
        facts.append(f"{len(runs)} trained model(s) included; best primary metric: "
                     f"{best['primary_metric']} = {_fmt(best.get('primary_metric_value'))} "
                     f"({best['model_name']} v{best['version']}).")
    for f in facts:
        story.append(Paragraph(_clean(f), S.bullet, bulletText="\u2022"))

    # ---- 3. dataset overview
    h1("3. Dataset overview")
    rows = []
    for column in list(df.columns)[:40]:
        rows.append([column, overview["dtypes"][column], f"{int(df[column].isna().sum()):,}",
                     f"{int(df[column].nunique()):,}"])
    story.append(_table(S, ["Column", "Type", "Missing", "Distinct"], rows,
                        [7 * cm, 3.5 * cm, 3 * cm, 3 * cm]))
    if len(df.columns) > 40:
        story.append(Paragraph(f"Showing the first 40 of {len(df.columns)} columns.", S.small))

    # ---- 4. data quality
    h1("4. Data quality findings")
    comps = quality["score_components"]
    story.append(_table(S, ["Component", "Score"],
                        [[k.replace("_", " ").title(), _fmt(v)] for k, v in comps.items()],
                        [8 * cm, 4 * cm]))
    story.append(Spacer(1, 4))
    for w in quality["warnings"][:12] or ["No data-quality warnings were raised."]:
        story.append(Paragraph(_clean(w), S.bullet, bulletText="\u2022"))
    insight_rows = [[i["column"], i["severity"], i["message"]]
                    for i in eda["column_insights"][:20]]
    if insight_rows:
        story += [Paragraph("Column-level findings", S.h2),
                  _table(S, ["Column", "Severity", "Finding"], insight_rows,
                         [3.8 * cm, 2 * cm, 10.7 * cm])]

    # ---- 5. EDA
    h1("5. Exploratory data analysis")
    num = eda["numeric_statistics"]
    if num:
        story.append(Paragraph("Numeric columns", S.h2))
        story.append(_table(
            S, ["Column", "Mean", "Std", "Min", "Median", "Max", "Skew"],
            [[c, _fmt(s["mean"]), _fmt(s["std"]), _fmt(s["min"]), _fmt(s["median"]),
              _fmt(s["max"]), _fmt(s["skewness"])] for c, s in list(num.items())[:20]],
            [4.3 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 1.8 * cm]))
    cat = eda["categorical_statistics"]
    if cat:
        story.append(Paragraph("Categorical columns", S.h2))
        story.append(_table(
            S, ["Column", "Distinct", "Most frequent values"],
            [[c, s["unique"], "; ".join(f"{t['value']} ({t['count']})" for t in s["top_values"][:3])]
             for c, s in list(cat.items())[:15]], [4 * cm, 2 * cm, 10.5 * cm]))
    pairs = eda["correlations"]["top_pairs"]
    if pairs:
        story.append(Paragraph("Strongest correlations", S.h2))
        story.append(_table(S, ["Column A", "Column B", "Pearson r", "Strength"],
                            [[p["column_a"], p["column_b"], p["correlation"],
                              f"{p['strength']} ({p['direction']})"] for p in pairs[:8]],
                            [4.5 * cm, 4.5 * cm, 2.5 * cm, 5 * cm]))
        story.append(Paragraph("Correlation does not imply causation.", S.note))
    out = {c: v for c, v in eda["outliers"]["columns"].items() if v["outlier_count"]}
    if out:
        story.append(Paragraph("Outliers (IQR method)", S.h2))
        story.append(_table(S, ["Column", "Outliers", "% of values", "Lower bound", "Upper bound"],
                            [[c, v["outlier_count"], v["outlier_percentage"],
                              _fmt(v["lower_bound"]), _fmt(v["upper_bound"])]
                             for c, v in list(out.items())[:15]],
                            [4.5 * cm, 2.5 * cm, 2.8 * cm, 3.3 * cm, 3.4 * cm]))

    # ---- 6. visualizations
    h1("6. Visualizations")
    figures = [chart_missing(eda), chart_histograms(df, overview["numeric_columns"]),
               chart_heatmap(eda), chart_categories(eda)]
    shown = 0
    for figure in figures:
        img = _image(figure)
        if img is not None:
            story += [img, Spacer(1, 6)]
            shown += 1
    if not shown:
        story.append(Paragraph("No charts could be generated for this dataset.", S.body))

    # ---- 7. ML
    if runs:
        h1("7. Machine learning results")
        for run in runs:
            story.append(Paragraph(_clean(f"{run['model_name']}  (v{run['version']}, "
                                          f"{run['algorithm'].replace('_', ' ')})"), S.h2))
            cfg = run["config"]
            story.append(Paragraph(_clean(
                f"Task: {run['task']}. Target: {run.get('target') or 'n/a'}. "
                f"Features used: {len(run['features'])}. Random state: {cfg.get('random_state')}. "
                f"Test size: {cfg.get('test_size')}. Trained {run['created_at'][:19]} UTC "
                f"in {run.get('duration_ms', 0)} ms."), S.body))
            story.append(_table(
                S, ["Metric", "Value"],
                [[k, _fmt(v)] for k, v in run["metrics"].items() if isinstance(v, (int, float))],
                [7 * cm, 4 * cm]))
            for w in (run.get("details") or {}).get("warnings", [])[:6]:
                story.append(Paragraph(_clean(w), S.bullet, bulletText="\u2022"))
            imgs = [_image(chart_importance(run), 8), _image(chart_actual_vs_pred(run), 7),
                    _image(chart_confusion(run), 7)]
            for img in [i for i in imgs if i is not None][:2]:
                story += [Spacer(1, 4), img]
        story.append(Paragraph(
            "Metrics are computed on a held-out test partition. Compare against the baseline "
            "metrics before drawing conclusions.", S.note))

    # ---- 8. AI explanations
    n = 8 if runs else 7
    h1(f"{n}. AI-generated explanations (Google Gemini)")
    if ai_sections:
        for sec in ai_sections:
            story.append(Paragraph(_clean(sec["title"]), S.h2))
            story += _md(S, sec["text"])
            status = "verified" if sec["grounding"]["verified"] else "NOT fully verified"
            story.append(Paragraph(
                _clean(f"Model: {sec['model']}. Numeric grounding check: {status} "
                       f"({sec['grounding']['numbers_checked']} numbers checked against computed results)."),
                S.small))
    else:
        reason = "AI explanations were not requested." if not include_ai else (
            "AI explanations could not be generated.")
        story.append(Paragraph(_clean(reason), S.body))
    for note in ai_notes:
        story.append(Paragraph(_clean(note), S.note))

    # ---- 9. recommendations
    n += 1
    h1(f"{n}. Recommendations")
    recs = [f"{r['title']} - {r['reason']}" for r in cleaning_recs[:12]]
    recs += [r for r in quality.get("recommendations", [])[:6]]
    if not recs:
        recs = ["No data-cleaning actions are recommended."]
    for rec in dict.fromkeys(recs):
        story.append(Paragraph(_clean(rec), S.bullet, bulletText="\u2022"))

    # ---- 10. limitations
    n += 1
    h1(f"{n}. Limitations")
    limits = [
        "All statistics describe this file only; they may not generalise to other data.",
        "Correlation and feature importance do not establish causation.",
        "Outlier flags use a statistical rule (IQR) and may include legitimate values.",
    ]
    if overview["row_count"] < 500:
        limits.append(f"The dataset is small ({overview['row_count']} rows); "
                      "model metrics have high variance.")
    if missing["total_missing_cells"]:
        limits.append("Missing values reduce reliability; imputation inside models is simple "
                      "(median / most frequent).")
    if include_ai:
        limits.append("AI text is generated by a language model. Every number is checked against "
                      "computed results, but interpretations can still be wrong; verify before acting.")
    for item in limits:
        story.append(Paragraph(_clean(item), S.bullet, bulletText="\u2022"))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.2 * cm, _clean(title)[:70])
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm, title=title,
                            author="AI Data Intelligence Platform")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return {"sections": sections, "charts": shown}
