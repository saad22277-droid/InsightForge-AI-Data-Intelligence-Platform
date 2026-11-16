"""Render the architecture diagrams used by the docs and the technical report.

Usage: python scripts/generate_diagrams.py   ->  docs/diagrams/*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)
PALETTE = {"ui": "#dfeafe", "api": "#ebe9fd", "svc": "#dcf7ec", "store": "#fdeccd", "ext": "#fde1e1", "guard": "#f3f4f8"}


def box(ax, x, y, w, h, text, kind="svc", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=PALETTE[kind], ec="#6558e8" if kind != "ext" else "#ef4444", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#202338", wrap=True)


def arrow(ax, p, q, text=None, style="-|>", dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, lw=1.2, color="#55576b",
                                 linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, text, fontsize=7, ha="center", color="#55576b",
                bbox=dict(fc="white", ec="none", pad=0.6))


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(w / 2, h - 0.25, title, ha="center", fontsize=12, fontweight="bold", color="#202338")
    return fig, ax


def system():
    fig, ax = canvas(11, 6.6, "System architecture")
    box(ax, 0.4, 4.9, 10.2, 1.1, "React + TypeScript + Vite   (Recharts, Axios, Lucide)\nOverview - Datasets - Cleaning - Quality - EDA - Visualizations - ML Studio - Ask AI - Reports - Experiments - Settings", "ui", 8.5, True)
    box(ax, 0.4, 3.35, 10.2, 1.05, "FastAPI  (routers: datasets, analysis, cleaning, ml, experiments, ai, reports, system)\nPydantic validation - request-id logging middleware - AIServiceError handler - CORS", "api", 8.5, True)
    names = [("Data services\ndata_io - eda\nquality - charts", 0.4), ("Cleaning engine\nrecommend - preview\n- apply", 2.95),
             ("ML engine\nsklearn pipelines\nregistry - predict", 5.5), ("AI layer\ntools - orchestrator\ngrounding - Gemini", 8.05)]
    for t, x in names:
        box(ax, x, 1.75, 2.55, 1.15, t, "svc", 8)
        arrow(ax, (x + 1.27, 3.35), (x + 1.27, 2.9))
    box(ax, 0.4, 0.2, 4.0, 1.1, "PostgreSQL\ndatasets - versions - cleaning_runs - ml_runs\nchat_conversations - chat_messages - reports", "store", 8)
    box(ax, 4.7, 0.2, 3.0, 1.1, "Local filesystem\ndata/raw (CSV versions)\ndata/models (.joblib) - data/reports", "store", 8)
    box(ax, 8.0, 0.2, 2.6, 1.1, "Google Gemini API\n(optional, free tier)", "ext", 8.5, True)
    arrow(ax, (2.4, 1.75), (2.4, 1.3)); arrow(ax, (6.8, 1.75), (6.2, 1.3)); arrow(ax, (9.3, 1.75), (9.3, 1.3), "HTTPS", dashed=True)
    arrow(ax, (5.5, 4.9), (5.5, 4.4), "REST / JSON", style="<|-|>")
    fig.savefig(OUT / "system_architecture.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)


def chat():
    fig, ax = canvas(11, 5.6, "Dataset chat: grounded tool-calling flow")
    box(ax, 0.2, 3.6, 1.8, 0.9, "User question\n+ conversation\nhistory (PG)", "ui")
    box(ax, 2.4, 3.6, 2.0, 0.9, "Orchestrator\nsystem prompt +\nschema-only card", "api")
    box(ax, 4.9, 3.6, 1.9, 0.9, "Gemini\nplans tool calls\n(no code exec)", "ext", 8.5, True)
    box(ax, 7.3, 3.6, 3.4, 0.9, "Tool registry: validate args\n(schema, columns, ranges, limits)", "guard")
    box(ax, 7.3, 2.1, 3.4, 1.0, "10 read-only pandas tools\noverview - aggregate - outliers - correlation\nvalue_counts - filter_rows - quality ...", "svc")
    box(ax, 4.9, 2.1, 1.9, 1.0, "Sanitised JSON\nresults (row-capped,\nlabelled untrusted)", "guard")
    box(ax, 2.4, 2.1, 2.0, 1.0, "Gemini writes\nplain-language\nexplanation", "ext", 8.5, True)
    box(ax, 2.4, 0.5, 2.0, 1.0, "Grounding verifier:\nevery number must\nexist in results", "api", 8.5, True)
    box(ax, 5.2, 0.5, 2.3, 1.0, "Retry once with\ncorrection, else show\nraw computed results", "guard")
    box(ax, 8.0, 0.5, 2.7, 1.0, "Answer + evidence +\ngrounding badge,\nsaved to PostgreSQL", "store")
    arrow(ax, (2.0, 4.05), (2.4, 4.05)); arrow(ax, (4.4, 4.05), (4.9, 4.05)); arrow(ax, (6.8, 4.05), (7.3, 4.05), "tool call")
    arrow(ax, (9.0, 3.6), (9.0, 3.1)); arrow(ax, (7.3, 2.6), (6.8, 2.6)); arrow(ax, (4.9, 2.6), (4.4, 2.6))
    arrow(ax, (3.4, 2.1), (3.4, 1.5)); arrow(ax, (4.4, 1.0), (5.2, 1.0), "fails"); arrow(ax, (7.5, 1.0), (8.0, 1.0))
    # verified path: drawn underneath the retry box so nothing overlaps
    ax.plot([3.4, 3.4, 9.35], [0.5, 0.2, 0.2], color="#55576b", lw=1.2)
    arrow(ax, (9.35, 0.2), (9.35, 0.5))
    ax.text(6.3, 0.26, "verified", fontsize=7.5, ha="center", color="#0b7a53")
    arrow(ax, (5.85, 3.1), (5.85, 3.6), dashed=True)
    ax.text(5.95, 3.35, "results back to Gemini\n(may call more tools)", fontsize=7, ha="left", va="center", color="#55576b")
    fig.savefig(OUT / "chat_flow.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)


def pipeline():
    fig, ax = canvas(11, 4.6, "Dataset processing and ML pipeline")
    steps = [("Upload CSV\nsize/type checks\nUUID filename", "ui"), ("Version 1\nSHA-256 stored\noriginal kept", "store"),
             ("EDA + quality\nscore", "svc"), ("Cleaning\nrecommend > preview\n> apply = new version", "svc"),
             ("Train/test split\n(before fitting)", "api"), ("Pipeline:\nimpute > scale/encode\n> select > model", "api"),
             ("Metrics vs\nbaseline + CV", "svc"), ("Registry: joblib\n+ config + SHA-256\n+ versions", "store")]
    for i, (t, k) in enumerate(steps):
        x = 0.2 + (i % 4) * 2.7
        y = 2.7 if i < 4 else 0.7
        box(ax, x, y, 2.3, 1.1, t, k, 8)
        if i % 4 != 3:
            arrow(ax, (x + 2.3, y + 0.55), (x + 2.7, y + 0.55))
    ax.plot([0.2 + 3 * 2.7 + 1.15, 0.2 + 3 * 2.7 + 1.15, 1.35, 1.35], [2.7, 2.25, 2.25, 1.8], color="#55576b", lw=1.2)
    arrow(ax, (1.35, 1.9), (1.35, 1.8))
    ax.text(5.5, 2.33, "active dataset version feeds the ML studio (fingerprint recorded with every run)", fontsize=7.5, ha="center", color="#55576b")
    ax.text(5.5, 0.25, "Leakage guards: target never a feature - ID/duplicate-of-target columns excluded - preprocessing fitted on training rows only", ha="center", fontsize=7.5, color="#b42318")
    fig.savefig(OUT / "data_ml_pipeline.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    system(); chat(); pipeline()
    print("diagrams written to", OUT)
