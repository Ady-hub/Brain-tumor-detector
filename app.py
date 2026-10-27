"""
app.py — NeuroLens Gradio interface.

Loads trained models from ./models_cpu/ and launches the interactive
brain-tumor MRI classification app.

If no trained models are found, offers to train them automatically.

Usage:
    python app.py
    python app.py --share    # generate a public Gradio link
    python app.py --port 7860
"""
import os
import sys
import argparse
import tempfile
import time

import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neurolens.config import (
    DEVICE_INFO, MODEL_DIR, EX_DIR, CLASSES3, CLASSES2,
    MALIGNANT_TYPES, N_P1, N_P2, N_ABL, IMG, PAPER_REPORTED, WORK,
)
from neurolens.enhance import odtwche, enhancement_stats
from neurolens.models import (
    REG, RESULTS, load_registry_from_disk,
    to_gray224, predict, grad_cam, overlay, types3, paper2_name, safe,
)


# ---------------------------------------------------------------------------
# Bootstrap: load models or offer training
# ---------------------------------------------------------------------------
def bootstrap():
    """Load models; if none found, offer to train."""
    if load_registry_from_disk():
        print(f"Loaded {len(REG)} model(s): {', '.join(REG)}")
        return True

    print("\n" + "=" * 60)
    print("No trained models found in:", MODEL_DIR)
    print("=" * 60)
    print("\nYou need to train models first. Run:")
    print("    python train.py")
    print("\nOr type 'train' now to train interactively.")
    try:
        choice = input("\nTrain now? [train/quit]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "quit"
    if choice == "train":
        # Import and run training inline
        print()
        from train import main as train_main
        sys.argv = ["train.py"]  # reset argv for argparse
        train_main()
        # Reload after training
        load_registry_from_disk()
        return bool(REG)
    return False


# ---------------------------------------------------------------------------
# Colour palette and CSS
# ---------------------------------------------------------------------------
PAL = {
    "Glioma": "#F0648C", "Meningioma": "#F2B544", "Pituitary": "#8C7BFF",
    "Benign": "#3FD0B6", "Malignant": "#F0648C",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root { --ink:#0A0F1E; --panel:#121A2F; --panel2:#18223D; --line:rgba(237,232,220,.09); --bone:#EDE8DC; --muted:#98A2BA; }
.gradio-container { background: radial-gradient(1200px 520px at 8% -10%, #1B2A55 0%, rgba(10,15,30,0) 60%), var(--ink) !important;
  font-family:'IBM Plex Sans', system-ui, sans-serif !important; color: var(--bone); max-width: 1240px !important; }
.gradio-container h1,.gradio-container h2,.gradio-container h3 { font-family:'Fraunces', Georgia, serif !important; letter-spacing:-.01em; }
footer { display:none !important; }
button[role="tab"] { font-weight:500 !important; color: var(--muted) !important; font-size: 15px !important; }
button[role="tab"][aria-selected="true"] { color: var(--bone) !important; border-color: #3FD0B6 !important; }
.nl-hero { display:flex; gap:32px; align-items:center; justify-content:space-between; padding: 34px 36px; margin: 6px 0 14px;
  border:1px solid var(--line); border-radius:22px; background: linear-gradient(135deg, rgba(24,34,61,.92), rgba(18,26,47,.55)); overflow:hidden; }
.nl-hero h1 { font-size: clamp(2.2rem, 4.2vw, 3.5rem); line-height:1.02; margin:0 0 10px; font-weight:600; color: var(--bone); }
.nl-hero p { margin:0; max-width: 560px; font-size: 16.5px; line-height:1.6; color: var(--muted); }
.nl-chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
.nl-chip { border:1px solid var(--line); border-radius:999px; padding:6px 13px; font-size:13px; color: var(--bone); background: rgba(255,255,255,.03); }
.nl-chip i { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }
.nl-ring { flex:0 0 auto; opacity:.95; }
.nl-warn { border:1px solid rgba(242,181,68,.35); background: rgba(242,181,68,.07); color:#F2D9A0; border-radius:14px; padding:11px 16px; font-size:14px; margin-bottom:10px; }
.nl-card { border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:16px; padding:20px 22px; background: var(--panel); margin-bottom:12px; }
.nl-cap { color: var(--muted); font-size:14px; margin-bottom:2px; }
.nl-top { font-family:'Fraunces', Georgia, serif; font-size:2.35rem; line-height:1.1; font-weight:600; color: var(--accent); }
.nl-conf { color: var(--muted); font-size:14px; margin:4px 0 16px; font-variant-numeric: tabular-nums; }
.nl-row { display:grid; grid-template-columns: 112px 1fr 64px; gap:12px; align-items:center; margin:9px 0; }
.nl-name { font-size:15px; color: var(--bone); }
.nl-track { height:13px; border-radius:999px; background: rgba(255,255,255,.07); overflow:hidden; }
.nl-fill { height:100%; border-radius:999px; animation: nlgrow .95s cubic-bezier(.2,.8,.2,1); box-shadow: 0 0 14px -2px currentColor; }
.nl-pct { text-align:right; font-variant-numeric: tabular-nums; font-size:15px; color: var(--bone); font-weight:500; }
@keyframes nlgrow { from { width:0 } }
.nl-note { margin-top:12px; padding:10px 14px; border-radius:12px; font-size:14px; line-height:1.5; background: rgba(255,255,255,.04); color: var(--muted); }
.nl-note b { color: var(--bone); }
.nl-note.ok { background: rgba(63,208,182,.09); color:#A9EBDD; } .nl-note.bad { background: rgba(240,100,140,.10); color:#F6B5C8; }
.nl-empty { border:1px dashed rgba(237,232,220,.18); border-radius:16px; padding:34px 26px; color: var(--muted); text-align:left; font-size:16px; line-height:1.6; }
.nl-empty b { color: var(--bone); font-family:'Fraunces',Georgia,serif; font-size:1.25rem; font-weight:600; display:block; margin-bottom:4px; }
table.nl-t { width:100%; border-collapse:collapse !important; border:0 !important; font-size:14.5px; font-variant-numeric: tabular-nums; }
table.nl-t tr, table.nl-t th, table.nl-t td { background: transparent !important; border-left:0 !important; border-right:0 !important; border-top:0 !important; }
table.nl-t th { text-align:left; color: var(--muted); font-weight:500; padding:10px 12px; border-bottom:1px solid var(--line) !important; }
table.nl-t td { padding:11px 12px; border-bottom:1px solid var(--line) !important; color: var(--bone); }
table.nl-t tr.best td { background: rgba(63,208,182,.10) !important; }
.nl-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; }
.nl-about { color: var(--muted); line-height:1.7; font-size:15.5px; } .nl-about b { color: var(--bone); }
.nl-about h3 { color: var(--bone); margin: 18px 0 6px; }
.gradio-container .primary { font-weight:600 !important; border-radius:12px !important; }
.gradio-container .block { border-radius:16px !important; }
@media (max-width: 760px) { .nl-hero { flex-direction:column; align-items:flex-start; padding:24px; } .nl-ring { display:none; } .nl-row { grid-template-columns: 90px 1fr 56px; } }
@media (prefers-reduced-motion: reduce) { .nl-fill { animation:none; } }
"""

HERO = """
<div class="nl-hero">
  <div>
    <h1>NeuroLens</h1>
    <p>Upload a brain MRI slice and two research papers read it side by side: one names the tumor type, the other judges whether it is benign or malignant.</p>
    <div class="nl-chips">
      <span class="nl-chip"><i style="background:#F0648C"></i>Paper 1: custom CNN</span>
      <span class="nl-chip"><i style="background:#3FD0B6"></i>Paper 2: enhancement + lightweight CNN</span>
      <span class="nl-chip"><i style="background:#F2B544"></i>Grad-CAM shows where the model looked</span>
    </div>
  </div>
  <svg class="nl-ring" width="190" height="190" viewBox="0 0 190 190" fill="none" aria-hidden="true">
    <ellipse cx="95" cy="95" rx="86" ry="78" stroke="#EDE8DC" stroke-opacity=".22" stroke-width="1.5"/>
    <ellipse cx="95" cy="95" rx="62" ry="56" stroke="#8C7BFF" stroke-opacity=".55" stroke-width="1.5"/>
    <ellipse cx="95" cy="95" rx="38" ry="34" stroke="#3FD0B6" stroke-opacity=".7" stroke-width="1.5"/>
    <circle cx="118" cy="78" r="13" fill="#F0648C" fill-opacity=".85"/>
    <circle cx="118" cy="78" r="22" stroke="#F0648C" stroke-opacity=".4" stroke-width="1.5"/>
    <path d="M95 17V173M17 95H173" stroke="#EDE8DC" stroke-opacity=".10"/>
  </svg>
</div>
<div class="nl-warn">Student research demo. It is not a medical device and must not be used to diagnose anyone.</div>
"""

EMPTY = '<div class="nl-empty"><b>No scan yet</b>Add a brain MRI slice, or pick one of the sample scans, then press the button.</div>'


# ---------------------------------------------------------------------------
# UI helper functions
# ---------------------------------------------------------------------------
def card_html(caption, classes, probs, model_name):
    probs = np.asarray(probs, float)
    k = int(probs.argmax())
    top = classes[k]
    rows = ""
    for i in np.argsort(-probs):
        c, p = PAL[classes[i]], probs[i] * 100
        rows += (
            f'<div class="nl-row"><div class="nl-name">'
            f'<span class="nl-dot" style="background:{c}"></span>{classes[i]}</div>'
            f'<div class="nl-track"><div class="nl-fill" '
            f'style="width:{p:.1f}%;background:{c};color:{c}"></div></div>'
            f'<div class="nl-pct">{p:.1f}%</div></div>'
        )
    low = (
        '<div class="nl-note"><b>Low confidence.</b> '
        'The top two answers are close, so treat this reading as uncertain.</div>'
        if probs.max() < 0.6 else ""
    )
    return (
        f'<div class="nl-card" style="--accent:{PAL[top]}">'
        f'<div class="nl-cap">{caption}</div>'
        f'<div class="nl-top">{top}</div>'
        f'<div class="nl-conf">{probs[k]*100:.1f}% confidence from {model_name}</div>'
        f'{rows}{low}</div>'
    )


def agreement_note(type_label, malig_label):
    expect = "Malignant" if type_label in MALIGNANT_TYPES else "Benign"
    if expect == malig_label:
        return (
            f'<div class="nl-note ok"><b>The two papers agree.</b> '
            f'{type_label} is normally {expect.lower()}, and Paper 2 also says '
            f'{malig_label.lower()}.</div>'
        )
    return (
        f'<div class="nl-note bad"><b>The two papers disagree.</b> '
        f'{type_label} is normally {expect.lower()}, but Paper 2 says '
        f'{malig_label.lower()}. Treat both readings as uncertain.</div>'
    )


def dark_fig(w=6.4, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#121A2F")
    ax.set_facecolor("#121A2F")
    for s in ax.spines.values():
        s.set_color("#2A3558")
    ax.tick_params(colors="#98A2BA")
    ax.xaxis.label.set_color("#98A2BA")
    ax.yaxis.label.set_color("#98A2BA")
    ax.title.set_color("#EDE8DC")
    return fig, ax


# ---------------------------------------------------------------------------
# Tab handlers
# ---------------------------------------------------------------------------
def diagnose(img, mode, backbone, show_cam):
    if img is None:
        return EMPTY, None, None, None
    g = to_gray224(img)
    enh = odtwche(g)
    html = ""
    cam_img = None
    p1 = p2 = None
    if mode != "Paper 2: benign or malignant":
        p, ms = predict(backbone, g)
        p1 = p
        html += card_html(
            "Paper 1 reading: tumor type", CLASSES3, p,
            f"{backbone} ({ms:.0f} ms)"
        )
        if show_cam:
            cam_img = overlay(g, grad_cam(backbone, g))
    if mode != "Paper 1: tumor type" and paper2_name():
        p, ms = predict(paper2_name(), g)
        p2 = p
        html += card_html(
            "Paper 2 reading: benign or malignant", CLASSES2, p,
            f"{paper2_name()} ({ms:.0f} ms)"
        )
        if show_cam and cam_img is None:
            cam_img = overlay(enh, grad_cam(paper2_name(), g))
    if p1 is not None and p2 is not None:
        html += agreement_note(
            CLASSES3[int(p1.argmax())], CLASSES2[int(p2.argmax())]
        )
    return html, g, enh, cam_img


def showdown(img):
    if img is None:
        return EMPTY
    g = to_gray224(img)
    rows = ""
    votes = []
    for n, e in REG.items():
        if e["paper"] == "ablation":
            continue
        p, ms = predict(n, g)
        k = int(p.argmax())
        lab = e["classes"][k]
        if e["n_classes"] == 3:
            votes.append(lab)
        rows += (
            f'<tr><td>{n}</td>'
            f'<td><span class="nl-dot" style="background:{PAL[lab]}"></span>{lab}</td>'
            f'<td>{p[k]*100:.1f}%</td>'
            f'<td>{e["model"].count_params()/1e6:.2f} M</td>'
            f'<td>{ms:.0f} ms</td></tr>'
        )
    top = max(set(votes), key=votes.count) if votes else None
    summary = (
        f'<div class="nl-note ok"><b>{votes.count(top)} of {len(votes)} '
        f'tumor-type models say {top}.</b> '
        'Paper 1 claims a small custom network can match or beat much larger ones, '
        'so compare the size and speed columns.</div>'
    ) if top else ""
    return (
        f'<div class="nl-card" style="--accent:#8C7BFF">'
        f'<div class="nl-cap">Every trained model reads the same scan</div>'
        f'<table class="nl-t"><tr><th>Model</th><th>Reading</th>'
        f'<th>Confidence</th><th>Size</th><th>Speed</th></tr>{rows}</table>'
        f'{summary}</div>'
    )


def lab(img, contrast_cut, noise, brightness):
    if img is None:
        return None, None, None, EMPTY
    g = to_gray224(img).astype(float)
    deg = (g - g.mean()) * (1 - contrast_cut / 100) + g.mean() + brightness
    deg = np.clip(
        deg + np.random.default_rng(1).normal(0, noise, deg.shape), 0, 255
    ).astype(np.uint8)
    enh = odtwche(deg)
    s = enhancement_stats(deg, enh)

    fig, ax = dark_fig(6.4, 3.3)
    ax.hist(deg[deg > 0].ravel(), bins=64, range=(0, 255), alpha=0.65,
            color="#F2B544", label="degraded scan")
    ax.hist(enh[enh > 0].ravel(), bins=64, range=(0, 255), alpha=0.65,
            color="#3FD0B6", label="after enhancement")
    ax.set_xlabel("pixel intensity")
    ax.set_ylabel("pixels")
    ax.legend(facecolor="#121A2F", labelcolor="#EDE8DC", edgecolor="#2A3558")
    fig.tight_layout()

    html = ""
    if N_ABL in REG:
        html += card_html(
            "Without enhancement (raw-scan lightweight CNN)",
            CLASSES2, predict(N_ABL, deg)[0], N_ABL
        )
    if paper2_name():
        html += card_html(
            "With Paper 2 enhancement",
            CLASSES2, predict(paper2_name(), deg)[0], paper2_name()
        )
    html += (
        f'<div class="nl-note">Contrast (std) {s["contrast_before"]:.1f} to '
        f'{s["contrast_after"]:.1f}. Entropy {s["entropy_before"]:.2f} to '
        f'{s["entropy_after"]:.2f} bits. PSNR {s["psnr"]:.1f} dB. '
        f'Drag the sliders to make the scan worse and watch whether the '
        f'readings hold.</div>'
    )
    return deg, enh, fig, html


def dash_table():
    rows = ""
    best = {
        t: max((r["acc"] for r in RESULTS.values() if r["n_classes"] == t), default=0)
        for t in (2, 3)
    }
    for n, r in sorted(RESULTS.items(), key=lambda kv: (kv[1]["n_classes"], -kv[1]["acc"])):
        rep = PAPER_REPORTED.get(n)
        rep = f"{rep:.2f}%" if rep else "-"
        cls = "best" if r["acc"] == best[r["n_classes"]] else ""
        rows += (
            f'<tr class="{cls}"><td>{n}</td>'
            f'<td>{"tumor type" if r["n_classes"]==3 else "benign / malignant"}</td>'
            f'<td>{r["acc"]:.2f}%</td><td>{rep}</td>'
            f'<td>{r["f1"]:.1f}%</td><td>{r["auc"]:.1f}%</td>'
            f'<td>{r["params"]/1e6:.2f} M</td><td>{r["latency_ms"]:.0f} ms</td></tr>'
        )
    return (
        f'<div class="nl-card" style="--accent:#3FD0B6">'
        f'<div class="nl-cap">Test-set results (highlighted = best per task)</div>'
        f'<table class="nl-t"><tr><th>Model</th><th>Task</th><th>Accuracy</th>'
        f'<th>Paper reported</th><th>F1</th><th>AUC</th><th>Size</th><th>Speed</th></tr>'
        f'{rows}</table></div>'
    )


def dash_model(name):
    from sklearn.metrics import roc_curve
    r = RESULTS[name]
    cm = np.array(r["cm"])
    cl = r["classes"]

    fig, ax = dark_fig(5.2, 4.4)
    ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(cl)), cl)
    ax.set_yticks(range(len(cl)), cl)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix", loc="left", color="#EDE8DC")
    for i in range(len(cl)):
        for j in range(len(cl)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=13)
    fig.tight_layout()

    probs_path = MODEL_DIR / f"{safe(name)}_probs.npy"
    y_path = MODEL_DIR / f"{safe(name)}_y.npy"
    fig2, ax2 = dark_fig(5.2, 4.4)
    if probs_path.exists() and y_path.exists():
        pr = np.load(probs_path)
        y = np.load(y_path)
        for k, c in enumerate(cl):
            if len(cl) == 2 and k == 0:
                continue
            fpr, tpr, _ = roc_curve((y == k).astype(int), pr[:, k])
            ax2.plot(fpr, tpr, color=PAL[c], lw=2.2, label=c)
    ax2.plot([0, 1], [0, 1], color="#2A3558", ls="--")
    ax2.set_xlabel("false positive rate")
    ax2.set_ylabel("true positive rate")
    ax2.set_title(f"ROC curve (AUC {r['auc']:.1f}%)", loc="left", color="#EDE8DC")
    ax2.legend(facecolor="#121A2F", labelcolor="#EDE8DC", edgecolor="#2A3558")
    fig2.tight_layout()

    pc = "  |  ".join(f"{c}: {v:.1f}%" for c, v in r["per_class"].items())
    return fig, fig2, f'<div class="nl-note"><b>Per-class accuracy.</b> {pc}</div>'


def dash_bars():
    names = list(RESULTS.keys())
    fig, ax = dark_fig(9.6, 4.0)
    if not names:
        return fig
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [RESULTS[n]["acc"] for n in names], 0.38,
           color="#3FD0B6", label="ours")
    ax.bar(xs + 0.2, [PAPER_REPORTED.get(n, np.nan) for n in names], 0.38,
           color="#8C7BFF", label="paper reported")
    ax.set_xticks(xs, [
        n.replace(" (baseline)", "").replace("Paper ", "P").replace("on raw scans ", "raw ")
        for n in names
    ], rotation=18, ha="right")
    ax.set_ylim(max(0, min(RESULTS[n]["acc"] for n in names) - 10), 106)
    ax.set_ylabel("accuracy %")
    ax.legend(facecolor="#121A2F", labelcolor="#EDE8DC", edgecolor="#2A3558")
    fig.tight_layout()
    return fig


def batch(files):
    from fpdf import FPDF
    if not files:
        return pd.DataFrame(), None
    rows = []
    pdf = FPDF()
    pdf.set_auto_page_break(True, 12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 17)
    pdf.cell(0, 10, "NeuroLens - brain MRI reading report",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Student research demo. Not a medical device. Not for diagnosis.")

    for f in files:
        path = f if isinstance(f, str) else f.name
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        g = to_gray224(im)
        p1, _ = predict(N_P1 if N_P1 in REG else types3()[0], g)
        p2 = predict(paper2_name(), g)[0] if paper2_name() else None
        t, tc = CLASSES3[int(p1.argmax())], p1.max() * 100
        m, mc = (CLASSES2[int(p2.argmax())], p2.max() * 100) if p2 is not None else ("-", 0.0)
        agree = "yes" if m == ("Malignant" if t in MALIGNANT_TYPES else "Benign") else "check"
        rows.append({
            "Scan": os.path.basename(path),
            "Tumor type (Paper 1)": t,
            "Confidence 1 %": round(tc, 1),
            "Benign or malignant (Paper 2)": m,
            "Confidence 2 %": round(mc, 1),
            "Papers agree": agree,
        })
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
            tmp = tmp_f.name
        g3 = lambda a: np.repeat(a[..., None], 3, axis=-1)
        cam_name = N_P1 if N_P1 in REG else types3()[0]
        strip = np.hstack([
            g3(g), g3(odtwche(g)), overlay(g, grad_cam(cam_name, g))
        ])[..., ::-1]
        cv2.imwrite(tmp, strip)
        if pdf.get_y() > 215:
            pdf.add_page()
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, os.path.basename(path), new_x="LMARGIN", new_y="NEXT")
        pdf.image(tmp, w=120)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0, 5,
            f"Paper 1: {t} ({tc:.1f}%)    Paper 2: {m} ({mc:.1f}%)    Papers agree: {agree}",
            new_x="LMARGIN", new_y="NEXT",
        )
    out = str(WORK / "neurolens_report.pdf")
    pdf.output(out)
    return pd.DataFrame(rows), out


# ---------------------------------------------------------------------------
# About text
# ---------------------------------------------------------------------------
ABOUT = """<div class="nl-about">
<h3>Paper 1: a small CNN that beats big ones</h3>
Rasheed et al., <i>Brain Sciences</i>, 2023. A custom CNN sorts scans into glioma, meningioma or pituitary tumor. This CPU version uses a smaller three-block CNN and a stratified subset, so its metrics are not a reproduction of the paper's 98.04% result.
<h3>Paper 2: fix the image first, then classify</h3>
Agarwal et al., <i>Results in Engineering</i>, 2024. Low-contrast scans are enhanced (Otsu thresholds, a simplified PSO-style weighting, histogram equalisation, adaptive gamma, Wiener filter), then classified as benign or malignant. This CPU version substitutes a lightweight CNN for InceptionV3; its results are not a reproduction of the paper's 98.89% result.
<h3>What this app adds</h3>
<b>Grad-CAM</b> heatmaps show where the network looked, which both papers list as missing. The <b>enhancement lab</b> tests Paper 2's robustness claim. The raw-scan ablation is available as an optional training flag.
<h3>Honest limits</h3>
Both papers use the same 3,064 scans from two Chinese hospitals, so results may not hold for other scanners. Our Paper 2 labels treat glioma as malignant and the other two as benign. Our numbers use a fresh split and may differ from the papers.
</div>"""


# ---------------------------------------------------------------------------
# Build and launch the Gradio app
# ---------------------------------------------------------------------------
def build_app():
    import gradio as gr

    _vars = dict(
        body_background_fill="#0A0F1E", body_text_color="#EDE8DC",
        background_fill_primary="#121A2F", background_fill_secondary="#0E1527",
        block_background_fill="#121A2F",
        block_border_color="rgba(237,232,220,0.09)",
        block_label_background_fill="#0E1527",
        block_label_text_color="#98A2BA", block_title_text_color="#98A2BA",
        border_color_primary="rgba(237,232,220,0.12)",
        input_background_fill="#0E1527",
        input_border_color="rgba(237,232,220,0.12)",
        body_text_color_subdued="#98A2BA",
        button_primary_background_fill="#3FD0B6",
        button_primary_background_fill_hover="#59E0C8",
        button_primary_text_color="#06111A",
        button_secondary_background_fill="#18223D",
        button_secondary_text_color="#EDE8DC",
        panel_background_fill="#121A2F",
        table_even_background_fill="#121A2F",
        table_odd_background_fill="#0E1527",
        color_accent_soft="#18223D",
    )
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    ).set(**_vars, **{k + "_dark": v for k, v in _vars.items()})

    examples = sorted(str(p) for p in EX_DIR.glob("*.png"))

    with gr.Blocks(theme=theme, css=CSS, title="NeuroLens") as demo:
        gr.HTML(HERO)
        with gr.Tabs():
            # Tab 1: Diagnose
            with gr.Tab("Diagnose"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        inp = gr.Image(
                            label="Brain MRI slice", type="numpy",
                            image_mode="L", sources=["upload", "clipboard"],
                            height=330,
                        )
                        mode = gr.Radio(
                            ["Both papers", "Paper 1: tumor type",
                             "Paper 2: benign or malignant"],
                            value="Both papers",
                            label="Which paper should read it?",
                        )
                        backbone = gr.Dropdown(
                            types3(),
                            value=N_P1 if N_P1 in REG else (types3()[0] if types3() else None),
                            label="Tumor-type model (Paper 1 and baselines)",
                        )
                        cam = gr.Checkbox(True, label="Show Grad-CAM heatmap")
                        go = gr.Button("Read this scan", variant="primary", size="lg")
                        if examples:
                            gr.Examples(examples, inputs=inp,
                                        label="Sample scans from the unseen test set")
                    with gr.Column(scale=7):
                        out_html = gr.HTML(EMPTY)
                        with gr.Row():
                            o1 = gr.Image(label="Scan (224 × 224)", interactive=False,
                                          height=210, show_download_button=False)
                            o2 = gr.Image(label="Enhanced (Paper 2)", interactive=False,
                                          height=210, show_download_button=False)
                            o3 = gr.Image(label="Grad-CAM heatmap", interactive=False,
                                          height=210, show_download_button=False)
                go.click(diagnose, [inp, mode, backbone, cam], [out_html, o1, o2, o3])
                inp.upload(diagnose, [inp, mode, backbone, cam], [out_html, o1, o2, o3])

            # Tab 2: Model showdown
            with gr.Tab("Model showdown"):
                sd_in = gr.Image(
                    label="Brain MRI slice", type="numpy", image_mode="L",
                    sources=["upload", "clipboard"], height=260,
                )
                sd_btn = gr.Button("Run every model", variant="primary")
                sd_out = gr.HTML(EMPTY)
                sd_btn.click(showdown, sd_in, sd_out)
                if examples:
                    gr.Examples(examples, inputs=sd_in, label="Samples")

            # Tab 3: Enhancement lab
            with gr.Tab("Enhancement lab"):
                gr.Markdown(
                    "Make a scan worse on purpose, then see how Paper 2's "
                    "enhancement step and classifier cope."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4):
                        lab_in = gr.Image(
                            label="Brain MRI slice", type="numpy", image_mode="L",
                            sources=["upload", "clipboard"], height=240,
                        )
                        s1 = gr.Slider(0, 80, 40, step=5, label="Reduce contrast (%)")
                        s2 = gr.Slider(0, 30, 6, step=1, label="Add noise")
                        s3 = gr.Slider(-40, 40, 0, step=5, label="Shift brightness")
                        lab_btn = gr.Button("Degrade and enhance", variant="primary")
                        if examples:
                            gr.Examples(examples, inputs=lab_in, label="Samples")
                    with gr.Column(scale=7):
                        with gr.Row():
                            l1 = gr.Image(label="Degraded scan", interactive=False,
                                          height=210, show_download_button=False)
                            l2 = gr.Image(label="Enhanced scan", interactive=False,
                                          height=210, show_download_button=False)
                        l3 = gr.Plot(show_label=False)
                        l4 = gr.HTML(EMPTY)
                lab_btn.click(lab, [lab_in, s1, s2, s3], [l1, l2, l3, l4])

            # Tab 4: Metrics dashboard
            with gr.Tab("Metrics dashboard"):
                if RESULTS:
                    gr.HTML(dash_table())
                    gr.Plot(dash_bars(), show_label=False)
                    pick = gr.Dropdown(
                        list(RESULTS), value=list(RESULTS)[0],
                        label="Inspect a model",
                    )
                    with gr.Row():
                        cmp_ = gr.Plot(show_label=False)
                        roc_ = gr.Plot(show_label=False)
                    pc_html = gr.HTML()
                    pick.change(dash_model, pick, [cmp_, roc_, pc_html])
                    demo.load(dash_model, pick, [cmp_, roc_, pc_html])
                else:
                    gr.Markdown("No evaluation results yet. Run `python train.py` first.")

            # Tab 5: Batch and report
            with gr.Tab("Batch and report"):
                gr.Markdown("Upload several scans to get a table and a downloadable PDF report.")
                files_input = gr.File(
                    file_count="multiple", file_types=["image"],
                    label="Brain MRI slices",
                )
                b_btn = gr.Button("Read all scans", variant="primary")
                b_tbl = gr.Dataframe(interactive=False)
                b_pdf = gr.File(label="PDF report")
                b_btn.click(batch, files_input, [b_tbl, b_pdf])

            # Tab 6: About
            with gr.Tab("About"):
                gr.HTML(ABOUT)

    return demo


def main():
    parser = argparse.ArgumentParser(description="NeuroLens — Gradio app")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Local port (default: 7860)")
    args = parser.parse_args()

    print(DEVICE_INFO)
    print()

    if not bootstrap():
        print("Cannot start without trained models. Exiting.")
        sys.exit(1)

    demo = build_app()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        debug=False,
    )


if __name__ == "__main__":
    main()
