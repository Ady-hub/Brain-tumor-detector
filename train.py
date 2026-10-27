"""
train.py — Train the lightweight models locally.

Run this once before launching app.py if no pre-trained models exist.
Models are saved to ./models_cpu/ and reused on subsequent runs.

Usage:
    python train.py
    python train.py --retrain       # force retraining even if models exist
    python train.py --ablation      # also train the raw-scan ablation model
"""
import sys
import json
import argparse

import numpy as np
import pandas as pd

from neurolens.config import (
    DEVICE_INFO, RETRAIN, RUN_ABLATION, MODEL_DIR,
    N_P1, N_P2, N_ABL, EP_CUSTOM, EP_P2, PAPER_REPORTED,
    CLASSES3, SEED, AUG_PAPER1, IMG, BATCH,
)
from neurolens.dataset import (
    load_dataset, prepare_splits, load_and_enhance, save_example_images, make_ds,
)
from neurolens.models import (
    fit_custom, train_or_load, build_custom_cnn, RESULTS, REG, safe,
)


def main():
    parser = argparse.ArgumentParser(description="NeuroLens — Train models locally")
    parser.add_argument("--retrain", action="store_true", help="Force retraining")
    parser.add_argument("--ablation", action="store_true", help="Train raw-scan ablation model")
    args = parser.parse_args()

    # Override config flags from CLI
    import neurolens.config as cfg
    if args.retrain:
        cfg.RETRAIN = True
    if args.ablation:
        cfg.RUN_ABLATION = True

    print(DEVICE_INFO)
    print()

    # ----- Load dataset -----
    X_all, y3_all = load_dataset()
    splits = prepare_splits(X_all, y3_all)
    X_all = splits["X_all"]
    y3_all = splits["y3_all"]
    y2_all = splits["y2_all"]
    idx_tr = splits["idx_tr"]
    idx_va = splits["idx_va"]
    idx_te = splits["idx_te"]

    # ----- Enhance -----
    X_enh = load_and_enhance(X_all)

    # ----- Save example images -----
    save_example_images(X_all, y3_all, idx_te)

    # ----- Load existing results if available -----
    if (MODEL_DIR / "results.json").exists() and not cfg.RETRAIN:
        try:
            RESULTS.update(json.loads((MODEL_DIR / "results.json").read_text()))
        except Exception:
            pass

    # ----- Paper 1: 3-class tumor-type CNN -----
    Xtr, ytr = X_all[idx_tr], y3_all[idx_tr]
    Xva, yva = X_all[idx_va], y3_all[idx_va]
    train_or_load(
        N_P1,
        lambda: fit_custom(Xtr, ytr, Xva, yva, EP_CUSTOM, n_classes=3),
        "custom", None, 3, False, "1",
        X_all, X_enh, y3_all, y2_all, idx_te,
    )

    # ----- Paper 2: enhanced 2-class CNN -----
    Xtr2, ytr2 = X_enh[idx_tr], y2_all[idx_tr]
    Xva2, yva2 = X_enh[idx_va], y2_all[idx_va]
    train_or_load(
        N_P2,
        lambda: fit_custom(Xtr2, ytr2, Xva2, yva2, EP_P2, n_classes=2),
        "custom", None, 2, True, "2",
        X_all, X_enh, y3_all, y2_all, idx_te,
    )

    # ----- Optional ablation -----
    if cfg.RUN_ABLATION:
        train_or_load(
            N_ABL,
            lambda: fit_custom(
                X_all[idx_tr], y2_all[idx_tr],
                X_all[idx_va], y2_all[idx_va],
                EP_P2, n_classes=2,
            ),
            "custom", None, 2, False, "ablation",
            X_all, X_enh, y3_all, y2_all, idx_te,
        )
    else:
        print("Raw-scan ablation skipped (use --ablation to enable).")

    # ----- Summary -----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    rows = []
    for n, r in RESULTS.items():
        rows.append({
            "Model": n,
            "Task": "tumor type (3 classes)" if r["n_classes"] == 3 else "benign vs malignant",
            "Accuracy %": round(r["acc"], 2),
            "F1 %": round(r["f1"], 2),
            "AUC %": round(r["auc"], 2),
            "Parameters": f"{r['params']:,}",
            "ms per scan": round(r["latency_ms"], 1),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nModels saved to:", MODEL_DIR)
    print("Run `python app.py` to launch the Gradio interface.\n")

    # Print model summary
    m = build_custom_cnn()
    print(f"Lightweight CNN: {m.count_params():,} parameters")
    del m


if __name__ == "__main__":
    main()
