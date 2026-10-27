"""
test_neurolens.py — Comprehensive end-to-end test suite for NeuroLens.

Tests:
1. Contrast enhancement (ODTWCHE & metrics)
2. Model building (custom CNN 3-class & 2-class)
3. Model training on CPU (1 epoch with synthetic data)
4. Model serialization & registry saving/loading
5. Inference, predictions & Grad-CAM heatmap generation
6. PDF report generation with fpdf2
7. Gradio app interface construction & callback verification
"""
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2

# Set project root in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from neurolens.config import (
    IMG, CLASSES3, CLASSES2, MODEL_DIR, EX_DIR, WORK,
    N_P1, N_P2, DEVICE_INFO
)
from neurolens.enhance import odtwche, enhancement_stats
from neurolens.models import (
    build_custom_cnn, fit_custom, register, evaluate,
    load_registry_from_disk, predict, grad_cam, overlay,
    REG, RESULTS, to_gray224
)
import app


def test_1_enhancement():
    print("\n--- Test 1: Contrast Enhancement (ODTWCHE) ---")
    dummy = (np.random.rand(IMG, IMG) * 180 + 30).astype(np.uint8)
    enh = odtwche(dummy)
    assert enh.shape == (IMG, IMG), f"Shape mismatch: {enh.shape}"
    assert enh.dtype == np.uint8, f"Type mismatch: {enh.dtype}"

    stats = enhancement_stats(dummy, enh)
    assert "entropy_before" in stats and "entropy_after" in stats
    assert "psnr" in stats and stats["psnr"] > 0
    print(f"  Enhancement passed: entropy {stats['entropy_before']:.2f} -> {stats['entropy_after']:.2f}, PSNR {stats['psnr']:.1f} dB")


def test_2_model_builders():
    print("\n--- Test 2: Model Builders ---")
    m3 = build_custom_cnn(n_classes=3)
    assert m3.input_shape == (None, IMG, IMG, 1)
    assert m3.output_shape == (None, 3)
    p3 = m3.count_params()
    print(f"  3-class CNN params: {p3:,}")

    m2 = build_custom_cnn(n_classes=2)
    assert m2.output_shape == (None, 2)
    p2 = m2.count_params()
    print(f"  2-class CNN params: {p2:,}")
    assert p3 > 0 and p2 > 0


def test_3_training_and_saving():
    print("\n--- Test 3: CPU Training and Model Saving ---")
    n_samples = 15
    np.random.seed(42)
    X_synth = np.random.randint(20, 230, size=(n_samples, IMG, IMG), dtype=np.uint8)
    y3_synth = np.array([0, 1, 2] * (n_samples // 3))
    y2_synth = (y3_synth == 0).astype(int)

    # Train 1 epoch for 3-class model
    print("  Training Paper 1 model (3-class) on CPU for 1 epoch...")
    m3 = fit_custom(X_synth[:10], y3_synth[:10], X_synth[10:13], y3_synth[10:13], epochs=1, n_classes=3)
    p1_path = MODEL_DIR / f"{app.safe(N_P1)}.keras"
    m3.save(p1_path)
    register(N_P1, m3, "custom", None, 3, False, "1")

    # Train 1 epoch for 2-class model
    print("  Training Paper 2 model (2-class) on CPU for 1 epoch...")
    m2 = fit_custom(X_synth[:10], y2_synth[:10], X_synth[10:13], y2_synth[10:13], epochs=1, n_classes=2)
    p2_path = MODEL_DIR / f"{app.safe(N_P2)}.keras"
    m2.save(p2_path)
    register(N_P2, m2, "custom", None, 2, True, "2")

    # Evaluate both
    idx_te = np.array([12, 13, 14])
    evaluate(N_P1, X_synth, X_synth, y3_synth, y2_synth, idx_te)
    evaluate(N_P2, X_synth, X_synth, y3_synth, y2_synth, idx_te)

    assert p1_path.exists(), f"Missing {p1_path}"
    assert p2_path.exists(), f"Missing {p2_path}"
    assert (MODEL_DIR / "registry.json").exists()
    assert (MODEL_DIR / "results.json").exists()
    print("  Models trained, evaluated, and saved successfully.")


def test_4_loading_and_inference():
    print("\n--- Test 4: Model Loading from Disk and Inference ---")
    REG.clear()
    RESULTS.clear()
    loaded = load_registry_from_disk()
    assert loaded, "Failed to load registry from disk"
    assert N_P1 in REG and N_P2 in REG

    dummy = (np.random.rand(IMG, IMG) * 200).astype(np.uint8)
    p1, ms1 = predict(N_P1, dummy)
    assert len(p1) == 3
    assert abs(sum(p1) - 1.0) < 1e-4

    p2, ms2 = predict(N_P2, dummy)
    assert len(p2) == 2
    assert abs(sum(p2) - 1.0) < 1e-4

    print(f"  Inference successful: Paper 1 ({ms1:.1f} ms), Paper 2 ({ms2:.1f} ms)")


def test_5_grad_cam():
    print("\n--- Test 5: Grad-CAM and Heatmap Overlay ---")
    dummy = (np.random.rand(IMG, IMG) * 200).astype(np.uint8)
    cam = grad_cam(N_P1, dummy)
    assert cam.shape == (IMG, IMG), f"CAM shape mismatch: {cam.shape}"
    assert cam.min() >= 0.0 and cam.max() <= 1.0

    heat = overlay(dummy, cam)
    assert heat.shape == (IMG, IMG, 3), f"Overlay shape mismatch: {heat.shape}"
    assert heat.dtype == np.uint8
    print("  Grad-CAM and overlay passed.")


def test_6_pdf_report_and_examples():
    print("\n--- Test 6: Example Generation and PDF Report ---")
    # Save test example images
    ex1 = EX_DIR / "glioma_test.png"
    ex2 = EX_DIR / "meningioma_test.png"
    sample = (np.random.rand(IMG, IMG) * 200).astype(np.uint8)
    cv2.imwrite(str(ex1), sample)
    cv2.imwrite(str(ex2), sample)

    # Test batch PDF generation
    df, pdf_path = app.batch([str(ex1), str(ex2)])
    assert not df.empty, "Batch DataFrame is empty"
    assert len(df) == 2
    assert Path(pdf_path).exists()
    assert Path(pdf_path).stat().st_size > 1000
    print(f"  PDF report generated successfully: {pdf_path} ({Path(pdf_path).stat().st_size} bytes)")


def test_7_gradio_app():
    print("\n--- Test 7: Gradio Interface Construction ---")
    demo = app.build_app()
    assert demo is not None
    # Test diagnose function
    sample = (np.random.rand(IMG, IMG) * 200).astype(np.uint8)
    html, g, enh, cam_img = app.diagnose(sample, "Both papers", N_P1, True)
    assert html != app.EMPTY
    assert g.shape == (IMG, IMG)
    assert enh.shape == (IMG, IMG)
    assert cam_img.shape == (IMG, IMG, 3)

    # Test showdown function
    sd_html = app.showdown(sample)
    assert "Paper 1" in sd_html

    # Test enhancement lab function
    deg, enh2, fig, lab_html = app.lab(sample, 40, 6, 0)
    assert deg.shape == (IMG, IMG)
    assert enh2.shape == (IMG, IMG)
    print("  Gradio interface and UI callbacks tested successfully.")


def run_all():
    print("=" * 60)
    print("NeuroLens Automated Test Suite")
    print(DEVICE_INFO)
    print("=" * 60)

    test_1_enhancement()
    test_2_model_builders()
    test_3_training_and_saving()
    test_4_loading_and_inference()
    test_5_grad_cam()
    test_6_pdf_report_and_examples()
    test_7_gradio_app()

    print("\n" + "=" * 60)
    print("ALL 7 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
