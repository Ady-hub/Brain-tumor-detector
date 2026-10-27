# NeuroLens — Brain Tumor MRI Classification using Deep Learning

A fully local, CPU-friendly deep-learning application that classifies brain MRI scans into **Glioma**, **Meningioma**, or **Pituitary** tumor types, with an additional **benign vs malignant** classifier, **Grad-CAM** visualisations, contrast **enhancement**, and a **PDF report** generator — all through an interactive **Gradio** web interface.

> **⚠️ Research demo only.** This is a student project, not a medical device. It must not be used for clinical diagnosis.

---

## Features

| Feature | Description |
|---|---|
| **3-class tumor classification** | Glioma / Meningioma / Pituitary (Paper 1) |
| **Benign vs Malignant** | Binary classification with contrast enhancement (Paper 2) |
| **Grad-CAM heatmaps** | Visual explanation of what the model focuses on |
| **Enhancement lab** | Degrade a scan and watch how enhancement + classification cope |
| **Model showdown** | Compare all trained models side-by-side on one scan |
| **Metrics dashboard** | Accuracy, F1, AUC, confusion matrices, ROC curves |
| **Batch & PDF report** | Upload multiple scans → table + downloadable PDF |
| **CPU-first** | Runs on any machine — GPU is used automatically if available |

---

## Quick Start (Windows)

### 1. Prerequisites

- **Python 3.10 or 3.11** (recommended) — [download from python.org](https://www.python.org/downloads/)
- **Git** (optional, for cloning)

### 2. Setup

Open **Command Prompt** or **PowerShell** and run:

```powershell
# Navigate to the project folder
cd C:\Users\lenovo\.gemini\antigravity\scratch\NeuroLens

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train the Models

The first run downloads the dataset (~900 MB from figshare) and trains two lightweight CNN models. This takes roughly **5–15 minutes on CPU**.

```powershell
python train.py
```

Models are saved to `./models_cpu/` and **reused** on subsequent runs. To force retraining:

```powershell
python train.py --retrain
```

To also train the optional raw-scan ablation model:

```powershell
python train.py --ablation
```

### 4. Run Automated Tests

To verify all components (enhancement, model training, Grad-CAM, PDF reporting, Gradio UI) without downloading the full dataset:

```powershell
python test_neurolens.py
```

### 5. Launch the App

```powershell
python app.py
```

Then open **http://localhost:7860** in your browser.

To create a public shareable link:

```powershell
python app.py --share
```

---

## Dataset

The project uses the **Cheng et al. brain tumor MRI dataset** (3,064 T1 contrast-enhanced scans from 233 patients), hosted on [figshare](https://doi.org/10.6084/m9.figshare.1512427.v5).

- The dataset is **downloaded automatically** on first run and cached locally.
- A Kaggle mirror also works — download the images, organise them into `glioma/`, `meningioma/`, `pituitary/` sub-folders, and set the `NEUROLENS_DATASET_DIR` environment variable:

```powershell
set NEUROLENS_DATASET_DIR=C:\path\to\your\dataset
python train.py
```

---

## Project Structure

```
NeuroLens/
├── app.py                  # Gradio web interface (main entry point)
├── train.py                # Training script (run once)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── neurolens/              # Core library
│   ├── __init__.py
│   ├── config.py           # All settings and paths
│   ├── dataset.py          # Dataset download, loading, preprocessing
│   ├── enhance.py          # ODTWCHE contrast enhancement
│   └── models.py           # CNN builders, training, evaluation, Grad-CAM
├── models_cpu/             # Saved trained models and metrics (created at training)
├── examples/               # Sample MRI images for the UI (created at training)
└── venv/                   # Python virtual environment (created during setup)
```

---

## CPU vs GPU

- The code **automatically detects** whether a CUDA GPU is available.
- If no GPU is present, all computation runs on **CPU** — no configuration needed.
- The lightweight CNN (~15K parameters) is designed for fast CPU inference.
- Training uses a **stratified subset** of 1,500 scans and short epochs for speed.
- Heavy transfer-learning baselines (VGG16, ResNet50, etc.) are disabled by default.

---

## Configuration

All tuneable settings are in [`neurolens/config.py`](neurolens/config.py):

| Setting | Default | Description |
|---|---|---|
| `CPU_MAX_SCANS` | 1500 | Max scans to use (stratified subset) |
| `IMG` | 224 | Image size (width and height) |
| `BATCH` | 8 | Batch size for training |
| `EP_CUSTOM` | 3 | Epochs for Paper 1 model |
| `EP_P2` | 3 | Epochs for Paper 2 model |
| `RETRAIN` | False | Force retraining on next run |
| `RUN_ABLATION` | False | Train raw-scan ablation model |
| `RUN_5FOLD_CV` | False | Run 5-fold cross-validation |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `pip install` fails for TensorFlow | Make sure you're using Python 3.10 or 3.11 (not 3.13+). TensorFlow doesn't support the latest Python versions yet. |
| Dataset download fails | Download manually from [figshare](https://doi.org/10.6084/m9.figshare.1512427.v5), set `NEUROLENS_DATASET_DIR` |
| Out of memory during training | Reduce `CPU_MAX_SCANS` or `BATCH` in `config.py` |
| Port 7860 is in use | Run `python app.py --port 8080` |

---

## Credits

- **Paper 1:** Rasheed et al., *Brain Sciences*, 2023 — custom CNN for tumor classification
- **Paper 2:** Agarwal et al., *Results in Engineering*, 2024 — contrast enhancement + InceptionV3
- **Dataset:** Cheng et al., figshare (3,064 T1-CE MRI scans)
