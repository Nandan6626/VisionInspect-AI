# VisionInspect AI – Intelligent Industrial Quality Inspection (PatchCore + Anomalib)

VisionInspect AI is an industrial-grade visual quality inspection web application built with **Django**, **Anomalib PatchCore** (Wide ResNet-50-2 backbone), **OpenCLIP (ViT-B-32)**, **OpenCV**, and **ReportLab**.

It provides automated pre-screening to reject non-industrial photographs (such as selfies, buildings, landscapes, animals, food, etc.), runs deep memory-bank surface anomaly detection using PatchCore, visualizes pixel-level anomaly heatmaps, and generates audit-ready PDF inspection reports.

---

## 1. System Architecture & Key Capabilities

```
User Photograph
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Industrial Suitability Pre-Screen (OpenCLIP ViT-B-32)    │
│    • Evaluates semantic image categories                     │
│    • Checks Laplacian sharpness and illumination            │
└───────────────────────┬─────────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
    [INVALID / UNCERTAIN]           [VALID]
         │                             │
    Reject Photograph                  ▼
    (Display exact message)    ┌─────────────────────────────────────────┐
                               │ 2. PatchCore Anomaly Detection          │
                               │    • Backbone: Wide ResNet-50-2         │
                               │    • Multi-scale patch feature memory   │
                               │    • Computes continuous anomaly score  │
                               └───────────────────┬─────────────────────┘
                                                   │
                                                   ▼
                               ┌─────────────────────────────────────────┐
                               │ 3. Pixel-Level Anomaly Localization     │
                               │    • High-resolution JET heatmap        │
                               │    • [Original | Heatmap | Overlay]     │
                               └───────────────────┬─────────────────────┘
                                                   │
                                                   ▼
                               ┌─────────────────────────────────────────┐
                               │ 4. Quality Diagnosis & ReportLab PDF    │
                               │    • Normal vs. Potential Anomaly       │
                               │    • Certified PDF report download      │
                               └─────────────────────────────────────────┘
```

### 1.1 Pre-Screening Validation Layer (OpenCLIP)
Before any anomaly detection takes place, the system screens images against non-industrial categories:
- **Portraits / Selfies** (stops the previous issue where selfies were classified as defects)
- **Buildings & Architecture** (e.g., Burj Khalifa, houses)
- **Animals & Wildlife**
- **Food & Meals**
- **Vehicles & Everyday Objects**
- **Landscapes & Nature**
- **Cartoons & Digital Graphics**

**Exact User Responses:**
- Rejected: `"Invalid Image: The uploaded photograph does not appear to contain a supported industrial component. Please upload a clear photograph of a manufactured component."`
- Uncertain: `"Unable to verify this image for industrial inspection. Please upload another clear component photograph."`

### 1.2 PatchCore Anomaly Detection Engine
- Framework: **Anomalib 2.7.0**
- Backbone: Pretrained **Wide ResNet-50-2** extracting features at `layer2` and `layer3`.
- Memory Bank: Coreset subsampling of normal defect-free reference features.
- Output: Anomaly score $[0.0, 1.0]$, threshold comparison, and pixel-level 2D anomaly heatmap.

### 1.3 Pixel Heatmap & Multi-Modal Diagnostics
- Applies normalized JET color-mapping (Blue = Normal, Red/Yellow = Surface anomaly).
- Automatically stitches a 3-panel comparison image: `[Original Photograph | PatchCore Heatmap | Defect Overlay]`.

### 1.4 PDF Inspection Reports
- Generates industrial PDF reports using **ReportLab**.
- Includes inspection ID, audit timestamp, anomaly score, evaluation decision, side-by-side heatmap, and quality engineering disclaimer.

---

## 2. Directory Structure

```
VisionInspect-AI/
├── manage.py                          # Django management CLI
├── requirements.txt                   # All dependencies (Anomalib, OpenCLIP, Django, etc.)
├── README.md                          # Documentation
├── db.sqlite3                         # SQLite database with audit records
│
├── models/
│   ├── best.pt                        # Legacy YOLO classification model (preserved)
│   └── patchcore/                     # Active PatchCore model
│       ├── patchcore_model.ckpt       # Trained PatchCore checkpoint
│       └── config.json                # Model backbone and threshold config
│
├── datasets/
│   └── mvtec_grid/                    # Industrial training and test benchmark
│       ├── train/good/                # 10 defect-free normal training images
│       └── test/                      # Validation images (good and defective)
│
├── test_images/
│   ├── industrial_grid_good.png       # Sample normal grid component
│   ├── industrial_grid_defect.png     # Sample anomalous grid component
│   └── non_industrial_portrait.jpg    # Sample non-industrial portrait for testing validation
│
├── visioninspect/                     # Django root configuration
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── inspection/                        # Core inspection application
│   ├── models.py                      # Inspection audit database model
│   ├── views.py                       # Dashboard, Upload, Result, PDF, Analytics controllers
│   ├── urls.py                        # Web routing
│   ├── tests.py                       # Automated test suite
│   │
│   ├── services/
│   │   ├── image_validator.py         # OpenCLIP ViT-B-32 validation layer
│   │   ├── patchcore_service.py       # PatchCore inference service
│   │   ├── anomaly_visualizer.py      # Heatmap and side-by-side visualizer
│   │   └── pdf_generator.py           # ReportLab PDF generator
│   │
│   ├── training/
│   │   ├── prepare_dataset.py         # MVTec AD benchmark downloader
│   │   ├── train_patchcore.py         # PatchCore training script
│   │   └── evaluate_patchcore.py      # Model benchmark evaluation script
│   │
│   └── templates/inspection/
│       ├── base.html                  # Responsive navigation layout
│       ├── dashboard.html             # Industrial telemetry dashboard
│       ├── upload.html                # Drag-and-drop component upload
│       ├── result.html                # Diagnosis and heatmap viewer
│       ├── history.html               # Searchable inspection audit logs
│       └── analytics.html             # Operational metrics and charts
│
└── media/
    ├── inspections/                   # Uploaded component photographs
    └── heatmaps/                      # Generated anomaly heatmaps & comparison panels
```

---

## 3. Quick Start Guide

### 3.1 Start the Server

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run Django development server
python manage.py runserver 127.0.0.1:8000
```

Open your browser at: **http://127.0.0.1:8000/**

### 3.2 Testing the Application

#### Test 1: Upload a Non-Industrial Photograph (Validation Check)
1. Navigate to **http://127.0.0.1:8000/inspect/**.
2. Upload `test_images/non_industrial_portrait.jpg` (or any photo of a person, building, animal, food, car).
3. Click **"Inspect Component Now"**.
4. **Result:** The system identifies the non-industrial content and halts with:
   > *"Invalid Image: The uploaded photograph does not appear to contain a supported industrial component. Please upload a clear photograph of a manufactured component."*
   > *Identified non-industrial subject: Human Portrait / Selfie (99.83% confidence)*

#### Test 2: Upload a Normal Industrial Component
1. Go to **http://127.0.0.1:8000/inspect/**.
2. Upload `test_images/industrial_grid_good.png`.
3. Click **"Inspect Component Now"**.
4. **Result:** Passes pre-screening, runs PatchCore, displays anomaly score, and provides a downloadable PDF report.

#### Test 3: Upload a Defective Industrial Component
1. Go to **http://127.0.0.1:8000/inspect/**.
2. Upload `test_images/industrial_grid_defect.png`.
3. Click **"Inspect Component Now"**.
4. **Result:** Flags surface anomaly, renders the **Pixel Anomaly Heatmap** highlighting abnormal wire mesh regions in red/yellow, and provides a comparison view.

---

## 4. Retraining or Training on a New Industrial Category

To train PatchCore on a different industrial component category (e.g. `bottle`, `capsule`, `cable`, `metal_nut`):

```powershell
# Step 1: Prepare dataset
python inspection/training/prepare_dataset.py

# Step 2: Train PatchCore using normal samples only
python inspection/training/train_patchcore.py

# Step 3: Evaluate trained model on test samples
python inspection/training/evaluate_patchcore.py
```

---

## 5. Running Automated Tests

Run the full Django test suite to verify model loading, OpenCLIP validation, PatchCore inference, heatmap generation, and PDF reports:

```powershell
python manage.py test inspection
```
