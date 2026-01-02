# Baseline YOLOv8-Seg Cross-Validation Experiment

This experiment evaluates the **baseline YOLOv8 segmentation model** on the HRSID ship dataset
using **5-Fold Cross-Validation** without any architectural or loss-function modifications.

It serves as the **reference baseline** for all subsequent experiments
(Coordinate Attention, Tversky / Focal-Tversky loss).

---

## Folder Structure
```
experiments/base_cv/
├── base_cv.py          # 5-Fold Cross-Validation training script (baseline)
└── README.md
```
The dataset is expected to be stored outside this folder.

---

## Requirements

`pip install ultralytics`

Python 3.9+ recommended.  
CUDA-enabled PyTorch is strongly recommended.

---

## Dataset Requirements

This experiment assumes that the dataset is **already converted**
to YOLO **segmentation** format.

Expected directory:

`data/HRSID_YOLO_Format/`

Required file:

`data/HRSID_YOLO_Format/HRSID_data.yaml`

The YAML file must define:
- `path`
- `train`
- `val`
- `names`

Dataset conversion is handled in a separate pipeline and is not part of this experiment.

---

## Model Configuration

The baseline model uses the **official pretrained YOLOv8 segmentation checkpoint**:

`yolov8n-seg.pt`

No architectural changes, custom layers, or loss modifications are applied.

---

## Cross-Validation Setup

- 5-fold cross-validation (KFold, shuffle=True, random_state=42)
- Each fold:
  - Creates a fold-specific data.yaml
  - Trains a fresh YOLOv8 model
  - Performs validation on the held-out split
- Metrics are aggregated across folds

---

## Running the Experiment

From the repository root:

`python experiments/base_cv/train_cv.py`

---

## Outputs

Results are saved to:
```
data/YOLO_CV_Results/
├── fold_1/
│   └── weights/best.pt
├── fold_2/
│   └── weights/best.pt
├── fold_3/
│   └── weights/best.pt
├── fold_4/
│   └── weights/best.pt
└── fold_5/
    └── weights/best.pt
```
The console reports **mean ± standard deviation** across folds for:
- mAP50-95_box
- mAP50_box
- mAP50-95_mask
- mAP50_mask

---

## Notes

- No Ultralytics source files are modified
- No custom modules are registered
- No loss functions are patched
- Each fold uses a fresh model instance
- Fully reproducible from GitHub

---

## Experiment Purpose

This experiment establishes a **clean and reproducible baseline**
for ship instance segmentation on the HRSID dataset.

All subsequent experiments are compared against these results
to assess the impact of architectural and loss-function modifications.
