# Coordinate Attention Experiment (YOLOv8-Seg)

This experiment evaluates **Coordinate Attention (CoordAtt)** integrated into **YOLOv8 segmentation** for ship instance segmentation on the HRSID dataset using **5-fold cross-validation**.

It includes:
- Custom implementation of:
  - `CoordAtt` (Coordinate Attention block)
- Modified YOLOv8-seg architecture defined via YAML
- A 5-Fold Cross-Validation training pipeline

Ultralytics does not provide Coordinate Attention natively, therefore this experiment registers the custom module at runtime so that it can be resolved by the YOLO model parser.

---

## Folder Structure

    experiments/coordinate_attention/
    └── README.md
    ├── coordinate_attention.py     # CoordAtt module implementation
    ├── requirements.txt     
    ├── train.py                    # 5-Fold Cross-Validation training script
    ├── yolov8-seg-ca.yaml          # YOLOv8-seg architecture with CoordAtt blocks
    
    

---

## Prepare the environment
Make sure you are in the right directory, use:
```
cd experiments/coordinate_attention
```
Install dependencies:
```
pip install -r requirements.txt
```
---

## Dataset Requirements

This experiment assumes that the dataset is **already converted** to YOLO **segmentation** format.

Expected directory:

    data/HRSID_YOLO_Format/

Required file:

    data/HRSID_YOLO_Format/HRSID_data.yaml

The YAML file must define:
- `path`
- `train`
- `val`
- `names`

Dataset conversion is handled in a separate pipeline and is not part of this experiment.

---

## Coordinate Attention Integration (Important)

Ultralytics does **not** recognize custom modules automatically.

To enable `CoordAtt` inside YOLO YAML parsing and model construction, the module is registered at runtime inside the training script.

Inside `train.py`, before model creation and **for every fold**:

    from coordinate_attention import CoordAtt
    import ultralytics.nn.modules as M
    import ultralytics.nn.tasks as T

    M.CoordAtt = CoordAtt
    T.CoordAtt = CoordAtt

This is required because:
- YAML parsing resolves layers via `ultralytics.nn.tasks`
- Module construction resolves layers via `ultralytics.nn.modules`
- A fresh YOLO model is instantiated for each cross-validation fold

---

## Model Initialization (Critical)

The model must be initialized from YAML, not from a `.pt` checkpoint:

    model = YOLO("yolov8-seg-ca.yaml")

Using a pretrained `.pt` model will not activate the Coordinate Attention blocks.

---

## Running the Experiment (5-Fold CV)

From the experiment root:

    python train_cv.py

This script will:
- Build 5 cross-validation folds
- Re-register `CoordAtt` before each fold
- Create a fresh YOLOv8-seg model per fold
- Train and validate each fold independently
- Aggregate final metrics across folds

---

## Outputs

Results are saved to:

    YOLO_CoordAt_CV/
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

The console reports mean ± standard deviation for:
- `mAP50-95_box`
- `mAP50_box`
- `mAP50-95_mask`
- `mAP50_mask`

---

## Experiment Purpose

This experiment evaluates whether Coordinate Attention improves spatial feature representation and ship segmentation performance compared to the YOLOv8-seg baseline on the HRSID dataset.


