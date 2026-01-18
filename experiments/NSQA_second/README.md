# YOLOv8 Hyperparameter Optimization & Cross-Validation Suite

This folder contains scripts for multi-objective hyperparameter optimization and cross-validation for YOLOv8 segmentation models. It is designed for high-performance training with automated evaluation, logging, and GPU memory management.
The optimization stage is intentionally decoupled from cross-validation to avoid bias and reduce computational cost.

---

## Folder Structure

```
experiments/NSQA_second/
├── README.md
├── requirements.txt  
├── search.py                # MOGA search
└── train.py                 # 5-Fold Cross-Validation training script

```

---

# Features

## MOGA / NSGA-II Hyperparameter Optimization (search.py)

- Implements NSGA-II / MOGA for multi-objective optimization.

- Optimizes key YOLOv8 hyperparameters:

  - Learning rate, optimizer parameters, and loss weights.

  - Data augmentations: color, geometric, flips, mosaic, mixup, copy-paste.

- Evaluates individuals based on mAP50-95 mask, FP, FN, and training time.

- Logs results to CSV for each generation.

- Supports Pareto front visualization.

## Cross-Validation Script (train.py)

- Performs k-fold cross-validation (default 5-fold) for YOLOv8 models.

- Generates fold-specific YAML configs dynamically.

- Trains and validates YOLOv8 models for each fold.

- Collects metrics:

  - Bounding box: mAP50-95_box, mAP50_box

  - Mask segmentation: mAP50-95_mask, mAP50_mask

- Automatically handles GPU memory cleanup after each fold.

- Saves best model paths per fold

- Extracts Top-5 models with best trade-offs.

---

# Usage

## Step 1. Prepare the environment
Make sure you are in the right directory, use:
  
```
cd experiments/NSQA_second
```
Install dependencies:
```
pip install -r requirements.txt
```
---
## Step 2. Multi-Objective Hyperparameter Optimization

```
python search.py
```
- Results saved in PROJECT_DIR (set in the script).

- Generates:

  - CSV log of all individuals per generation.

  - Pareto front plots: mAP vs Time, mAP vs FN, mAP vs FP.

  - JSON file with Top-5 models.

  Choose the model that best suits your objective from `top5_models.json`. Use its hyperparameters for next step - cross-validation.

## Step 3. Cross-Validation
Use the hyperparameters of the selected model for cross-validation. Adjust them in confugaration section in the file `train.py`. After that do :

```
python train.py
```
- Output directory: OUTPUT_DIR (set in the script).

- Generates:

  - Fold-specific YAML configs.

  - Trained YOLOv8 models for each fold.

  - Final cross-validation metrics summary.

  - Best model per fold path.

---

## Configuration

- Dataset YAML (CONFIG_YAML_PATH): YOLOv8 dataset file containing train, val, and names.

- Hyperparameters (HYPERPARAMETERS): Predefined for cross-validation or MOGA optimization.

Adjust paths and parameters at the top of the scripts before running. 

---

## Outputs

- PROJECT_DIR / OUTPUT_DIR

  - Logs, fold configs, trained weights, and evaluation results.

  - pareto_*.png (MOGA script)

  - top5_models.json (MOGA script)

  - train_cv.txt / val_cv.txt per fold (CV script)

- Console output summarizing metrics per generation/fold.

---
## Notes
- Scripts are designed for GPU training; ensure enough VRAM for imgsz and batch size.

- Uses Ultralytics YOLOv8 API (train/val).

- Automatic cleanup prevents memory leaks between folds/individual evaluations.

- The scripts are modular and can be extended for custom optimization or datasets.
