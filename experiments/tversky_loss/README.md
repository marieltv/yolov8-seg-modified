# Tversky & Focal-Tversky Loss Experiment (YOLOv8-Seg)

This experiment evaluates Tversky Loss and Focal-Tversky Loss for ship instance segmentation using YOLOv8-seg.

It includes:

- Custom implementation of:
  - `TverskyLoss`
  - `FocalTverskyLoss`
- A patched YOLO segmentation loss that blends:
  - Binary Cross-Entropy (BCE)
  - Focal-Tversky Loss
- A 5-Fold Cross-Validation training pipeline

Ultralytics does not support Focal-Tversky natively, therefore this experiment requires patching the internal Ultralytics loss system before training.

---

## Folder Structure

```
experiments/tversky/
├── losses_tversky.py        # Custom Tversky & Focal-Tversky losses
├── modified_loss.py         # Patched Ultralytics loss.py with Focal-Tversky integration
├── install_patches.py       # Automatic installer for this experiment
├── train_cv.py              # 5-Fold Cross-Validation training script
└── README.md
```

---

## Requirements

```
pip install ultralytics
```

Python 3.9+ recommended.

---

## Step 1 — Install Custom Loss Patch (Required)

Run the installer once before training:

```
python experiments/tversky/install_patches.py
```

This script performs:

1. Copies:
   `experiments/tversky/losses_tversky.py` → `ultralytics/utils/losses_tversky.py`

2. Creates a backup of the original YOLO loss:
   `ultralytics/utils/loss.py` → `loss_original.py`

3. Replaces the active YOLO loss with the modified version:
   `modified_loss.py` → `ultralytics/utils/loss.py`

After installation, Ultralytics will use the patched Tversky / Focal-Tversky mask loss automatically.

---

## Step 2 — Run the Training Script (5-Fold CV)

```
python experiments/tversky/train_cv.py
```

This will:

- Train YOLOv8-seg using BCE + Focal-Tversky blended mask loss
- Perform 5-fold cross-validation
- Save:
  - metrics per fold
  - best model weights
- Print mean ± std for:
  - mAP50-95 (box)
  - mAP50 (box)
  - mAP50-95 (mask)
  - mAP50 (mask)

---

## Restoring Original YOLO Loss (Optional)

To undo the patch:

Rename the backup file:

```
ultralytics/utils/loss_original.py → loss.py
```

This restores default YOLO loss behavior.

---

## Notes

- Run the installer only once per environment.
- Re-run only if:
  - Ultralytics is reinstalled
  - Python environment changes
- Do not run the installer inside the training loop.
- Do not mix experiments without restoring the original loss.

---

## Experiment Purpose

This experiment evaluates whether Focal-Tversky improves ship segmentation performance on the HRSID dataset compared to the YOLOv8-seg baseline.
