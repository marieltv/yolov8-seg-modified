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
experiments/tversky_loss/
├── README.md
├── install_patches.py       # Automatic installer for this experiment
├── losses_tversky.py        # Custom Tversky & Focal-Tversky losses
├── modified_loss.py         # Patched Ultralytics loss.py with Focal-Tversky integration
├── requirements.txt          
└── train.py                 # 5-Fold Cross-Validation training script

```

---

## Step 1- Prepare the environment
Make sure you are in the right directory, use:
```
cd experiments/tversky_loss
```
Install dependencies:
```
pip install -r requirements.txt
```
Python 3.9+ recommended.

---

## Step 2 — Install Custom Loss Patch (Required)

Run the installer once before training:

```
python install_patches.py
```

This script performs:

1. Copies:
   `experiments/tversky_loss/losses_tversky.py` → `ultralytics/utils/losses_tversky.py`

2. Creates a backup of the original YOLO loss:
   `ultralytics/utils/loss.py` → `loss_original.py`

3. Replaces the active YOLO loss with the modified version:
   `modified_loss.py` → `ultralytics/utils/loss.py`

After installation, Ultralytics will use the patched Tversky / Focal-Tversky mask loss automatically.

---

## Step 3 — Run the Training Script (5-Fold CV)

```
python train.py
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
## Different parameters

`Important`: for tversky_loss we need to define 3 parameters 

- &alpha; (alpha) - the higher this value, the more the model penalizes false positives (cases where the model detects a ship that is not actually present).
- &beta; (beta) - the higher this value, the more the model penalizes false negatives (cases where the model fails to detect a ship that is actually present).
- &gamma; (gamma) -  this parameter increases the importance of 'hard' examples, such as small ships, partially overlapping objects, and low-contrast cases. When &gamma; > 1, the model focuses more heavily on the specific pixels where its predictions are most inaccurate; at this point, the Tversky loss effectively becomes Focal Tversky loss.

  The sum of &alpha; and &beta; must be 1. You can adjust parameters (as well as what you want prioritize more bce or our tversky loss) in file `modified_loss.py`, you must search lines 315-317:
  ```
  def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

         # --- Add Focal Tversky ---
        self.ft = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=2)
        self.lambda_bce = 0.5
        self.lambda_ft = 0.5
  ```

And you must make the same adjustments in `losses_tversky.py` in lines 21-21

```
 def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1e-6,
```

and lines 79-81:

```
 def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 2.0,
        smooth: float = 1e-6,
```

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
