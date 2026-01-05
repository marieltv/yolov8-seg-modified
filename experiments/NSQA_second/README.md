# YOLOv8 Hyperparameter Optimization & Cross-Validation Suite

This folder contains scripts for multi-objective hyperparameter optimization and cross-validation for YOLOv8 segmentation models. It is designed for high-performance training with automated evaluation, logging, and GPU memory management.

---

## Folder Structure

```
experiments/NSQA_second/
├── README.md
├── requirements.txt  
├── search.py                # Patched Ultralytics loss.py with Focal-Tversky integration
└── train.py                 # 5-Fold Cross-Validation training script

```

---
