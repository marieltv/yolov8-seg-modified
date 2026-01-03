"""
YOLOv8 Cross-Validation Runner with Fixed Hyperparameters
========================================================

This script performs k-fold cross-validation on a YOLOv8 segmentation model
using a pre-defined set of hyperparameters. It generates fold-specific YAML
configs, trains and validates models, collects metrics, and manages GPU
memory efficiently.

Author: —
"""

from __future__ import annotations

import os
import gc
import yaml
import torch
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import KFold
from typing import Dict, List


# ============================================================
# Configuration
# ============================================================

CONFIG_YAML_PATH: str = "data/HRSID_YOLO_Format/HRSID_data.yaml"
OUTPUT_DIR: str = "data/MOGA/YOLO_MOGA_CV_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CONFIG_YAML_PATH, "r") as f:
    external_config: Dict = yaml.safe_load(f)

CLASS_NAMES: List[str] = list(external_config["names"].values())
BASE_PATH: str = external_config.get("path", "")

MODEL_TYPE: str = "yolov8n-seg.pt"

HYPERPARAMETERS: Dict[str, float] = {
    "epochs": 35,
    "batch": 16,
    "imgsz": 1024,
    "workers": 8,
    "patience": 15,
    "lr0": 0.0133384406,
    "lrf": 0.0943503125,
    "momentum": 0.9363948701,
    "weight_decay": 0.0006245278,
    "warmup_epochs": 4.5641236178,
    "warmup_momentum": 0.9,
    "box": 6.873444535,
    "cls": 0.4523117117,
    "dfl": 1.5006213836,
    "hsv_h": 0.0036966943,
    "hsv_s": 0.3544652367,
    "hsv_v": 0.1243070762,
    "degrees": 7.6029454039,
    "translate": 0.0529742432,
    "scale": 0.6356195046,
    "shear": 1.3605850179,
    "perspective": 0.0001125202,
    "fliplr": 0.4155649185,
    "flipud": 0.0140880466,
    "mosaic": 0.675288344,
    "mixup": 0.023213606,
    "copy_paste": 0.3,
}


# ============================================================
# Utilities
# ============================================================

def get_all_image_paths(config: Dict) -> List[str]:
    """Collect all valid image-label pairs from the dataset."""
    all_images: List[str] = []
    base_path = config.get("path", "")

    for split in ["train", "val"]:
        split_dir = config.get(split)
        if not split_dir:
            continue
        img_dir = os.path.join(base_path, split_dir)
        if not os.path.exists(img_dir):
            print(f"[WARN] Directory not found: {img_dir}")
            continue

        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                img_path = os.path.join(img_dir, fname)
                label_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                if os.path.exists(label_path):
                    all_images.append(img_path)
                else:
                    print(f"[WARN] Missing label: {fname}")

    all_images = sorted(list(set(all_images)))
    print(f"✓ Found {len(all_images)} valid image-label pairs.")
    return all_images


def create_cv_yaml(
    output_path: str,
    train_paths: List[str],
    val_paths: List[str],
    class_names: List[str],
    base_path: str,
) -> None:
    """Create YAML config file for the current CV fold."""
    fold_dir = os.path.dirname(output_path)
    os.makedirs(fold_dir, exist_ok=True)

    train_txt = os.path.join(fold_dir, "train_cv.txt")
    val_txt = os.path.join(fold_dir, "val_cv.txt")

    with open(train_txt, "w") as f:
        f.write("\n".join(train_paths) + "\n")
    with open(val_txt, "w") as f:
        f.write("\n".join(val_paths) + "\n")

    data = {
        "train": os.path.abspath(train_txt),
        "val": os.path.abspath(val_txt),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    with open(output_path, "w") as f:
        yaml.dump(data, f)

    print(f"✔ Created fold YAML: {output_path}")


# ============================================================
# Main Cross-Validation
# ============================================================

if __name__ == "__main__":
    all_images = get_all_image_paths(external_config)
    if not all_images:
        raise ValueError("No valid images found in the dataset.")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_metrics: Dict[str, List[float]] = {
        "mAP50-95_box": [],
        "mAP50_box": [],
        "mAP50-95_mask": [],
        "mAP50_mask": [],
        "model_paths": [],
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images), start=1):
        print(f"\n{'='*50}\n       FOLD {fold}/5\n{'='*50}")

        train_paths = [all_images[i] for i in train_idx]
        val_paths = [all_images[i] for i in val_idx]
        print(f"Train={len(train_paths)}  |  Val={len(val_paths)}")

        fold_yaml = os.path.join(OUTPUT_DIR, f"fold_{fold}", "data.yaml")
        create_cv_yaml(fold_yaml, train_paths, val_paths, CLASS_NAMES, BASE_PATH)

        model = YOLO(MODEL_TYPE)
        args = HYPERPARAMETERS.copy()
        args.update({
            "data": fold_yaml,
            "project": OUTPUT_DIR,
            "name": f"fold_{fold}",
            "verbose": True,
            "exist_ok": True,
        })

        try:
            print(f"\nStarting training fold {fold}...")
            model.train(**args)

            print(f"\nValidating fold {fold}...")
            val_results = model.val(data=fold_yaml)

            # Extract metrics
            box_map50_95 = getattr(getattr(val_results, "box", None), "map", 0.0)
            box_map50 = getattr(getattr(val_results, "box", None), "map50", 0.0)
            mask_map50_95 = getattr(getattr(val_results, "seg", None), "map", 0.0)
            mask_map50 = getattr(getattr(val_results, "seg", None), "map50", 0.0)

            # Fallback to results dict
            if box_map50_95 == 0.0 and mask_map50_95 == 0.0:
                metrics_dict = val_results.results_dict
                box_map50_95 = metrics_dict.get("metrics/mAP50-95(B)", 0.0)
                box_map50 = metrics_dict.get("metrics/mAP50(B)", 0.0)
                mask_map50_95 = metrics_dict.get("metrics/mAP50-95(M)", 0.0)
                mask_map50 = metrics_dict.get("metrics/mAP50(M)", 0.0)

            print(f"\nBOX  → mAP50-95: {box_map50_95:.4f}, mAP50: {box_map50:.4f}")
            print(f"MASK → mAP50-95: {mask_map50_95:.4f}, mAP50: {mask_map50:.4f}")

            cv_metrics["mAP50-95_box"].append(box_map50_95)
            cv_metrics["mAP50_box"].append(box_map50)
            cv_metrics["mAP50-95_mask"].append(mask_map50_95)
            cv_metrics["mAP50_mask"].append(mask_map50)

            best_model_path = os.path.join(OUTPUT_DIR, f"fold_{fold}", "weights", "best.pt")
            if os.path.exists(best_model_path):
                cv_metrics["model_paths"].append(best_model_path)

        except Exception as e:
            print(f"[ERROR] Fold {fold}: {e}")
            import traceback
            traceback.print_exc()

            cv_metrics["mAP50-95_box"].append(0.0)
            cv_metrics["mAP50_box"].append(0.0)
            cv_metrics["mAP50-95_mask"].append(0.0)
            cv_metrics["mAP50_mask"].append(0.0)

        finally:
            del model
            if "val_results" in locals():
                del val_results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ============================================================
    # Final CV Summary
    # ============================================================

    print(f"\n{'='*60}\n           CROSS-VALIDATION RESULTS\n{'='*60}")
    for key in ["mAP50-95_box", "mAP50_box", "mAP50-95_mask", "mAP50_mask"]:
        values = cv_metrics[key]
        if values:
            print(f"{key:20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    if cv_metrics["mAP50-95_mask"]:
        best_fold_idx = int(np.argmax(cv_metrics["mAP50-95_mask"]))
        print(f"\nBest fold: {best_fold_idx + 1} (mAP50-95 mask = {cv_metrics['mAP50-95_mask'][best_fold_idx]:.4f})")
        if len(cv_metrics["model_paths"]) > best_fold_idx:
            print(f"Model path: {cv_metrics['model_paths'][best_fold_idx]}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")
