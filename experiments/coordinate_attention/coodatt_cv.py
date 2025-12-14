"""
5-Fold Cross-Validation Training Script for YOLOv8-Seg with Coordinate Attention.

This script performs cross-validation training using a YOLOv8 segmentation model
augmented with a custom Coordinate Attention (CoordAtt) module.

Key characteristics:
- Uses a custom YOLO YAML model definition containing CoordAtt blocks
- Registers CoordAtt dynamically into Ultralytics runtime namespaces
- Trains a fresh model instance per fold
- Reports box and mask mAP metrics (mAP50 and mAP50–95)

Important:
- Ultralytics must be installed before running this script
- The CoordAtt module must be importable (coordinate_attention.py)
"""

import os
import yaml
import numpy as np
import gc
import torch
from typing import List, Dict
from sklearn.model_selection import KFold
from ultralytics import YOLO

from coordinate_attention import CoordAtt
import ultralytics.nn.modules as M
import ultralytics.nn.tasks as T


# -------------------------------------------------
# Configuration paths
# -------------------------------------------------
CONFIG_YAML_PATH: str = "/content/drive/MyDrive/Colab_Data/HRSID_YOLO_Format/HRSID_data.yaml"
output_dir: str = "/content/drive/MyDrive/Colab_Data/YOLO_CoordAt_CV"
os.makedirs(output_dir, exist_ok=True)


# -------------------------------------------------
# Load dataset configuration
# -------------------------------------------------
with open(CONFIG_YAML_PATH, "r") as f:
    external_config: Dict = yaml.safe_load(f)

class_names: List[str] = list(external_config["names"].values())
base_path: str = external_config.get("path", "")


# -------------------------------------------------
# Model configuration
# -------------------------------------------------
# IMPORTANT:
#   This must be a YAML model definition, NOT a pretrained .pt file.
model_path: str = "yolov8-seg-ca.yaml"


# -------------------------------------------------
# Training hyperparameters
# -------------------------------------------------
HYPERPARAMETERS: Dict[str, float] = {
    "epochs": 35,
    "batch": 16,
    "imgsz": 1024,
    "workers": 8,
    "patience": 15,
    "lr0": 0.02,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "copy_paste": 0.0,
}


def get_all_image_paths(config: Dict) -> List[str]:
    """
    Collect all image paths that have corresponding YOLO label files.

    Args:
        config (Dict): Dataset configuration loaded from data.yaml.

    Returns:
        List[str]: Sorted list of absolute image paths valid for training.
    """
    all_images: List[str] = []
    base_path: str = config.get("path", "")

    for split in ["train", "val"]:
        img_dir = os.path.join(base_path, config.get(split, ""))
        if not os.path.exists(img_dir):
            continue

        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                img_path = os.path.join(img_dir, fname)
                label_path = (
                    img_path.replace("/images/", "/labels/")
                    .rsplit(".", 1)[0] + ".txt"
                )

                if os.path.exists(label_path):
                    all_images.append(img_path)

    return sorted(list(set(all_images)))


def create_cv_yaml(
    output_path: str,
    train_paths: List[str],
    val_paths: List[str],
    class_names: List[str],
) -> None:
    """
    Create a fold-specific YOLO data.yaml file using image path lists.

    Args:
        output_path (str): Path where the fold YAML will be written.
        train_paths (List[str]): Training image paths for the fold.
        val_paths (List[str]): Validation image paths for the fold.
        class_names (List[str]): Dataset class names.
    """
    fold_dir = os.path.dirname(output_path)
    os.makedirs(fold_dir, exist_ok=True)

    train_txt = os.path.join(fold_dir, "train_cv.txt")
    val_txt = os.path.join(fold_dir, "val_cv.txt")

    with open(train_txt, "w") as f:
        f.write("\n".join(train_paths) + "\n")

    with open(val_txt, "w") as f:
        f.write("\n".join(val_paths) + "\n")

    data = {
        "train": train_txt,
        "val": val_txt,
        "names": {i: name for i, name in enumerate(class_names)},
    }

    with open(output_path, "w") as f:
        yaml.dump(data, f)


# =================================================
# Main execution
# =================================================
if __name__ == "__main__":
    all_images: List[str] = get_all_image_paths(external_config)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_metrics: Dict[str, List[float]] = {
        "mAP50-95_box": [],
        "mAP50_box": [],
        "mAP50-95_mask": [],
        "mAP50_mask": [],
        "model_paths": [],
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"\n======== FOLD {fold + 1}/5 ========")

        train_paths = [all_images[i] for i in train_idx]
        val_paths = [all_images[i] for i in val_idx]

        fold_yaml = os.path.join(output_dir, f"fold_{fold + 1}", "data.yaml")
        create_cv_yaml(fold_yaml, train_paths, val_paths, class_names)

        # -------------------------------------------------
        # Runtime registration of CoordAtt (REQUIRED)
        # -------------------------------------------------
        M.CoordAtt = CoordAtt
        T.CoordAtt = CoordAtt

        # IMPORTANT:
        # A fresh model must be created per fold
        model = YOLO(model_path)

        args = HYPERPARAMETERS.copy()
        args["data"] = fold_yaml
        args["project"] = output_dir
        args["name"] = f"fold_{fold + 1}"
        args["exist_ok"] = True

        try:
            print(f"Training fold {fold + 1}...")
            model.train(**args)

            print(f"Validating fold {fold + 1}...")
            res = model.val(data=fold_yaml)

            box_map: float = res.box.map
            box_map50: float = res.box.map50
            mask_map: float = res.seg.map
            mask_map50: float = res.seg.map50

            print(f"BOX  mAP50-95 = {box_map:.4f}")
            print(f"MASK mAP50-95 = {mask_map:.4f}")

            cv_metrics["mAP50-95_box"].append(box_map)
            cv_metrics["mAP50_box"].append(box_map50)
            cv_metrics["mAP50-95_mask"].append(mask_map)
            cv_metrics["mAP50_mask"].append(mask_map50)

            best_path = os.path.join(
                output_dir, f"fold_{fold + 1}", "weights", "best.pt"
            )
            if os.path.exists(best_path):
                cv_metrics["model_paths"].append(best_path)

        except Exception as e:
            print(f"ERROR IN FOLD {fold + 1}: {e}")
            cv_metrics["mAP50-95_box"].append(0.0)
            cv_metrics["mAP50_box"].append(0.0)
            cv_metrics["mAP50-95_mask"].append(0.0)
            cv_metrics["mAP50_mask"].append(0.0)

        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------------------------------
    # Final CV results
    # -------------------------------------------------
    print("\n===== FINAL RESULTS =====")
    for key in ["mAP50-95_box", "mAP50_box", "mAP50-95_mask", "mAP50_mask"]:
        arr = np.array(cv_metrics[key])
        print(f"{key}: {arr.mean():.4f} ± {arr.std():.4f}")

    best_idx: int = int(np.argmax(cv_metrics["mAP50-95_mask"]))
    print(
        f"\nBest fold: {best_idx + 1}, "
        f"model: {cv_metrics['model_paths'][best_idx]}"
    )
