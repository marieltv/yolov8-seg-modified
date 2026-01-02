import os
import yaml
import numpy as np
import gc
import torch
from typing import List, Dict, Tuple
from ultralytics import YOLO
from sklearn.model_selection import KFold
from ultralytics.utils.losses_tversky import FocalTverskyLoss


CONFIG_YAML_PATH: str = "/content/drive/MyDrive/Colab_Data/HRSID_YOLO_Format/HRSID_data.yaml"
output_dir: str = "/content/drive/MyDrive/Colab_Data/YOLO_Tversky_CV"
os.makedirs(output_dir, exist_ok=True)

with open(CONFIG_YAML_PATH, "r") as f:
    external_config = yaml.safe_load(f)

class_names: List[str] = list(external_config["names"].values())
base_path: str = external_config.get("path", "")

model_type: str = "yolov8n-seg.pt"

# Standard YOLO hyperparameters for segmentation + augmentation
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
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "copy_paste": 0.0,
}


def get_all_image_paths(config: dict) -> List[str]:
    """
    Collect all dataset image paths that have valid label files.

    Args:
        config (dict): Parsed dataset YAML dictionary containing
            paths to train/val directories.

    Returns:
        List[str]: Sorted list of full paths to training images
            for use in cross-validation.

    Notes:
        - Only images with corresponding YOLO `.txt` labels are included.
        - Missing label warnings are printed for debugging.
    """
    all_images: List[str] = []
    base_path: str = config.get("path", "")

    for split in ["train", "val"]:
        rel = config.get(split)
        if rel is None:
            continue

        img_dir = os.path.join(base_path, rel)
        if not os.path.exists(img_dir):
            print(f"[WARN] Not found: {img_dir}")
            continue

        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                img_path = os.path.join(img_dir, f)

                # Verify YOLO label exists
                label_path = (
                    img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                )
                if os.path.exists(label_path):
                    all_images.append(img_path)
                else:
                    print(f"[WARN] Missing label for: {f}")

    all_images = sorted(list(set(all_images)))
    print(f" Found {len(all_images)} photos for cv.")
    return all_images


def create_cv_yaml(
    output_path: str,
    train_paths: List[str],
    val_paths: List[str],
    class_names: List[str],
    base_path: str,
) -> None:
    """
    Create a fold-specific YOLO `data.yaml` file for cross-validation.

    Writes:
        - `train_cv.txt` containing absolute paths to training images
        - `val_cv.txt` containing absolute paths to validation images
        - `data.yaml` referencing these lists

    Args:
        output_path (str): Path to the fold-specific YAML file to write.
        train_paths (List[str]): Image paths for this fold's training set.
        val_paths (List[str]): Image paths for this fold's validation set.
        class_names (List[str]): Names of dataset classes.
        base_path (str): Root directory of dataset (unused but kept for clarity).
    """
    fold_dir = os.path.dirname(output_path)
    os.makedirs(fold_dir, exist_ok=True)

    train_txt = os.path.join(fold_dir, "train_cv.txt")
    val_txt = os.path.join(fold_dir, "val_cv.txt")

    # Save lists
    with open(train_txt, "w") as f:
        f.write("\n".join(train_paths) + "\n")
    with open(val_txt, "w") as f:
        f.write("\n".join(val_paths) + "\n")

    # YAML dictionary
    data = {
        "train": os.path.abspath(train_txt),
        "val": os.path.abspath(val_txt),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    with open(output_path, "w") as f:
        yaml.dump(data, f)

    print(f"✔ Created fold config: {output_path}")


# ===============================
#           MAIN LOOP
# ===============================
if __name__ == "__main__":
    all_images: List[str] = get_all_image_paths(external_config)

    if len(all_images) == 0:
        raise ValueError("There are no valid images for cv!")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_metrics: Dict[str, List[float]] = {
        "mAP50-95_box": [],
        "mAP50_box": [],
        "mAP50-95_mask": [],
        "mAP50_mask": [],
        "model_paths": [],
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        # Reset FocalTversky print-once flag
        FocalTverskyLoss.initialized = False

        print("\n" + "=" * 50)
        print(f"       Fold {fold + 1}/5")
        print("=" * 50)

        train_paths = [all_images[i] for i in train_idx]
        val_paths = [all_images[i] for i in val_idx]
        print(f" Train={len(train_paths)}  |  Val={len(val_paths)}")

        # Create fold's data.yaml
        fold_yaml = os.path.join(output_dir, f"fold_{fold + 1}", "data.yaml")
        create_cv_yaml(fold_yaml, train_paths, val_paths, class_names, base_path)

        # Load model fresh each fold
        model = YOLO(model_type)

        # Training args
        args = HYPERPARAMETERS.copy()
        args["data"] = fold_yaml
        args["project"] = output_dir
        args["name"] = f"fold_{fold + 1}"
        args["verbose"] = True
        args["exist_ok"] = True

        try:
            print(f"\n Start of training fold number {fold + 1}...")
            results = model.train(**args)

            print(f"\n Validation of fold number {fold + 1}...")
            val_results = model.val(data=fold_yaml)

            # Extract metrics
            try:
                # New YOLOv8-friendly access
                box_map50_95 = val_results.box.map if hasattr(val_results, "box") else 0
                box_map50 = val_results.box.map50 if hasattr(val_results, "box") else 0

                mask_map50_95 = val_results.seg.map if hasattr(val_results, "seg") else 0
                mask_map50 = val_results.seg.map50 if hasattr(val_results, "seg") else 0

            except AttributeError:
                # Fallback for dict-style metrics
                metrics_dict = val_results.results_dict
                box_map50_95 = metrics_dict.get("metrics/mAP50-95(B)", 0.0)
                box_map50 = metrics_dict.get("metrics/mAP50(B)", 0.0)
                mask_map50_95 = metrics_dict.get("metrics/mAP50-95(M)", 0.0)
                mask_map50 = metrics_dict.get("metrics/mAP50(M)", 0.0)

            print(f"\n BOX  → mAP50-95: {box_map50_95:.4f}, mAP50: {box_map50:.4f}")
            print(f" MASK → mAP50-95: {mask_map50_95:.4f}, mAP50: {mask_map50:.4f}")

            # Store metrics
            cv_metrics["mAP50-95_box"].append(box_map50_95)
            cv_metrics["mAP50_box"].append(box_map50)
            cv_metrics["mAP50-95_mask"].append(mask_map50_95)
            cv_metrics["mAP50_mask"].append(mask_map50)

            # Save model path
            best_model = os.path.join(
                output_dir, f"fold_{fold + 1}", "weights", "best.pt"
            )
            if os.path.exists(best_model):
                cv_metrics["model_paths"].append(best_model)

        except Exception as e:
            print(f" There is an error in fold number {fold + 1}: {e}")
            import traceback

            traceback.print_exc()

            # Insert placeholders for failed fold
            cv_metrics["mAP50-95_box"].append(0.0)
            cv_metrics["mAP50_box"].append(0.0)
            cv_metrics["mAP50-95_mask"].append(0.0)
            cv_metrics["mAP50_mask"].append(0.0)

    # ===============================
    #         FINAL RESULTS
    # ===============================
    print("\n" + "=" * 60)
    print("           Final CROSS-VALIDATION Results")
    print("=" * 60)

    for key in ["mAP50-95_box", "mAP50_box", "mAP50-95_mask", "mAP50_mask"]:
        values = cv_metrics[key]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key:20s}: {mean_val:.4f} ± {std_val:.4f}")

    # Best fold (based on segmentation mAP50-95)
    if cv_metrics["mAP50-95_mask"]:
        best_fold_idx = int(np.argmax(cv_metrics["mAP50-95_mask"]))
        print(
            f"\n The best fold: {best_fold_idx + 1} "
            f"(mAP50-95 mask = {cv_metrics['mAP50-95_mask'][best_fold_idx]:.4f})"
        )
        if len(cv_metrics["model_paths"]) > best_fold_idx:
            print(f" Model: {cv_metrics['model_paths'][best_fold_idx]}")

    print(f"\n All results saved in: {output_dir}")
