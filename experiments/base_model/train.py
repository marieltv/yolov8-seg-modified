import os
import yaml
import numpy as np
import gc
import torch
from typing import Dict, List, Any
from ultralytics import YOLO
from sklearn.model_selection import KFold


# =========================
# CONFIGURATION
# =========================

CONFIG_YAML_PATH: str = "data/HRSID_YOLO_Format/HRSID_data.yaml"
output_dir: str = "data/YOLO_CV_Results"
os.makedirs(output_dir, exist_ok=True)

with open(CONFIG_YAML_PATH, "r") as f:
    external_config: Dict[str, Any] = yaml.safe_load(f)

class_names: List[str] = list(external_config["names"].values())
base_path: str = external_config.get("path", "")

model_type: str = "yolov8n-seg.pt"

HYPERPARAMETERS: Dict[str, Any] = {
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


# =========================
# DATA COLLECTION
# =========================

def get_all_image_paths(config: Dict[str, Any]) -> List[str]:
    """
    Collects all valid image–label pairs from YOLO dataset configuration.

    Args:
        config: Parsed data.yaml configuration.

    Returns:
        Sorted list of valid image paths.
    """
    all_images: List[str] = []
    base_path: str = config.get("path", "")

    for split in ["train", "val"]:
        rel = config.get(split)
        if rel is None:
            continue

        img_dir = os.path.join(base_path, rel)
        if not os.path.exists(img_dir):
            print(f"[WARN] Directory not found: {img_dir}")
            continue

        for f in os.listdir(img_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                img_path = os.path.join(img_dir, f)
                label_path = (
                    img_path
                    .replace("/images/", "/labels/")
                    .rsplit(".", 1)[0] + ".txt"
                )

                if os.path.exists(label_path):
                    all_images.append(img_path)
                else:
                    print(f"[WARN] Missing label for: {f}")

    all_images = sorted(list(set(all_images)))
    print(f"✓ Found {len(all_images)} valid image–label pairs.")
    return all_images


# =========================
# YAML CREATION FOR FOLDS
# =========================

def create_cv_yaml(
    output_path: str,
    train_paths: List[str],
    val_paths: List[str],
    class_names: List[str],
    base_path: str,
) -> None:
    """
    Creates YOLO-compatible YAML configuration for a single CV fold.

    Args:
        output_path: Path to save the fold YAML file.
        train_paths: List of training image paths.
        val_paths: List of validation image paths.
        class_names: List of class names.
        base_path: Dataset base directory.
    """
    fold_dir = os.path.dirname(output_path)
    os.makedirs(fold_dir, exist_ok=True)

    train_txt = os.path.join(fold_dir, "train_cv.txt")
    val_txt = os.path.join(fold_dir, "val_cv.txt")

    with open(train_txt, "w") as f:
        f.write("\n".join(train_paths) + "\n")

    with open(val_txt, "w") as f:
        f.write("\n".join(val_paths) + "\n")

    data: Dict[str, Any] = {
        "train": os.path.abspath(train_txt),
        "val": os.path.abspath(val_txt),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✔ Fold config created: {output_path}")


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":

    all_images: List[str] = get_all_image_paths(external_config)

    if len(all_images) == 0:
        raise ValueError("No valid images found!")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_metrics: Dict[str, List[float | str]] = {
        "mAP50-95_box": [],
        "mAP50_box": [],
        "mAP50-95_mask": [],
        "mAP50_mask": [],
        "model_paths": [],
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"\n{'='*50}")
        print(f"       FOLD {fold + 1}/5")
        print(f"{'='*50}")

        train_paths: List[str] = [all_images[i] for i in train_idx]
        val_paths: List[str] = [all_images[i] for i in val_idx]
        print(f"📊 Train={len(train_paths)}  |  Val={len(val_paths)}")

        fold_yaml = os.path.join(output_dir, f"fold_{fold+1}", "data.yaml")
        create_cv_yaml(fold_yaml, train_paths, val_paths, class_names, base_path)

        model = YOLO(model_type)

        args: Dict[str, Any] = HYPERPARAMETERS.copy()
        args["data"] = fold_yaml
        args["project"] = output_dir
        args["name"] = f"fold_{fold+1}"
        args["verbose"] = True
        args["exist_ok"] = True

        try:
            print(f"\n Training fold {fold+1}...")
            results = model.train(**args)

            print(f"\n Validating fold {fold+1}...")
            val_results = model.val(data=fold_yaml)

            try:
                box_map50_95 = val_results.box.map
                box_map50 = val_results.box.map50
                mask_map50_95 = val_results.seg.map
                mask_map50 = val_results.seg.map50
            except AttributeError:
                metrics_dict = val_results.results_dict
                box_map50_95 = metrics_dict.get("metrics/mAP50-95(B)", 0.0)
                box_map50 = metrics_dict.get("metrics/mAP50(B)", 0.0)
                mask_map50_95 = metrics_dict.get("metrics/mAP50-95(M)", 0.0)
                mask_map50 = metrics_dict.get("metrics/mAP50(M)", 0.0)

            print(f"\n BOX  → mAP50-95: {box_map50_95:.4f}, mAP50: {box_map50:.4f}")
            print(f" MASK → mAP50-95: {mask_map50_95:.4f}, mAP50: {mask_map50:.4f}")

            cv_metrics["mAP50-95_box"].append(box_map50_95)
            cv_metrics["mAP50_box"].append(box_map50)
            cv_metrics["mAP50-95_mask"].append(mask_map50_95)
            cv_metrics["mAP50_mask"].append(mask_map50)

            best_model = os.path.join(
                output_dir, f"fold_{fold+1}", "weights", "best.pt"
            )
            if os.path.exists(best_model):
                cv_metrics["model_paths"].append(best_model)

        except Exception as e:
            print(f" Error in fold {fold+1}: {e}")
            cv_metrics["mAP50-95_box"].append(0.0)
            cv_metrics["mAP50_box"].append(0.0)
            cv_metrics["mAP50-95_mask"].append(0.0)
            cv_metrics["mAP50_mask"].append(0.0)

        finally:
            del model
            if "results" in locals():
                del results
            if "val_results" in locals():
                del val_results

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("           FINAL CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")

    for key in ["mAP50-95_box", "mAP50_box", "mAP50-95_mask", "mAP50_mask"]:
        values = cv_metrics[key]
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key:20s}: {mean_val:.4f} ± {std_val:.4f}")

    if len(cv_metrics["mAP50-95_mask"]) > 0:
        best_fold_idx = int(np.argmax(cv_metrics["mAP50-95_mask"]))
        print(
            f"\n Best fold: {best_fold_idx + 1} "
            f"(mAP50-95 mask = {cv_metrics['mAP50-95_mask'][best_fold_idx]:.4f})"
        )
        if len(cv_metrics["model_paths"]) > best_fold_idx:
            print(f" Model: {cv_metrics['model_paths'][best_fold_idx]}")

    print(f"\n All results saved in: {output_dir}")
