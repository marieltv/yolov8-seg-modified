import json
import os
import shutil
import random
from typing import List, Dict
from tqdm import tqdm  # Progress bar
import yaml

# Root directory after extracting HRSID_JPG.zip
hrs_id_root_dir: str = "data/HRSID_raw/HRSID_JPG"

# Correct image directory
hrs_id_images_dir: str = os.path.join(hrs_id_root_dir, "JPEGImages")

# Correct annotation JSON paths
hrs_id_train_annotations_json: str = os.path.join(
    hrs_id_root_dir, "annotations", "train2017.json"
)
hrs_id_test_annotations_json: str = os.path.join(
    hrs_id_root_dir, "annotations", "test2017.json"
)

# Output directory for YOLO-formatted dataset
yolo_output_dir: str = "data/HRSID_YOLO_Format"

# Path to dynamic data.yaml
dynamic_data_yaml_path: str = os.path.join(yolo_output_dir, "HRSID_data.yaml")

# Fixed number of images per split
train_count: int = 800
val_count: int = 200
test_count: int = 200

# COCO category IDs start from 1, but YOLO uses 0-based indexing
class_names: List[str] = ["Ship"]  # HRSID has only one class (ID=1 in COCO)


def create_dynamic_data_yaml(
    output_path: str,
    train_img_dir: str,
    val_img_dir: str,
    test_img_dir: str,
    class_names: List[str]
) -> None:
    """
    Creates a YOLO-compatible data.yaml configuration file.
    """

    base_path_safe = os.path.dirname(output_path)

    data: Dict = {
        "path": base_path_safe,
        "train": os.path.relpath(train_img_dir, base_path_safe),
        "val": os.path.relpath(val_img_dir, base_path_safe),
        "test": os.path.relpath(test_img_dir, base_path_safe),
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)}
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def convert_coco_to_yolo_segmentation(
    json_path: str,
    images_dir: str,
    output_labels_dir: str
) -> List[str]:
    """
    Converts COCO JSON annotations into YOLO polygon segmentation format.

    Returns:
        A list of image filenames that contain valid annotations.
    """

    print(f"Starting COCO to YOLO conversion: {json_path}")
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    image_id_to_filename = {
        img["id"]: img["file_name"] for img in coco_data["images"]
    }
    image_id_to_size = {
        img["id"]: (img["width"], img["height"]) for img in coco_data["images"]
    }

    annotations_by_image: Dict[str, List[Dict]] = {}

    for ann in coco_data["annotations"]:
        image_filename = image_id_to_filename.get(ann["image_id"])
        if image_filename:
            annotations_by_image.setdefault(image_filename, []).append(ann)

    os.makedirs(output_labels_dir, exist_ok=True)

    annotated_image_filenames: List[str] = []

    for image_filename, annotations in tqdm(
        annotations_by_image.items(), desc="Converting annotations"
    ):
        img_width, img_height = image_id_to_size[annotations[0]["image_id"]]
        label_file_path = os.path.join(
            output_labels_dir,
            os.path.splitext(image_filename)[0] + ".txt"
        )

        has_annotations: bool = False

        with open(label_file_path, "w") as f:
            for ann in annotations:
                category_id: int = ann["category_id"] - 1

                segmentation = ann["segmentation"]

                if (
                    isinstance(segmentation, list)
                    and len(segmentation) > 0
                    and isinstance(segmentation[0], list)
                ):
                    segmentation_coords = segmentation[0]
                else:
                    segmentation_coords = segmentation

                if not segmentation_coords:
                    continue

                normalized_coords: List[str] = []
                for i in range(0, len(segmentation_coords), 2):
                    x = segmentation_coords[i] / img_width
                    y = segmentation_coords[i + 1] / img_height
                    normalized_coords.append(f"{x:.6f}")
                    normalized_coords.append(f"{y:.6f}")

                f.write(f"{category_id} {' '.join(normalized_coords)}\n")
                has_annotations = True

        if has_annotations:
            annotated_image_filenames.append(image_filename)
        else:
            os.remove(label_file_path)

    print(
        f"Conversion completed. Labels created for {len(annotated_image_filenames)} images."
    )

    return annotated_image_filenames


def copy_files_to_split_dir(
    filenames: List[str],
    source_images_dir: str,
    dest_images_dir: str,
    dest_labels_dir: str
) -> None:
    """
    Copies images and corresponding YOLO labels into a target dataset split.
    """

    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)

    split_name = os.path.basename(os.path.dirname(dest_images_dir))
    print(f"Copying {len(filenames)} files to **{split_name}** set...")

    for filename in tqdm(filenames, desc=f"Copying to {split_name}"):
        src_image_path = os.path.join(source_images_dir, filename)
        dst_image_path = os.path.join(dest_images_dir, filename)
        shutil.copy(src_image_path, dst_image_path)

        label_filename = os.path.splitext(filename)[0] + ".txt"
        src_label_path = os.path.join(
            yolo_output_dir, "temp_all_yolo_labels", label_filename
        )
        dst_label_path = os.path.join(dest_labels_dir, label_filename)

        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
        else:
            open(dst_label_path, "a").close()


# ===================== MAIN EXECUTION BLOCK =====================
if __name__ == "__main__":

    print("\n--- 1. Converting all annotations into a temporary directory ---")

    temp_all_yolo_labels_dir = os.path.join(
        yolo_output_dir, "temp_all_yolo_labels"
    )
    os.makedirs(temp_all_yolo_labels_dir, exist_ok=True)

    train_val_image_filenames = convert_coco_to_yolo_segmentation(
        hrs_id_train_annotations_json,
        hrs_id_images_dir,
        temp_all_yolo_labels_dir,
    )

    test_image_filenames = convert_coco_to_yolo_segmentation(
        hrs_id_test_annotations_json,
        hrs_id_images_dir,
        temp_all_yolo_labels_dir,
    )

    print("\n--- 2. Splitting and copying files into final dataset structure ---")

    random.seed(42)
    random.shuffle(train_val_image_filenames)
    random.shuffle(test_image_filenames)

    selected_train_files = train_val_image_filenames[:train_count]
    remaining_train_val_files = train_val_image_filenames[train_count:]
    selected_val_files = remaining_train_val_files[:val_count]
    selected_test_files = test_image_filenames[:test_count]

    train_images_dir = os.path.join(yolo_output_dir, "train", "images")
    val_images_dir = os.path.join(yolo_output_dir, "val", "images")
    test_images_dir = os.path.join(yolo_output_dir, "test", "images")

    train_labels_dir = os.path.join(yolo_output_dir, "train", "labels")
    val_labels_dir = os.path.join(yolo_output_dir, "val", "labels")
    test_labels_dir = os.path.join(yolo_output_dir, "test", "labels")

    copy_files_to_split_dir(
        selected_train_files,
        hrs_id_images_dir,
        train_images_dir,
        train_labels_dir,
    )

    copy_files_to_split_dir(
        selected_val_files,
        hrs_id_images_dir,
        val_images_dir,
        val_labels_dir,
    )

    copy_files_to_split_dir(
        selected_test_files,
        hrs_id_images_dir,
        test_images_dir,
        test_labels_dir,
    )

    shutil.rmtree(temp_all_yolo_labels_dir)
    print(f"Temporary directory removed: {temp_all_yolo_labels_dir}")

    print("\n--- 3. Creating data.yaml ---")

    create_dynamic_data_yaml(
        dynamic_data_yaml_path,
        train_images_dir,
        val_images_dir,
        test_images_dir,
        class_names,
    )
