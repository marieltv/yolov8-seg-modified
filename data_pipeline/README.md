# HRSID Dataset Conversion Pipeline (COCO → YOLO Segmentation)

This folder contains scripts for converting the **HRSID dataset** from its original **COCO JSON polygon format** into the **YOLO segmentation format** required for training YOLOv8-seg and YOLOv11-seg models.

The pipeline performs:

- ZIP extraction  
- JSON annotation processing  
- Polygon → YOLO mask conversion  
- Dataset splitting (train / val / test)  
- Automatic generation of `data.yaml`  

After execution, the dataset is ready for training with all experiments in this repository.

---

## Folder Structure

```
data_pipeline/
├── README.md               # This file
├── converting_polygons.py  # Converts COCO polygons → YOLO segmentation format
├── requirements.txt        # Necessary dependencies
├── unzip.py                # Extracts HRSID JPG archive

```

---

## Input Expectations

Before running the scripts, make sure **HRSID dataset** is here:

```
data/HRSID_raw/
    ├── HRSID_jpg.zip
    └── annotations/
          ├── train2017.json
          └── test2017.json
```

These are required for the polygon conversion step.

---
## Step 1- Prepare the environment
Make sure you are in the right directory, use:
```
cd data_pipeline
```
Install dependencies:
```
pip install -r requirements.txt
```

---

## Step 2 — Unzip the HRSID Images

```
python data_pipeline/unzip.py
```

This extracts:

```
data/HRSID_raw/HRSID_JPG/JPEGImages/
```

---

## Step 3 — Convert COCO Annotations to YOLO Segmentation Format

```
python data_pipeline/converting_polygons.py
```

This script will:

- Read COCO JSON segmentation polygons  
- Normalize polygon coordinates  
- Generate YOLO `.txt` label files  
- Create train / val / test splits  
- Build the directory:

```
data/HRSID_YOLO_Format/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    ├── test/
    └── HRSID_data.yaml
```

---

## Output

After both steps, the dataset is fully processed and ready for training:

```
data/HRSID_YOLO_Format/
```

Training scripts (baseline, CoordAtt, Tversky, etc.) can now be executed without further preprocessing.

---

## Notes

- Ensure the HRSID dataset is correctly placed before running the scripts.
- The split sizes (train/val/test) can be configured inside `converting_polygons.py`.
- The resulting `HRSID_data.yaml` is automatically compatible with YOLOv8-seg.

---

If you encounter any issues during conversion, ensure:
- JSON annotations exist and are correctly extracted
- Image paths inside JSON match the filenames in `JPEGImages/`
- Python version ≥ 3.9

