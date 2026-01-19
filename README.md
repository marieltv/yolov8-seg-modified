# Ship Detection and Segmentation in Satellite Images  
**YOLOv8 + Cross-Validation + Evolutionary Hyperparameter Optimization**

---

## 1. Business Problem

Maritime monitoring relies on accurate and timely detection of ships in satellite imagery for applications such as:

- Port traffic analysis  
- Illegal fishing detection  
- Maritime security  
- Environmental monitoring  

Manual annotation and visual inspection of satellite images is:

- Time-consuming  
- Error-prone  
- Not scalable for large geographic areas or high revisit rates  

The core problem addressed in this project is:

> **How can ships be detected and segmented automatically, accurately, and reliably from satellite images at scale?**

---

## 2. Why AI / Machine Learning Was Needed

Traditional image processing methods (thresholding, edge detection, handcrafted features):

- Fail under varying lighting and weather conditions  
- Struggle with small or partially occluded ships  
- Do not generalize across different sea states or sensors  

Deep learning–based computer vision models:

- Learn robust visual features directly from data  
- Handle scale, texture, and background variation  
- Generalize better to unseen scenes  

AI/ML enables:

- End-to-end automation of ship detection  
- Pixel-level segmentation instead of only bounding boxes  
- Deployment-ready inference pipelines  

This makes ML the only practical solution for reliable large-scale maritime monitoring.

---

## 3. Solution Overview & Technology Stack

### Model Architecture

- **YOLOv8-Segmentation** (Ultralytics) as a base model
  - Real-time instance segmentation
  - Strong performance–speed tradeoff
  - Suitable for production pipelines
- Seperate experiments were conducted in order to investigate if we can improve map50-95 mask, FN and FP without significant increase in the inference time:
  - Coordinate Attention Module
    - Coordinate Attention is designed to enhance spatial and channel-wise feature representation by embedding positional information into channel attention.
It has shown gains in lightweight vision backbones and detection tasks.
  - Tversky/Focal Tversky loss functions
    - Ship segmentation suffers from strong foreground–background class imbalance.
Tversky and Focal-Tversky losses are frequently proposed as improvements over BCE/Dice for imbalanced segmentation tasks.

---

### Dataset

- **HRSID (High-Resolution Ship Detection Dataset)**
  - Satellite imagery with ship annotations
  - Converted from COCO to YOLO-Seg format  
  - Preprocessed into a reproducible training layout

---

### Training & Evaluation Strategy

To ensure statistically reliable results:

- **5-Fold Cross-Validation**
  - Prevents overfitting to a single split  
  - Measures performance stability  
  - Enables fair experiment comparison

 As results we got everage of metrics across folds with standart deviation.

---

### Hyperparameter Optimization
 
- **NSGA-II (Multi-objective GA)**  

Objectives:

- Maximize segmentation accuracy  
- Improve detection stability  
- Balance training speed vs performance  

Optimized hyperparameters were manually transferred into
cross-validation training scripts to evaluate final performance fairly.

---

### Engineering Design

- Modular experiment structure  
- Separate folders for:
  - data
  - data pipeline
  - Baseline CV  
  - Coordinate Attention Module
  - Tversky/Focal-Tversky 
  - NSGA-optimized CV  

Each with seperate README with detailed information and instructions.

- Reproducibility features:
  - Fixed random seeds  
  - YAML-based dataset configs  
  - Explicit hyperparameter logging  

---

### Technology Stack

- **Python**
- **PyTorch**
- **Ultralytics YOLOv8**
- **scikit-learn** (K-Fold CV)
- **NumPy**
- **YAML**
- **Matplotlib** (analysis)
- **CUDA** (GPU training)

---

### High-Level Pipeline

```text
Raw Satellite Images
        │
        ▼
COCO → YOLO-Seg Conversion
        │
        ▼
Train / Val Split Generation (K-Fold)
        │
        ▼
YOLOv8-Seg Training
        │
        ▼
Cross-Validation Evaluation
        │
        ▼
Metric Aggregation & Comparison
```
---

### Folder structure

```text
.
├── data/
│   ├── HRSID_raw/                # Raw dataset (not stored in repo)
│   ├── HRSID_YOLO_Format/        # Converted YOLO-Seg dataset
│   └── README.md                 # Dataset download instructions
│
├── experiments/
│   ├── baseline_cv/
│   ├── ga_cv/
│   ├── nsga_cv/
│   └── coord_attention/
│
├── tools/
│   ├── coco_to_yolo.py
│   ├── dataset_utils.py
│
├── requirements.txt
└── README.md
```
