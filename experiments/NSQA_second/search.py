"""
Multi-Objective Genetic Algorithm (NSGA-II) for YOLOv8 Segmentation
=================================================================

This module implements a production-ready Multi-Objective Genetic Algorithm
based on NSGA-II to optimize YOLOv8 segmentation training hyperparameters.

Objectives (minimized):
- Negative mAP50–95 (mask)
- Training time
- False Negative rate (FN)
- False Positive rate (FP)

The script is designed as a single-file experiment runner with:
- Reproducible hyperparameter search space
- Robust error handling
- CSV logging
- Pareto front visualization
- Top-k model extraction

Author: —
"""

from __future__ import annotations

import os
import time
import random
import copy
import csv
import shutil
import traceback
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

DATA_YAML_PATH: str = "data/HRSID_YOLO_Format/HRSID_data.yaml"
MODEL_CFG_OR_WEIGHTS: str = "yolov8n-seg.pt"

PROJECT_DIR: str = "data/MOGA/search_results"
os.makedirs(PROJECT_DIR, exist_ok=True)

CSV_PATH: str = os.path.join(PROJECT_DIR, "moga_log.csv")

EPOCHS_PER_INDIVIDUAL: int = 10
POP_SIZE: int = 8
N_GENERATIONS: int = 5
P_CROSS: float = 0.9
P_MUT: float = 0.3


# ============================================================
# Hyperparameter Search Space
# ============================================================

PARAM_SPACE: Dict[str, Tuple[float, float]] = {
    "lr0": (0.001, 0.02),
    "lrf": (0.01, 0.2),
    "momentum": (0.90, 0.97),
    "weight_decay": (0.0001, 0.001),
    "warmup_epochs": (1.0, 5.0),
    "warmup_momentum": (0.6, 0.9),
    "box": (5.0, 10.0),
    "cls": (0.3, 1.0),
    "dfl": (1.0, 2.0),
    "hsv_h": (0.0, 0.02),
    "hsv_s": (0.0, 0.4),
    "hsv_v": (0.0, 0.4),
    "degrees": (0.0, 15.0),
    "translate": (0.05, 0.2),
    "scale": (0.5, 0.9),
    "shear": (0.0, 5.0),
    "perspective": (0.0, 0.0005),
    "fliplr": (0.3, 0.7),
    "flipud": (0.0, 0.3),
    "mosaic": (0.5, 1.0),
    "mixup": (0.0, 0.15),
    "copy_paste": (0.0, 0.3),
}


# ============================================================
# Genetic Operators
# ============================================================

def random_individual() -> Dict[str, float]:
    """Sample a random individual from the search space."""
    return {k: random.uniform(v[0], v[1]) for k, v in PARAM_SPACE.items()}


def mutate(individual: Dict[str, float]) -> Dict[str, float]:
    """Apply Gaussian mutation to an individual."""
    mutated = copy.deepcopy(individual)
    for key, (low, high) in PARAM_SPACE.items():
        if random.random() < P_MUT:
            sigma = 0.15 * (high - low)
            mutated[key] = float(np.clip(
                np.random.normal(mutated[key], sigma), low, high
            ))
    return mutated


def crossover(
    parent1: Dict[str, float],
    parent2: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Uniform crossover between two parents."""
    child1, child2 = {}, {}
    for key in PARAM_SPACE:
        if random.random() < 0.5:
            child1[key], child2[key] = parent1[key], parent2[key]
        else:
            child1[key], child2[key] = parent2[key], parent1[key]
    return child1, child2


# ============================================================
# CSV Logging
# ============================================================

def init_csv_log() -> None:
    """Initialize CSV log file."""
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "generation", "index", "survivor",
                "mAP50_95_mask", "precision", "recall",
                "FN", "FP", "train_time",
                *PARAM_SPACE.keys(),
            ]
        )


def log_individual(
    generation: int,
    index: int,
    survivor: int,
    metrics: Dict[str, float],
    params: Dict[str, float],
) -> None:
    """Append individual evaluation results to CSV."""
    row = [
        generation,
        index,
        survivor,
        metrics["mAP50_95_mask"],
        metrics["precision_mask"],
        metrics["recall_mask"],
        metrics["FN"],
        metrics["FP"],
        metrics["train_time"],
        *[params[k] for k in PARAM_SPACE],
    ]
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ============================================================
# Evaluation
# ============================================================

def evaluate_individual(
    params: Dict[str, float],
    tag: str,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Train and evaluate a YOLOv8 model with given hyperparameters.

    Returns:
        fitness: List of minimized objectives.
        metrics: Raw evaluation metrics.
    """
    run_name = f"eval_{tag}"
    save_dir = os.path.join(PROJECT_DIR, run_name)

    model = YOLO(MODEL_CFG_OR_WEIGHTS)

    args: Dict[str, Any] = {
        "data": DATA_YAML_PATH,
        "epochs": EPOCHS_PER_INDIVIDUAL,
        "project": PROJECT_DIR,
        "name": run_name,
        "exist_ok": True,
        "imgsz": 1024,
        "batch": 16,
        "verbose": False,
        **params,
    }

    start_time = time.time()
    try:
        model.train(**args)
        train_time = time.time() - start_time

        val = model.val(data=DATA_YAML_PATH)

        if hasattr(val, "seg"):
            seg = val.seg
            map95 = float(seg.map)
            precision = float(seg.mp)
            recall = float(seg.mr)
        else:
            d = val.results_dict
            map95 = float(d.get("metrics/mAP50-95(M)", 0.0))
            precision = float(d.get("metrics/precision(M)", 0.0))
            recall = float(d.get("metrics/recall(M)", 0.0))

        FN = 1.0 - recall
        FP = 1.0 - precision

        metrics = {
            "mAP50_95_mask": map95,
            "precision_mask": precision,
            "recall_mask": recall,
            "FN": FN,
            "FP": FP,
            "train_time": train_time,
        }

        fitness = [-map95, train_time, FN, FP]

    except Exception:
        traceback.print_exc()
        metrics = {
            "mAP50_95_mask": 0.0,
            "precision_mask": 0.0,
            "recall_mask": 0.0,
            "FN": 1.0,
            "FP": 1.0,
            "train_time": 1e6,
        }
        fitness = [0.0, 1e6, 1.0, 1.0]

    shutil.rmtree(save_dir, ignore_errors=True)
    return fitness, metrics


# ============================================================
# NSGA-II Core
# ============================================================

def dominates(a: List[float], b: List[float]) -> bool:
    """Check Pareto dominance."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fast_non_dominated_sort(fitness: List[List[float]]) -> List[List[int]]:
    """Compute non-dominated fronts."""
    S = [[] for _ in fitness]
    n = [0] * len(fitness)
    fronts: List[List[int]] = [[]]

    for p in range(len(fitness)):
        for q in range(len(fitness)):
            if dominates(fitness[p], fitness[q]):
                S[p].append(q)
            elif dominates(fitness[q], fitness[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts


def crowding_distance(front: List[int], fitness: List[List[float]]) -> Dict[int, float]:
    """Compute crowding distance for a front."""
    distance = {i: 0.0 for i in front}
    num_objectives = len(fitness[0])

    for m in range(num_objectives):
        sorted_idx = sorted(front, key=lambda i: fitness[i][m])
        f_min = fitness[sorted_idx[0]][m]
        f_max = fitness[sorted_idx[-1]][m]

        distance[sorted_idx[0]] = distance[sorted_idx[-1]] = float("inf")

        if f_max == f_min:
            continue

        for j in range(1, len(sorted_idx) - 1):
            prev_f = fitness[sorted_idx[j - 1]][m]
            next_f = fitness[sorted_idx[j + 1]][m]
            distance[sorted_idx[j]] += (next_f - prev_f) / (f_max - f_min)

    return distance


def select_nsga2(
    population: List[Dict[str, float]],
    fitness: List[List[float]],
) -> Tuple[List[Dict[str, float]], List[List[float]]]:
    """Select next generation using NSGA-II."""
    fronts = fast_non_dominated_sort(fitness)
    new_pop, new_fit = [], []

    for front in fronts:
        if len(new_pop) + len(front) > POP_SIZE:
            cd = crowding_distance(front, fitness)
            sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
            needed = POP_SIZE - len(new_pop)
            selected = sorted_front[:needed]
            new_pop.extend(population[i] for i in selected)
            new_fit.extend(fitness[i] for i in selected)
            break
        else:
            new_pop.extend(population[i] for i in front)
            new_fit.extend(fitness[i] for i in front)

    return new_pop, new_fit


# ============================================================
# Visualization
# ============================================================

def plot_pareto(population: List[Dict[str, float]], fitness: List[List[float]]) -> None:
    """Plot Pareto fronts."""
    F = np.array(fitness)
    mAP = -F[:, 0]
    train_time = F[:, 1]
    FN = F[:, 2]
    FP = F[:, 3]

    plots = [
        (train_time, mAP, "Train Time", "mAP50–95 (mask)", "pareto_time.png"),
        (FN, mAP, "False Negative Rate", "mAP50–95 (mask)", "pareto_fn.png"),
        (FP, mAP, "False Positive Rate", "mAP50–95 (mask)", "pareto_fp.png"),
    ]

    for x, y, xlabel, ylabel, name in plots:
        plt.figure(figsize=(6, 5))
        plt.scatter(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(os.path.join(PROJECT_DIR, name))
        plt.close()


# ============================================================
# Main Optimization Loop
# ============================================================

def run_moga() -> Tuple[List[Dict[str, float]], List[List[float]]]:
    """Execute the full MOGA optimization."""
    init_csv_log()

    population = [random_individual() for _ in range(POP_SIZE)]
    fitness: List[List[float]] = []

    for i, individual in enumerate(population):
        fit, metrics = evaluate_individual(individual, f"g0_i{i}")
        fitness.append(fit)
        log_individual(0, i, 0, metrics, individual)

    for gen in range(1, N_GENERATIONS + 1):
        parents = []
        while len(parents) < POP_SIZE:
            i1, i2 = random.sample(range(POP_SIZE), 2)
            parents.append(
                population[i1] if dominates(fitness[i1], fitness[i2]) else population[i2]
            )

        offspring = []
        while len(offspring) < POP_SIZE:
            if random.random() < P_CROSS:
                p1, p2 = random.sample(parents, 2)
                c1, c2 = crossover(p1, p2)
                offspring.append(mutate(c1))
                if len(offspring) < POP_SIZE:
                    offspring.append(mutate(c2))
            else:
                offspring.append(mutate(random.choice(parents)))

        offspring_fitness = []
        for i, individual in enumerate(offspring):
            fit, metrics = evaluate_individual(individual, f"g{gen}_i{i}")
            offspring_fitness.append(fit)
            log_individual(gen, i, 0, metrics, individual)

        population, fitness = select_nsga2(
            population + offspring,
            fitness + offspring_fitness,
        )

    plot_pareto(population, fitness)
    return population, fitness


# ============================================================
# Top-K Extraction
# ============================================================

def extract_top5(
    population: List[Dict[str, float]],
    fitness: List[List[float]],
) -> pd.DataFrame:
    """Extract and save top-5 individuals by mAP."""
    records = []
    for params, fit in zip(population, fitness):
        records.append({
            "mAP50_95_mask": -fit[0],
            "train_time": fit[1],
            "FN": fit[2],
            "FP": fit[3],
            **params,
        })

    df = pd.DataFrame(records).sort_values(
        by="mAP50_95_mask", ascending=False
    )

    top5 = df.head(5)
    top5.to_json(
        os.path.join(PROJECT_DIR, "top5_models.json"),
        orient="records",
        indent=4,
    )
    return top5


if __name__ == "__main__":
    final_population, final_fitness = run_moga()
    extract_top5(final_population, final_fitness)
