import time
from copy import deepcopy

import numpy as np
import pandas as pd
import rootutils
import torch
from torch_geometric.transforms import Compose
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.benchmark_utils import create_phenotype_graph
from src.utils.graph_augmentations_phenotype import (
    AddEdgesByCellType,
    DropEdges,
    DropFeatures,
    DropImportance,
    FeatureNoise,
    ShufflePositions,
)

# ------- Parameters -------
graph_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 150000]
num_neighbors = 10
num_classes = 30
num_runs = 3
seed = 44

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

augmentations = {
    "DropFeatures": DropFeatures(p=0.2),
    "DropEdges": DropEdges(p=0.2),
    "DropImportance": DropImportance(mu=0.2, p_lambda=0.5),
    "FeatureNoise": FeatureNoise(feature_noise_std=1),
    "ShufflePositions": ShufflePositions(p_shuffle=0.2),
    "AddEdgesByCellType": AddEdgesByCellType(p_add=0.2, k_add=2),
}

augmentation_combos = {
    "DropFeatures": [augmentations["DropFeatures"]],
    "DropEdges": [augmentations["DropEdges"]],
    "FeatureNoise": [augmentations["FeatureNoise"]],
    "DropImportance": [augmentations["DropImportance"]],
    "ShufflePositions": [augmentations["ShufflePositions"]],
    "AddEdgesByCellType": [augmentations["AddEdgesByCellType"]],
    "Baseline": [augmentations["DropEdges"], augmentations["DropFeatures"]],
    "Baseline + FeatureNoise": [
        augmentations["DropEdges"],
        augmentations["DropFeatures"],
        augmentations["FeatureNoise"],
    ],
    "DropImportance + FeatureNoise": [
        augmentations["DropImportance"],
        augmentations["FeatureNoise"],
    ],
    "DropImportance + FeatureNoise + ShufflePositions": [
        augmentations["DropImportance"],
        augmentations["FeatureNoise"],
        augmentations["ShufflePositions"],
    ],
    "DropImportance + FeatureNoise + AddEdgesByCellType": [
        augmentations["DropImportance"],
        augmentations["FeatureNoise"],
        augmentations["AddEdgesByCellType"],
    ],
}

results = []

# ------- Benchmarking -------
for size in tqdm(graph_sizes, desc="Graph sizes"):
    graph = create_phenotype_graph(
        graph_name=f"graph_{size}",
        num_nodes=size,
        num_neighbors=num_neighbors,
        num_classes=num_classes,
        seed=seed,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    for name, transforms in augmentation_combos.items():
        compose = Compose(transforms)

        times = []
        memory_usages = []

        for _ in range(num_runs):
            data = deepcopy(graph)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start = time.time()
            _ = compose(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            duration = time.time() - start
            times.append(duration)

            # Memory in MB
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1024**2
                memory_usages.append(mem)
            else:
                memory_usages.append(0.0)

        results.append(
            {
                "augmentation": name,
                "num_nodes": size,
                "num_edges": data.num_edges,
                "avg_time_s": sum(times) / len(times),
                "max_memory_mb": max(memory_usages),
            }
        )

# ------- Results -------
df = pd.DataFrame(results)
pivot_time = df.pivot(index="num_nodes", columns="augmentation", values="avg_time_s")
pivot_mem = df.pivot(index="num_nodes", columns="augmentation", values="max_memory_mb")

print("=== Average Runtime (s) ===")
print(pivot_time.round(4))
print("\n=== Max Memory Usage (MB) ===")
print(pivot_mem.round(2))

df.to_csv("data/benchmark/phenotype_benchmark_results.csv", index=False)
