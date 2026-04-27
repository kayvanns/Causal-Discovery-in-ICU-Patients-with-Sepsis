import pickle
import os

dirs = [
    "/Users/kayvans/Documents/sepsis-causal-discovery/graphs_v3/v3",
    "/Users/kayvans/Documents/sepsis-causal-discovery/graphs_v3/v3_kci",
]

all_cols = set()
for d in dirs:
    for file in os.listdir(d):
        if not file.endswith(".pkl"):
            continue
        with open(os.path.join(d, file), "rb") as f:
            graph, cols = pickle.load(f)
        all_cols.update(cols)
        print(f"{file}: {len(cols)} cols")

print("\nAll unique column names across all runs:")
for c in sorted(all_cols):
    print(c)