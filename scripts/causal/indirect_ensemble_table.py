import pickle
import pandas as pd
import os
from collections import defaultdict

_ROOT       = os.path.join(os.path.dirname(__file__), "../..")
GRAPHS_DIRS = [
    os.path.join(_ROOT, "graphs_v3", "v3"),
    os.path.join(_ROOT, "graphs_v3", "v3_kci"),
]
RESULTS_DIR = os.path.join(_ROOT, "results")

def get_direct_edges(matrix, cols):
    n = len(cols)
    edges = defaultdict(list)
    for i in range(n):
        for j in range(n):
            if matrix[i, j] == -1 and matrix[j, i] == 1:
                edges[i].append(j)
    return edges

def find_all_paths(edges, start, end):
    all_paths = []
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        for neighbor in edges[node]:
            if neighbor == end:
                all_paths.append(path + [neighbor])
            elif neighbor not in path:
                stack.append((neighbor, path + [neighbor]))
    return all_paths

def build_ensemble_table(graph_dirs):
    run_edges = []
    for graph_dir in graph_dirs:
        for file in os.listdir(graph_dir):
            if not file.endswith(".pkl"):
                continue
            run_name = file.replace(".pkl", "")
            with open(os.path.join(graph_dir, file), "rb") as f:
                graph, cols = pickle.load(f)
            matrix = graph.graph
            n = len(cols)
            edges = get_direct_edges(matrix, cols)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    paths = find_all_paths(edges, i, j)
                    if paths:
                        indirect_paths = [p for p in paths if len(p) > 2]
                        run_edges.append({
                                "run":        run_name,
                                "cause":      cols[i],
                                "effect":     cols[j],
                                "direct":  1 if any(len(p) == 2 for p in paths) else 0,
                                "num_paths":  len(indirect_paths),
                            })

    df = pd.DataFrame(run_edges)

    
    label_map = {
    "anchor_age":                   "Age",
    "gender":                       "Gender",
    "race":                         "Race",
    "ckd_baseline":                 "CKD (Baseline)",
    "heart_rate_max":               "Heart Rate (max)",
    "blood_pressure_min":           "Blood Pressure (min)",
    "spO2_min":                     "SpO2 (min)",
    "FiO2_max":                     "FiO2 (max)",
    "lactate_max":                  "Lactate (max)",
    "bilirubin_max":                "Bilirubin (max)",
    "platelet_max":                 "Platelet (max)",
    "inr_max":                      "INR (max)",
    "temp_max_F":                   "Temperature (max)",
    "cvp_max":                      "CVP (max)",
    "hemoglobin_min":               "Hemoglobin (min)",
    "lymphocyte_abs_min":           "Lymphocytes Abs (min)",
    "fluid_balance":                "Fluid Balance",
    "BMI":                          "BMI",
    "wbc_max":                      "WBC (max)",
    "antibiotics_given":            "Antibiotics",
    "vaso_given":                   "Vasopressors",
    "diuretics_given":              "Diuretics",
    "aki_24h_onset_stage":          "AKI Onset (24h)",
    "mechvent_24h_onset":           "Mech. Vent Onset (24h)",
    "aki_post24h_stage":            "AKI Post-24h",
    "mechvent_post24h":             "Mech. Vent Post-24h",
    "hospital_expire_flag":         "Hospital Mortality",
    # missingness indicators
    "BMI_missing":                  "BMI Missing",
    "FiO2_max_missing":             "FiO2 Missing",
    "bilirubin_max_missing":        "Bilirubin Missing",
    "blood_pressure_min_missing":   "BP Missing",
    "cvp_max_missing":              "CVP Missing",
    "hemoglobin_min_missing":       "Hemoglobin Missing",
    "inr_max_missing":              "INR Missing",
    "lactate_max_missing":          "Lactate Missing",
    "lymphocyte_abs_min_missing":   "Lymphocytes Missing",
    "platelet_max_missing":         "Platelet Missing",
    "temp_max_F_missing":           "Temp Missing",
    "wbc_max_missing":              "WBC Missing",
}

    df["cause"] = df["cause"].map(label_map).fillna(df["cause"])
    df["effect"] = df["effect"].map(label_map).fillna(df["effect"])
    df["edge"] = df["cause"] + " --> " + df["effect"]
    df["present"] = 1

    table = df.pivot_table(
        index=["edge","cause","effect"],
        columns="run",
        values="present",
        aggfunc="max",
        fill_value=0
    ).reset_index()

    avg_paths = df.groupby(["edge","cause","effect"])["num_paths"].mean().round(1).reset_index()
    avg_paths.columns = ["edge", "cause", "effect", "avg_num_paths"]
    table = table.merge(avg_paths, on=["edge","cause","effect"])
    
    run_cols = [c for c in table.columns if c not in ["edge","cause","effect","avg_num_paths"]]
    table["agreement_score"] = table[run_cols].sum(axis=1)
    table = table.sort_values("agreement_score", ascending=False).reset_index(drop=True)
    return table


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table = build_ensemble_table(GRAPHS_DIRS)
    table.to_csv(os.path.join(RESULTS_DIR, "total_ensemble_table_v2.csv"), index=False)
