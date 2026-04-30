import os
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import pickle
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import kci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from collections import defaultdict

_ROOT       = os.path.join(os.path.dirname(__file__), "../..")
VERSION     = "v3_kci"
GRAPHS_DIR  = os.path.join(_ROOT, f"graphs/{VERSION}")
RESULTS_DIR = os.path.join(_ROOT, f"results/{VERSION}")
DATA_PATH   = os.path.join(_ROOT, "data/processed/analysis_cleaned.csv")

DEMO_COLS   = ["anchor_age", "gender", "race", "ckd_baseline"]
PHYS_COLS   = [
    "heart_rate_max", "blood_pressure_min", "spO2_min", "FiO2_max",
    "lactate_max", "bilirubin_max", "platelet_max", "inr_max",
    "temp_max_F", "cvp_max", "hemoglobin_min",
    "lymphocyte_abs_min", "fluid_balance", "BMI", "wbc_max"
]
TREAT_COLS  = ["antibiotics_given", "vaso_given", "diuretics_given"]
OUT24_COLS  = ["aki_24h_onset_stage", "mechvent_24h_onset"]
POST24_COLS = ["aki_post24h_stage", "mechvent_post24h"]
MORT_COLS   = ["hospital_expire_flag"]

CORE_COLS = DEMO_COLS + PHYS_COLS + TREAT_COLS + OUT24_COLS + POST24_COLS + MORT_COLS

TIER_MAP = {
    **{c: 0 for c in DEMO_COLS},
    **{c: 1 for c in PHYS_COLS + TREAT_COLS + OUT24_COLS},
    **{c: 2 for c in POST24_COLS},
    **{c: 3 for c in MORT_COLS},
}

CATEGORICAL_COLS = ["gender", "race"]
BINARY_COLS = [
    "hospital_expire_flag", "antibiotics_given", "vaso_given",
    "mechvent_24h_onset", "mechvent_post24h", "ckd_baseline", "diuretics_given"
]

SAMPLE_SIZE = 1500
ALPHA       = 0.1
SEED        = 42

def load_data(path: str):
    df = pd.read_csv(path)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    available = [c for c in CORE_COLS if c in df.columns]
    missing   = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Columns not found and skipped: {missing}")
    df = df[available].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, available

def impute_knn(arr: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    scaler = StandardScaler()
    complete_rows = arr[~np.isnan(arr).any(axis=1)]
    scaler.fit(complete_rows)
    arr_scaled         = scaler.transform(arr)
    arr_imputed_scaled = KNNImputer(n_neighbors=n_neighbors).fit_transform(arr_scaled)
    return scaler.inverse_transform(arr_imputed_scaled)

def make_indicator(df_raw: pd.DataFrame, col: str):
    if col not in df_raw.columns:
        return None, f"{col} not in dataframe"
    indicator = df_raw[col].isna().astype(float)
    missing_rate = indicator.mean()
    if missing_rate == 0.0:
        return None, f"{col} has no missing values"
    if missing_rate < 0.01 or missing_rate > 0.99:
        return None, f"{col} missing rate {missing_rate:.2%} — near-constant"
    print(f"  [indicator] {col}_missing: {missing_rate:.2%} missing")
    return indicator.rename(f"{col}_missing"), None

def build_background_knowledge(nodes: list, col_names: list) -> BackgroundKnowledge:
    col_to_node = {col: nodes[i] for i, col in enumerate(col_names)}
    bk = BackgroundKnowledge()
    for col, tier in TIER_MAP.items():
        if col in col_to_node:
            bk.add_node_to_tier(col_to_node[col], tier)
    for col in col_names:
        indicator_name = f"{col}_missing"
        if indicator_name in col_to_node:
            bk.add_node_to_tier(col_to_node[indicator_name], 1)
    for i, c1 in enumerate(DEMO_COLS):
        for c2 in DEMO_COLS[i+1:]:
            if c1 in col_to_node and c2 in col_to_node:
                bk.add_forbidden_by_node(col_to_node[c1], col_to_node[c2])
                bk.add_forbidden_by_node(col_to_node[c2], col_to_node[c1])
    for col in col_names:
        indicator_name = f"{col}_missing"
        if indicator_name in col_to_node:
            # indicator can't cause its own variable
            if col in col_to_node:
                bk.add_forbidden_by_node(col_to_node[indicator_name], col_to_node[col])
            
            # indicator can't cause demographics
            for demo in DEMO_COLS:
                if demo in col_to_node:
                    bk.add_forbidden_by_node(col_to_node[indicator_name], col_to_node[demo])
            
            # indicator can't cause ANY clinical variable
            for other_col in PHYS_COLS + TREAT_COLS + OUT24_COLS + POST24_COLS + MORT_COLS:
                if other_col in col_to_node:
                    bk.add_forbidden_by_node(col_to_node[indicator_name], col_to_node[other_col])
            
            # indicators can't cause each other
            for other_col in col_names:
                other_indicator = f"{other_col}_missing"
                if other_indicator in col_to_node and other_indicator != indicator_name:
                    bk.add_forbidden_by_node(col_to_node[indicator_name], col_to_node[other_indicator])
    return bk

def run_and_save(algo, data_sample, all_cols, run_name, alpha=ALPHA):
    try:
        if algo == "PC":
            cg = pc(data_sample, alpha=alpha, indep_test=kci)
            bk = build_background_knowledge(cg.G.get_nodes(), all_cols)
            cg = pc(data_sample, alpha=alpha, indep_test=kci, background_knowledge=bk)
            graph = cg.G
        elif algo == "FCI":
            g0, _ = fci(data_sample, independence_test_method=kci, alpha=alpha)
            bk = build_background_knowledge(g0.get_nodes(), all_cols)
            graph, _ = fci(data_sample, independence_test_method=kci, alpha=alpha, background_knowledge=bk)

        out_path   = os.path.join(GRAPHS_DIR, f"{run_name}.png")
        graph_path = os.path.join(GRAPHS_DIR, f"{run_name}.pkl")
        pyd = GraphUtils.to_pydot(graph, labels=all_cols)
        pyd.write_png(out_path)
        with open(graph_path, "wb") as f:
            pickle.dump((graph, all_cols), f)
        print(f"  SUCCESS → {out_path}")
        return "SUCCESS"

    except Exception as e:
        print(f"  FAILED: {traceback.format_exc()}")
        return "FAILED"

def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data ...")
    df_raw, col_names = load_data(DATA_PATH)
    print(f"Using {len(col_names)} columns, {len(df_raw)} rows.\n")

    raw_arr = df_raw.to_numpy().astype(float)

    print("Imputing ...")
    simple_arr = SimpleImputer(strategy="mean").fit_transform(raw_arr)
    knn_arr    = impute_knn(raw_arr)
    print("Done.\n")

    print("Checking missingness indicators ...")
    df_simple = pd.DataFrame(simple_arr, columns=col_names)
    df_knn    = pd.DataFrame(knn_arr,    columns=col_names)

    indicator_names = []
    for col in PHYS_COLS:
        indicator, reason = make_indicator(df_raw, col)
        if indicator is None:
            print(f"  skipping {col}: {reason}")
        else:
            df_simple[f"{col}_missing"] = indicator.values
            df_knn[f"{col}_missing"]    = indicator.values
            indicator_names.append(f"{col}_missing")

    all_cols = col_names + indicator_names

    np.random.seed(SEED)
    sample_idx = np.random.choice(len(df_simple), size=SAMPLE_SIZE, replace=False)

    data_simple     = df_simple[col_names].to_numpy().astype(float)[sample_idx]
    data_knn        = df_knn[col_names].to_numpy().astype(float)[sample_idx]
    data_simple_ind = df_simple[all_cols].to_numpy().astype(float)[sample_idx]
    data_knn_ind    = df_knn[all_cols].to_numpy().astype(float)[sample_idx]

    RUNS = [
        ("PC",  data_simple,     col_names, "PC_kci_simple"),
        ("FCI", data_simple,     col_names, "FCI_kci_simple"),
        ("PC",  data_knn,        col_names, "PC_kci_knn"),
        ("FCI", data_knn,        col_names, "FCI_kci_knn"),
        ("PC",  data_simple_ind, all_cols,  "PC_kci_simple_indicator"),
        ("FCI", data_simple_ind, all_cols,  "FCI_kci_simple_indicator"),
        ("PC",  data_knn_ind,    all_cols,  "PC_kci_knn_indicator"),
        ("FCI", data_knn_ind,    all_cols,  "FCI_kci_knn_indicator"),
    ]

    for algo, data, run_cols, run_name in RUNS:
        print(f"\n--- {run_name} ---")
        run_and_save(algo, data, run_cols, run_name)

    print("\nDone.")

if __name__ == "__main__":
    main()