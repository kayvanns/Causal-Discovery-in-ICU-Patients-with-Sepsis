import networkx as nx
import pandas as pd
from dowhy import CausalModel


df = pd.read_csv("data/processed/analysis_cleaned.csv")
consensus = pd.read_csv("results/consensus_edges.csv")


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

indicator_cols = ["FiO2_max", "bilirubin_max", "blood_pressure_min", "cvp_max",
                  "hemoglobin_min", "inr_max", "lactate_max", "lymphocyte_abs_min",
                  "platelet_max", "temp_max_F", "wbc_max", "BMI"]

for col in indicator_cols:
    df[f"{col}_missing"] = df[col].isna().astype(int)


df = df.rename(columns=label_map)
print(df.columns.tolist())

G = nx.DiGraph()
for _, row in consensus.iterrows():
    cause = row["cause"]
    effect = row["effect"]
    G.add_edge(cause, effect)

# What goes INTO mechvent_24h_onset (potential confounders)
print("Parents of MV Onset (24h):")
print(list(G.predecessors("Mech. Vent Onset (24h)")))

# What goes INTO aki_post24h_stage
print("\nParents of AKI Post-24h:")
print(list(G.predecessors("AKI Post-24h")))

# Common ancestors
mv_ancestors = nx.ancestors(G, "Mech. Vent Onset (24h)")
aki_ancestors = nx.ancestors(G, "AKI Post-24h")
print("\nCommon ancestors (potential confounders):")
print(mv_ancestors & aki_ancestors)

# What goes INTO aki_24h_onset_stage (potential confounders)
print("Parents of AKI Onset (24h):")
print(list(G.predecessors("AKI Onset (24h)")))

# What goes INTO mechvent_post24h
print("\nParents of MV Post-24h:")
print(list(G.predecessors("Mech. Vent Post-24h")))

# Common ancestors for AKI → MV
aki_early_ancestors = nx.ancestors(G, "AKI Onset (24h)")
mv_late_ancestors = nx.ancestors(G, "Mech. Vent Post-24h")
print("\nCommon ancestors (potential confounders for AKI → MV):")
print(aki_early_ancestors & mv_late_ancestors)



print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
assert nx.is_directed_acyclic_graph(G), "Graph has cycles, which is not allowed for a DAG and DoWhy model."

dot_lines = ["digraph {"]
for cause, effect in G.edges():
    dot_lines.append(f'    "{cause}" -> "{effect}";')
dot_lines.append("}")
dot_string = "\n".join(dot_lines)

print(dot_string[:500]) 

# ── DoWhy — MV → AKI ─────────────────────────────────────────────────────────
model_mv_to_aki = CausalModel(
    data=df,
    treatment="Mech. Vent Onset (24h)",
    outcome="AKI Post-24h",
    graph=dot_string
)
model_mv_to_aki.view_model()
identified_mv_to_aki = model_mv_to_aki.identify_effect(proceed_when_unidentifiable=True)
print("\nMV → AKI adjustment set:")
print(identified_mv_to_aki.get_backdoor_variables())

# ── DoWhy — AKI → MV ─────────────────────────────────────────────────────────
model_aki_to_mv = CausalModel(
    data=df,
    treatment="AKI Onset (24h)",
    outcome="Mech. Vent Post-24h",
    graph=dot_string
)
model_aki_to_mv.view_model()
identified_aki_to_mv = model_aki_to_mv.identify_effect(proceed_when_unidentifiable=True)
print("\nAKI → MV adjustment set:")
print(identified_aki_to_mv.get_backdoor_variables())