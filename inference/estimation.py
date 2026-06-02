import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dowhy import CausalModel
import dowhy.gcm as gcm
from dowhy.gcm import InvertibleStructuralCausalModel, fit, counterfactual_samples
from dowhy.gcm.ml import create_linear_regressor, create_logistic_regression_classifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

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
df = df.rename(columns=label_map)

# Build consensus DAG
G = nx.DiGraph()
for _, row in consensus.iterrows():
    G.add_edge(row["cause"], row["effect"])

# =============================================================================
# 1. DOMAIN KNOWLEDGE AUGMENTATION and DAG COMPLETION
# =============================================================================

print("=" * 70)
print("SECTION 1: DOMAIN KNOWLEDGE AUGMENTATION and DAG COMPLETION")
print("=" * 70)
 
domain_knowledge_edges = [
    # ── Parents of AKI Onset (24h) ────────────────────────────────────────
    # Hemodynamic instability → renal hypoperfusion (KDIGO 2012)
    ("Blood Pressure (min)",    "AKI Onset (24h)"),
    ("Vasopressors",            "AKI Onset (24h)"),   # marker of shock severity
    ("Lactate (max)",           "AKI Onset (24h)"),   # hypoperfusion proxy
    # Pre-existing vulnerability
    ("CKD (Baseline)",          "AKI Onset (24h)"),   # strongest known risk factor
    ("Age",                     "AKI Onset (24h)"),   # reduced renal reserve
    # Fluid management
    ("Fluid Balance",           "AKI Onset (24h)"),   # both hypo- and hypervolemia harmful
 
    # ── Parents of Mech. Vent Onset (24h) ────────────────────────────────
    # Respiratory failure markers (Berlin ARDS definition)
    ("SpO2 (min)",              "Mech. Vent Onset (24h)"),
    ("FiO2 (max)",              "Mech. Vent Onset (24h)"),
    # Shock / multi-organ failure
    ("Lactate (max)",           "Mech. Vent Onset (24h)"),
    ("Vasopressors",            "Mech. Vent Onset (24h)"),  # co-treatment in septic shock
    # Altered mental status / airway protection (proxied by severity)
    ("Heart Rate (max)",        "Mech. Vent Onset (24h)"),
]
edges_before = G.number_of_edges()
for cause, effect in domain_knowledge_edges:
    if not G.has_edge(cause, effect):
        print(f"Adding edge: {cause} → {effect}")
        G.add_edge(cause, effect)

edges_after = G.number_of_edges()
print(f"Added {edges_after - edges_before} edges based on domain knowledge.")

indicator_map = {
    "BMI":                   "BMI Missing",
    "INR (max)":             "INR Missing",
    "FiO2 (max)":            "FiO2 Missing",
    "CVP (max)":             "CVP Missing",
    "Lactate (max)":         "Lactate Missing",
    "Bilirubin (max)":       "Bilirubin Missing",
    "Lymphocytes Abs (min)": "Lymphocytes Missing",
    "Blood Pressure (min)":  "BP Missing",
    "Temperature (max)":     "Temp Missing",
    "Hemoglobin (min)":      "Hemoglobin Missing",
    "Platelet (max)":        "Platelet Missing",
    "WBC (max)":             "WBC Missing",
}

for col, indicator in indicator_map.items():
    if col in df.columns:
        df[indicator] = df[col].isnull().astype(float)
        print(f"  ✚ Created {indicator}: {df[indicator].sum():.0f} missing")

df = df.replace([np.inf, -np.inf], np.nan)
dag_nodes = list(G.nodes())
cols_to_impute = [c for c in dag_nodes if c in df.columns and df[c].isnull().any()]
print("Imputing columns:", cols_to_impute)
imputer = SimpleImputer(strategy="mean")
df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

assert df[dag_nodes].isnull().sum().sum() == 0, "Still have NaNs in DAG columns"
print("No NaNs in DAG columns")


scaler = StandardScaler()
cols_to_scale = [c for c in df.columns if c not in [
    "AKI Onset (24h)", "Mech. Vent Onset (24h)", 
    "AKI Post-24h", "Mech. Vent Post-24h",
    "Hospital Mortality"
] and df[c].dtype in ['float64', 'int64']]

df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

df["AKI Onset (24h)"] = (df["AKI Onset (24h)"] > 0).astype(int)
df["AKI Post-24h"]           = (df["AKI Post-24h"] > 0).astype(int)

print("Checking for cycles...")



cycles = list(nx.simple_cycles(G))
print(f'Found {len(cycles)} cycles before edge removal.')

for c in cycles:
    print(' --> '.join(c + [c[0]]))


edges_to_remove = [
    ("Lactate Missing", "Heart Rate (max)"),
    ("CVP Missing",     "Heart Rate (max)"),
    ("CVP Missing",     "FiO2 (max)"),
]

for u, v in edges_to_remove:
    if G.has_edge(u, v):
        G.remove_edge(u, v)
        print(f"Removed: {u} → {v}")

cycles_remaining = list(nx.simple_cycles(G))
print(f"Cycles remaining: {len(cycles_remaining)}")
assert len(cycles_remaining) == 0
print("Valid DAG")

print("Parents of AKI Onset (24h)      :", list(G.predecessors("AKI Onset (24h)")))
print("Parents of Mech. Vent Onset (24h):", list(G.predecessors("Mech. Vent Onset (24h)")))

dot_lines = ["digraph {"]
for cause, effect in G.edges():
    dot_lines.append(f'    "{cause}" -> "{effect}";')
dot_lines.append("}")
dot_string = "\n".join(dot_lines)

print(dot_string[-500:])

# =============================================================================
# 2. EFFECT IDENTIFICATION
# =============================================================================
print("=" * 70)
print("SECTION 2: Effect Identification")
print("=" * 70)

 
# ── Model A: AKI (24h) → Mech. Vent Post-24h ─────────────────────────────
model_aki_to_mv = CausalModel(
    data=df,
    treatment="AKI Onset (24h)",
    outcome="Mech. Vent Post-24h",
    graph=dot_string
)
identified_aki_to_mv = model_aki_to_mv.identify_effect(proceed_when_unidentifiable=True)
print("\n── AKI → MV ──")
print("Backdoor adjustment set:", identified_aki_to_mv.get_backdoor_variables())
 
# ── Model B: Mech. Vent (24h) → AKI Post-24h ─────────────────────────────
model_mv_to_aki = CausalModel(
    data=df,
    treatment="Mech. Vent Onset (24h)",
    outcome="AKI Post-24h",
    graph=dot_string
)
identified_mv_to_aki = model_mv_to_aki.identify_effect(proceed_when_unidentifiable=True)
print("\n── MV → AKI ──")
print("Backdoor adjustment set:", identified_mv_to_aki.get_backdoor_variables())

# =============================================================================
# 3. EFFECT ESTIMATION — MULTIPLE ESTIMATORS + CONFIDENCE INTERVALS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: EFFECT ESTIMATION + CONFIDENCE INTERVALS")
print("=" * 70)

def run_estimation(model, identified_estimand, direction_label):
    print(f"\nDirection: {direction_label}")
    est = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.generalized_linear_model",method_params={"glm_family": sm.families.Binomial()},
        confidence_intervals=True,
        test_significance=True, effect_modifiers=[]
    )
    ci = est.get_confidence_intervals()
    return {"estimate": est.value, "ci_low": ci}

results_aki_to_mv = run_estimation(
    model_aki_to_mv, identified_aki_to_mv,
    "AKI Onset (24h) → Mech. Vent Post-24h"
)
results_mv_to_aki = run_estimation(
    model_mv_to_aki, identified_mv_to_aki,
    "Mech. Vent Onset (24h) → AKI Post-24h"
)

# Save
pd.DataFrame([
    {"direction": "AKI Onset (24h) → Mech. Vent Post-24h", **results_aki_to_mv},
    {"direction": "Mech. Vent Onset (24h) → AKI Post-24h", **results_mv_to_aki},
]).to_csv("inference/estimation_results.csv", index=False)
print("\nSaved to inference/estimation_results.csv")