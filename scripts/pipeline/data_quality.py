import pandas as pd
import numpy as np

df = pd.read_csv("/Users/kayvans/Documents/sepsis-causal-discovery/data/processed/analysis.csv")

DEMO_COLS   = ["anchor_age", "gender", "race", "ckd_baseline", "BMI"]
PHYS_COLS   = ["Heart Rate (max)", "Blood Pressure (min)", "SpO2 (min)", "FiO2 (max)",
               "Lactate (max)", "Bilirubin (max)", "Platelet (max)", "INR (max)",
               "Temperature (max)", "Hemoglobin (min)", "Lymphocytes Abs (min)",
               "Fluid Balance (mL)", "CVP (max)", "WBC (max)"]
TREAT_COLS  = ["Antibiotics", "Vasopressors", "Diuretics"]
BINARY_COLS = ["aki_24h_onset", "mechvent_24h_onset", "aki_post24h", "mechvent_post24h",
               "hospital_expire_flag", "ckd_baseline", "Antibiotics", "Vasopressors", "Diuretics"]
CONT_COLS   = ["anchor_age", "BMI"] + PHYS_COLS

ALL_COLS = DEMO_COLS + PHYS_COLS + TREAT_COLS + \
           ["aki_24h_onset", "mechvent_24h_onset", "aki_post24h", 
            "mechvent_post24h", "hospital_expire_flag"]
available = [c for c in ALL_COLS if c in df.columns]
df = df[available]

print(f"Cohort: {len(df):,} ICU stays\n")

# ── Missingness ───────────────────────────────────────────────────────────────
print("=" * 55)
print("MISSINGNESS")
print("=" * 55)
miss = df.isnull().mean() * 100
miss = miss[miss > 0].sort_values(ascending=False)
for col, pct in miss.items():
    print(f"  {col:<30} {pct:.1f}%  (n={df[col].isnull().sum():,})")
print()

# ── Binary proportions ────────────────────────────────────────────────────────
print("=" * 55)
print("BINARY VARIABLES")
print("=" * 55)
for col in BINARY_COLS:
    if col not in df.columns:
        continue
    n      = df[col].notna().sum()
    prop   = df[col].mean() * 100
    print(f"  {col:<30} {prop:.1f}%  (n={n:,})")
print()

# ── Continuous summaries ──────────────────────────────────────────────────────
print("=" * 55)
print("CONTINUOUS VARIABLES")
print("=" * 55)
cont_available = [c for c in CONT_COLS if c in df.columns]
summary = df[cont_available].agg(["mean", "min", "median", "max"]).round(2)
summary.loc["Q1"] = df[cont_available].quantile(0.25).round(2)
summary.loc["Q3"] = df[cont_available].quantile(0.75).round(2)
summary.loc["missing_%"] = (df[cont_available].isnull().mean() * 100).round(1)
print(summary.to_string())