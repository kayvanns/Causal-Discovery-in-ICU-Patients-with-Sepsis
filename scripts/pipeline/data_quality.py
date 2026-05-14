import pandas as pd
import numpy as np

df = pd.read_csv("/Users/kayvans/Documents/sepsis-causal-discovery/data/processed/analysis.csv")

DEMO_COLS   = ["anchor_age", "gender", "race", "ckd_baseline", "BMI"]
PHYS_COLS   = ["Heart Rate (max)", "Blood Pressure (min)", "SpO2 (min)", "FiO2 (max)",
               "Lactate (max)", "Bilirubin (max)", "Platelet (max)", "INR (max)",
               "Temperature (max)", "Hemoglobin (min)", "Lymphocytes Abs (min)",
               "Fluid Balance (mL)", "CVP (max)", "WBC (max)"]
TREAT_COLS  = ["Antibiotics", "Vasopressors", "Diuretics"]
OUT24_COLS  = ["aki_24h_onset_stage", "mechvent_24h_onset", "aki_24h_onset"]
POST24_COLS = ["aki_post24h_stage", "mechvent_post24h", "aki_post24h"]
MORT_COLS   = ["hospital_expire_flag"]

ALL_COLS = DEMO_COLS + PHYS_COLS + TREAT_COLS + OUT24_COLS + POST24_COLS + MORT_COLS
available = [c for c in ALL_COLS if c in df.columns]
df = df[available]

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(available)}")
print()

print("=== MISSINGNESS ===")
miss = df.isnull().mean().sort_values(ascending=False)
miss_df = pd.DataFrame({
    "missing_pct": (miss * 100).round(1),
    "missing_n":   df.isnull().sum()
})
print(miss_df[miss_df["missing_pct"] > 0].to_string())
print()

print("=== CONTINUOUS VARIABLE DISTRIBUTIONS ===")
cont_cols = [c for c in PHYS_COLS + ["anchor_age"] if c in df.columns]
print(df[cont_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(2).to_string())
print()

print("=== BINARY AND CATEGORICAL DISTRIBUTIONS ===")
cat_cols = TREAT_COLS + OUT24_COLS + POST24_COLS + MORT_COLS + ["gender", "race", "ckd_baseline"]
for col in cat_cols:
    if col in df.columns:
        print(f"{col}: {df[col].value_counts().to_dict()} | missing: {df[col].isnull().sum()}")
print()

print("=== OUTLIER COUNTS (post source-filtering) ===")
bounds = {
    "Blood Pressure (min)":   (0, 200),
    "Heart Rate (max)":       (0, 300),
    "SpO2 (min)":             (50, 100),
    "FiO2 (max)":             (21, 100),
    "Temperature (max)":      (85, 115),
    "Lactate (max)":          (0, 30),
    "Bilirubin (max)":        (0, 100),
    "Platelet (max)":         (0, 2000),
    "INR (max)":              (0, 20),
    "Hemoglobin (min)":       (0, 25),
    "WBC (max)":              (0, 500),
    "Lymphocytes Abs (min)":  (0, 20),
    "Fluid Balance (mL)":     (-10000, 20000),
    "CVP (max)":              (-5, 30),
    "BMI":                    (10, 80),
    "anchor_age":             (18, 91),
}
for col, (lo, hi) in bounds.items():
    if col not in df.columns:
        continue
    n_low  = (df[col] < lo).sum()
    n_high = (df[col] > hi).sum()
    total  = df[col].notna().sum()
    if n_low > 0 or n_high > 0:
        print(f"{col}: {n_low} below {lo}, {n_high} above {hi} (out of {total} non-null)")
print("Done — no output above means all values within bounds")