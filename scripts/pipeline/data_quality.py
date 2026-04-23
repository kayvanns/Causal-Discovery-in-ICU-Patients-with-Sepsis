import pandas as pd
import numpy as np

df = pd.read_csv("/Users/kayvans/Documents/sepsis-causal-discovery/data/processed/analysis.csv")
df = df.rename(columns={"cvp": "cvp_max"})
print("=== Shape ===")
print(df.shape)

print("\n=== Missing rates ===")
print(df.isnull().mean().sort_values(ascending=False).round(3))

print("\n=== Continuous variables ===")
continuous = [
    "blood_pressure_min", "heart_rate_max", "spO2_min", "FiO2_max",
    "temp_max_F", "lactate_max", "bilirubin_max", "platelet_max",
    "inr_max", "hemoglobin_min", "wbc_max", "lymphocyte_abs_min", "cvp_max", "fluid_balance", "sofa_score",
    "anchor_age"
]
print(df[continuous].describe().round(2))

print("\n=== Implausible values check ===")
checks = {
    "blood_pressure_min":  (0, 200),
    "heart_rate_max":      (0, 300),
    "spO2_min":            (50, 100),
    "FiO2_max":            (21, 100),
    "temp_max_F":          (85, 115),
    "lactate_max":         (0, 30),
    "bilirubin_max":       (0, 100),
    "platelet_max":        (0, 2000),
    "inr_max":             (0, 20),
    "hemoglobin_min":      (0, 25),
    "wbc_max":             (0, 500),
    "lymphocyte_abs_min":  (0, 20),
    "cvp_max":             (-5, 50),
    "fluid_balance":       (-10000, 20000),
    "sofa_score":          (0, 24),
    "anchor_age":          (18, 91),
}
for col, (lo, hi) in checks.items():
    if col not in df.columns:
        print(f"  {col}: NOT IN DATAFRAME")
        continue
    n_low  = (df[col] < lo).sum()
    n_high = (df[col] > hi).sum()
    if n_low > 0 or n_high > 0:
        print(f"  {col}: {n_low} below {lo}, {n_high} above {hi}")
    else:
        print(f"  {col}: OK")

print("\n=== Binary variables ===")
binary = [
    "gender", "hospital_expire_flag", "antibiotics_given", "vaso_given",
    "aki_24h_onset_stage", "mechvent_24h_onset", "aki_post24h_stage",
    "mechvent_post24h", "diuretics_given", "ckd_baseline"
]
for col in binary:
    if col not in df.columns:
        print(f"  {col}: NOT IN DATAFRAME")
        continue
    print(f"  {col}: {df[col].value_counts().to_dict()} | missing: {df[col].isnull().sum()}")

    # blood pressure cannot be negative
df["blood_pressure_min"] = df["blood_pressure_min"].where(df["blood_pressure_min"] > 0, np.nan)

# heart rate above 300 is impossible in a living patient
df["heart_rate_max"] = df["heart_rate_max"].where(df["heart_rate_max"] <= 300, np.nan)

# SpO2 cannot be below 0 or meaningfully below 50 in a monitored patient
df["spO2_min"] = df["spO2_min"].where(df["spO2_min"] >= 50, np.nan)

# FiO2 cannot be below 21 (room air) or above 100
df["FiO2_max"] = df["FiO2_max"].clip(lower=21, upper=100)

# lactate above 30 is a data error
df["lactate_max"] = df["lactate_max"].where(df["lactate_max"] <= 30, np.nan)

# INR above 20 is physiologically implausible
df["inr_max"] = df["inr_max"].where(df["inr_max"] <= 20, np.nan)

df.to_csv("/Users/kayvans/Documents/sepsis-causal-discovery/data/processed/analysis_cleaned.csv", index=False)