import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("total_ensemble_table.csv")
df = df[["edge","cause","effect","agreement_score"]]
df = df.replace("aki_24h_onset_stage_y", "aki_24h_onset_stage")
print(df.head())

all_vars = sorted(set(df["cause"].tolist() + df["effect"].tolist()))


core_order = [
    "Age", "Gender", "Race",
    "Heart Rate (max)", "Blood Pressure (min)", "SpO2 (min)", "FiO2 (max)",
    "Lactate (max)", "Bilirubin (max)", "Platelet (max)", "INR (max)", "Temperature (max)",
    "Antibiotics", "Vasopressors",
    "AKI Onset (24h)", "Mech. Vent Onset (24h)",
    "AKI Post-24h", "Mech. Vent Post-24h",
    "Hospital Mortality"
]

indicator_vars = [v for v in all_vars if v not in core_order]

full_order = [v for v in core_order if v in all_vars] + sorted(indicator_vars)


matrix = df.pivot_table(
    index="effect",
    columns="cause",
    values="agreement_score",
    aggfunc="max",
    fill_value=0
)

matrix = matrix.reindex(index=full_order, columns=full_order, fill_value=0)
print(matrix)

from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

colors = ["#FFFFFF", "#FADBD8", "#F1948A", "#EF6657", 
          "#F04E3C", "#C5050C","#9B0000"]
cmap = LinearSegmentedColormap.from_list("uw", colors, N=7)
bounds = [0, 2, 4, 6, 8, 10, 12, 14]
norm = BoundaryNorm(bounds, ncolors=len(colors))

fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(
    matrix,
    ax=ax,
    cmap=cmap,
    norm=norm,
    linewidths=0.5,
    linecolor="#E1E5E7",
    cbar_kws={"label": "Agreement Score", "shrink": 0.5},
)
ax.set_xlabel("From", color= '#121212', fontsize=12, fontweight="bold")
ax.set_ylabel("To", color= '#121212', fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig("poster_figures/heatmap.png", dpi=300)