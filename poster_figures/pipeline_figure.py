import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

RED      = "#C5050C"
WHITE    = "#FFFFFF"
LTGRAY   = "#E1E5E7"
DARK     = "#121212"


FIG_W, FIG_H = 32, 8.5
BOX_W, BOX_H = 3.6, 1.15

ARROW_KW = dict(arrowstyle="-|>", color=DARK, lw=1.6, mutation_scale=16, zorder=3)

# (label lines, cx, cy, fill)
STAGES = [
    (["MIMIC-IV"],                      2.4,  4.6, LTGRAY),
    (["Cohort", "Definition"],          6.8,  4.6, LTGRAY),
    (["Feature Engineering", "& Merge"],11.2,  4.6, LTGRAY),
    (["Missing Data", "Handling"],     15.6,  4.6, LTGRAY),
    (["Causal", "Discovery"],          20.0,  4.6, RED),
    (["Ensemble", "Aggregation"],      24.4,  4.6, RED),
    (["Directionality", "Analysis"],     28.8,  4.6, LTGRAY),
]

SUB = [
    "94,458 ICU stays · 546,028 hospitalizations\nEHR & MetaVision ICU modules\n364,627 unique individuals",
    "Sepsis-3: suspected infection + SOFA ≥ 2\nAKI stage & mechanical ventilation flags\n→ 33,000 unique ICU stays",
    "Vitals · antibiotics · vasopressors\ndemographics · hospital mortality\nmerged with cohort on stay_id",
    "Mean & KNN imputation\nmissingness indicators (MNAR)\n→ 3 dataset variants",
    "PC & FCI · Fisher-Z & Kernel CI\ntemporal tier background knowledge\n14 algorithm runs",
    "Edge agreement across\nalgo × CI test × data variant\ndirect & indirect path scoring",
    "Logistic regression\nadjusted odds ratios\nindirect path analysis",
]



fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor(WHITE)

box_centers = []

for (lines, cx, cy, fill) in STAGES:
    x0 = cx - BOX_W / 2
    y0 = cy - BOX_H / 2
    is_red = fill == RED
    edge_col = RED if is_red else DARK
    text_col = WHITE if is_red else DARK
    lw = 2.2 if is_red else 1.3

    rect = FancyBboxPatch(
        (x0, y0), BOX_W, BOX_H,
        boxstyle="round,pad=0.13",
        linewidth=lw, edgecolor=edge_col, facecolor=fill, zorder=2
    )
    ax.add_patch(rect)

    label = "\n".join(lines)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=22, fontweight="bold", color=text_col, zorder=4,
            linespacing=1.35)
    box_centers.append(cx)

# Sub-labels below each box
for idx, ((_, cx, cy, _), sub) in enumerate(zip(STAGES, SUB)):
    if idx % 2 == 0:
        # even stages — text below the box
        ax.text(cx, cy - BOX_H / 2 - 0.4, sub,
                ha="center", va="top", fontsize=24, color=DARK,
                linespacing=1.38, zorder=4)
    else:
        # odd stages — text above the box
        ax.text(cx, cy + BOX_H / 2 + 0.4, sub,
                ha="center", va="bottom", fontsize=24, color=DARK,
                linespacing=1.38, zorder=4)

# Arrows 
for i in range(len(STAGES) - 1):
    x_start = box_centers[i] + BOX_W / 2
    x_end   = box_centers[i + 1] - BOX_W / 2
    y       = STAGES[i][2]
    ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                arrowprops=ARROW_KW, zorder=3)
 



plt.tight_layout(pad=0.3)
plt.savefig("poster_figures/pipeline_figure.png", dpi=300, bbox_inches="tight", facecolor=WHITE)

