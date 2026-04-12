import matplotlib.pyplot as plt
from matplotlib.patches import Circle

DARK  = "#121212"
WHITE = "#FFFFFF"
RED   = "#C5050C"
GRAY  = "#E1E5E7"

fig, ax = plt.subplots(figsize=(3, 5))
ax.set_xlim(0, 3)
ax.set_ylim(0, 5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(WHITE)

ci_center = (1.02, 2.5)
cd_center = (1.98, 2.5)
radius = 1

ci_circle = Circle(ci_center, radius, color=GRAY, alpha=0.7, zorder=2)
cd_circle = Circle(cd_center, radius, color=RED, alpha=0.2, zorder=2)
ax.add_patch(ci_circle)
ax.add_patch(cd_circle)

ci_circle_edge = Circle(ci_center, radius, fill=False, edgecolor=DARK, linewidth=1.5, zorder=3)
cd_circle_edge = Circle(cd_center, radius, fill=False, edgecolor=RED, linewidth=1.5, zorder=3)
ax.add_patch(ci_circle_edge)
ax.add_patch(cd_circle_edge)

ax.text(0.55, 2.5, 'Causal\nInference', ha='center', va='center',
        fontsize=9, fontweight='bold', color=DARK, zorder=5)
ax.text(2.45, 2.5, 'Causal\nDiscovery', ha='center', va='center',
        fontsize=9, fontweight='bold', color=RED, zorder=5)

ax.text(1.5, 2.5, 'Observational\nData', ha='center', va='center',
        fontsize=8, color=DARK, zorder=5)

rct_circle = Circle((0.72,3.07), 0.25, color=DARK, alpha=0.15, zorder=4)
ax.add_patch(rct_circle)
rct_edge = Circle((0.72, 3.07), 0.25, fill=False, edgecolor=DARK, linewidth=1.0, zorder=4, linestyle='--')
ax.add_patch(rct_edge)
ax.text(0.72,3.07, 'RCTs', ha='center', va='center',
        fontsize=7.5, fontweight='bold', color=DARK, zorder=5)

plt.tight_layout(pad=0.3)
plt.savefig('poster_figures/venn_diagram.png', dpi=150, bbox_inches='tight',
            facecolor=WHITE)

