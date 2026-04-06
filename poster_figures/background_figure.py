import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np

# Palette
DARK  = "#121212"
WHITE = "#FFFFFF"
RED   = "#C5050C"
LIGHT = "#E1E5E7"

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis('off')
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

# ── Causal Analysis outer box ─────────────────────────────────────────────────
ca_x, ca_y, ca_w, ca_h = 0.3, 0.4, 10.4, 5.2
ax.add_patch(FancyBboxPatch(
    (ca_x, ca_y), ca_w, ca_h,
    boxstyle='round,pad=0.12',
    facecolor=LIGHT, edgecolor=RED, linewidth=1.75
))
ax.text(ca_x + ca_w / 2, ca_y + ca_h - 0.28,
        'Causal Analysis', ha='center', va='top',
        fontsize=16, fontweight='bold', color=RED, zorder=3)

# ── Correlation vs. Causation note (minimal) ──────────────────────────────────
ax.text(ca_x + 0.22, ca_y + 0.22,
        'correlation  ≠  causation',
        ha='left', va='bottom',
        fontsize=10, color=DARK,fontstyle='italic')

# ── Causal Inference large ellipse (left-center) ─────────────────────────────
ci_cx, ci_cy, ci_rx, ci_ry = 4.05, 2.82, 3.55, 2.18
ax.add_patch(Ellipse(
    (ci_cx, ci_cy), 2 * ci_rx, 2 * ci_ry,
    facecolor=WHITE, edgecolor=DARK, linewidth=1.6, zorder=3,
))
ax.text(ci_cx, ci_cy + ci_ry - 0.28,
        'Causal Inference', ha='center', va='top',
        fontsize=14, fontweight='bold', color=DARK, zorder=4)
ax.text(ci_cx + 1, ci_cy - 0.3,
        'Estimate causal effect\n of treament on outcome', ha='center', va='center',
        fontsize=8.5, color=DARK, zorder=4)

# ── RCTs small ellipse (nested inside Causal Inference) ──────────────────────
rct_cx, rct_cy, rct_rx, rct_ry = 2.8, 2.6, 1.45, 1.35
ax.add_patch(Ellipse(
    (rct_cx, rct_cy), 2 * rct_rx, 2 * rct_ry,
    facecolor=LIGHT, edgecolor=DARK, linewidth=1.2,
    linestyle='--', zorder=4,
))
ax.text(rct_cx, rct_cy + 0.22,
        'RCTs', ha='center', va='center',
        fontsize=12, fontweight='bold', color=DARK, zorder=5)
ax.text(rct_cx, rct_cy - 0.30,
        'Gold standard but\n often infeasible or unethical', ha='center', va='center',
        fontsize=7.2, color=DARK, zorder=5)

# ── Causal Discovery ellipse (right, highlighted) ────────────────────────────
cd_cx, cd_cy, cd_rx, cd_ry = 7.75, 2.82, 1.9, 2
ax.add_patch(Ellipse(
    (cd_cx, cd_cy), 2 * cd_rx, 2 * cd_ry,
    facecolor=RED, edgecolor=RED, linewidth=1.8, zorder=3,
))
ax.text(cd_cx, cd_cy + cd_ry - 0.5,
        'Causal Discovery', ha='center', va='top',
        fontsize=10.5, fontweight='bold', color=WHITE,
        linespacing=1.3, zorder=4)
ax.text(cd_cx, cd_cy - 0.22,
        'Learn causal graph\nstructure from data', ha='center', va='center',
        fontsize=8.5, color=WHITE, zorder=4)

ax.text(6.3, 2.82,
        'Observational\nMethods',
        ha='center', va='center',
        fontsize=7.5, color=WHITE,
        fontstyle='italic', zorder=5)


plt.tight_layout()
out = 'poster_figures/background_figure.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=WHITE)

