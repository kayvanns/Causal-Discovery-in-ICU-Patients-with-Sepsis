import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
from causallearn.graph.Endpoint import Endpoint

DARK  = "#121212"
WHITE = "#FFFFFF"
RED   = "#C5050C"
GRAY  = "#E1E5E7"

# ── Load graph ────────────────────────────────────────────────────────────────
with open('graphs/FCI_mv_fisherz_raw_v2.pkl', 'rb') as f:
    graph_tuple = pickle.load(f)

causal_graph = graph_tuple[0]
var_names    = graph_tuple[1]
nodes        = causal_graph.get_nodes()
name_of      = {nodes[i]: var_names[i] for i in range(len(var_names))}

LABELS = {
    'anchor_age':            'Age',
    'gender':                'Gender',
    'race':                  'Race',
    'heart_rate_max':        'Heart Rate\n(max)',
    'blood_pressure_min':    'Blood Pressure\n(min)',
    'spO2_min':              'SpO₂\n(min)',
    'FiO2_max':              'FiO₂\n(max)',
    'lactate_max':           'Lactate\n(max)',
    'bilirubin_max':         'Bilirubin\n(max)',
    'platelet_max':          'Platelet\n(max)',
    'inr_max':               'INR\n(max)',
    'temp_max_F':            'Temperature\n(max)',
    'antibiotics_given':     'Antibiotics',
    'vaso_given':            'Vasopressors',
    'aki_24h_onset_stage_y': 'AKI\n(0–24h)',
    'mechvent_24h_onset':    'Mech Vent\n(0–24h)',
    'aki_post24h_stage':     'AKI\n(>24h)',
    'mechvent_post24h':      'Mech Vent\n(>24h)',
    'hospital_expire_flag':  'In-Hospital\nMortality',
}

OUTCOMES = {
    'aki_24h_onset_stage_y', 'mechvent_24h_onset',
    'aki_post24h_stage', 'mechvent_post24h', 'hospital_expire_flag',
}

def node_style(name):
    """Returns (facecolor, textcolor)"""
    if name in OUTCOMES:
        return RED, WHITE
    return GRAY, DARK

POS = {
    'anchor_age':            (1.9, 10.2),
    'gender':                (1.9,  6.8),
    'race':                  (1.9,  3.4),
    'heart_rate_max':        (5.2, 11.4),
    'blood_pressure_min':    (5.2,  9.3),
    'spO2_min':              (5.2,  7.2),
    'FiO2_max':              (5.2,  5.1),
    'temp_max_F':            (5.2,  2.6),
    'lactate_max':           (8.5, 11.4),
    'bilirubin_max':         (8.5,  9.3),
    'platelet_max':          (8.5,  7.2),
    'inr_max':               (8.5,  5.1),
    'antibiotics_given':     (11.8, 10.0),
    'vaso_given':            (11.8,  6.3),
    'aki_24h_onset_stage_y': (15.0,  9.2),
    'mechvent_24h_onset':    (15.0,  6.3),
    'aki_post24h_stage':     (18.3, 10.6),
    'mechvent_post24h':      (18.3,  7.2),
    'hospital_expire_flag':  (18.3,  3.3),
}

FIG_W, FIG_H = 22, 13.5
BOX_W, BOX_H = 1.75, 0.65
CIRCLE_R     = 0.12

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

# Section labels and dividers
SECTIONS = [
    ('Demographics',      0.15,  3.55),
    ('Vitals',            3.55,  6.85),
    ('Lab Values',        6.85, 10.15),
    ('Interventions',    10.15, 13.45),
    ('Clinical Outcomes', 13.45, 21.85),
]
for label, x0, x1 in SECTIONS:
    if label == 'Clinical Outcomes':
        ax.text((x0 + x1) / 2, FIG_H - 0.28, label,
                ha='center', va='top', fontsize=16, fontweight='bold',
                color=RED, zorder=1)
    else:
        ax.text((x0 + x1) / 2, FIG_H - 0.28, label,
                ha='center', va='top', fontsize=16, fontweight='bold',
                color=DARK, zorder=1)
for x_div in [3.55, 6.85, 10.15, 13.45]:
    ax.plot([x_div, x_div], [0, FIG_H],
            color=DARK, lw=0.5, alpha=0.3, zorder=1)

# ── Boundary attachment ───────────────────────────────────────────────────────
def attach(cx, cy, tx, ty, pad=0.07):
    dx, dy = tx - cx, ty - cy
    d = np.hypot(dx, dy)
    if d < 1e-9:
        return cx, cy
    ux, uy = dx / d, dy / d
    t = min(
        (BOX_W / 2 + pad) / abs(ux) if abs(ux) > 1e-9 else 1e9,
        (BOX_H / 2 + pad) / abs(uy) if abs(uy) > 1e-9 else 1e9,
    )
    return cx + t * ux, cy + t * uy

# ── Per-edge curvature ────────────────────────────────────────────────────────
CURVE = {
    ('anchor_age', 'temp_max_F'):                    0,
    ('anchor_age', 'mechvent_24h_onset'):            -0.25,
    ('anchor_age', 'aki_post24h_stage'):             -0.20,
    ('anchor_age', 'mechvent_post24h'):              -0.30,
    ('anchor_age', 'hospital_expire_flag'):          0.38,
    ('heart_rate_max',     'spO2_min'):               0.40,
    ('spO2_min',           'FiO2_max'):              -0.40,
    ('heart_rate_max',     'platelet_max'):           0.15,
    ('heart_rate_max',     'vaso_given'):             0.10,
    ('heart_rate_max',     'hospital_expire_flag'):   0.22,
    ('blood_pressure_min', 'vaso_given'):             -0.1,
    ('spO2_min', 'inr_max'):                          0.15,
    ('spO2_min', 'vaso_given'):                       0.12,
    ('spO2_min', 'aki_24h_onset_stage_y'):           -0.15,
    ('spO2_min', 'hospital_expire_flag'):             0.18,
    ('FiO2_max', 'lactate_max'):                      0.25,
    ('FiO2_max', 'vaso_given'):                       0.12,
    ('FiO2_max', 'mechvent_24h_onset'):               0.18,
    ('antibiotics_given', 'FiO2_max'):               0,
    ('lactate_max',   'bilirubin_max'):               0,
    ('lactate_max',   'inr_max'):                    -0.55,
    ('bilirubin_max', 'platelet_max'):               0,
    ('bilirubin_max', 'inr_max'):                     0.40,
    ('platelet_max',  'inr_max'):                    -0.35,
    ('bilirubin_max', 'aki_24h_onset_stage_y'):       0.12,
    ('bilirubin_max', 'hospital_expire_flag'):        0.18,
    ('platelet_max', 'vaso_given'):                   0.12,
    ('platelet_max', 'aki_24h_onset_stage_y'):       -0.12,
    ('inr_max', 'aki_24h_onset_stage_y'):            -0.18,
    ('inr_max', 'hospital_expire_flag'):              0.12,
    ('vaso_given', 'aki_24h_onset_stage_y'):         -0.15,
    ('vaso_given', 'hospital_expire_flag'):           0.15,
    ('mechvent_24h_onset',    'aki_post24h_stage'):  -0.20,
    ('aki_24h_onset_stage_y', 'mechvent_post24h'):    0.15,
    ('aki_24h_onset_stage_y', 'hospital_expire_flag'): 0.18,
    ('mechvent_post24h',  'aki_post24h_stage'):       0.40,
    ('mechvent_post24h',  'hospital_expire_flag'):   -0.42,
    ('aki_post24h_stage', 'hospital_expire_flag'):    0.40,
}

def get_rad(n1, n2):
    return CURVE.get((n1, n2), CURVE.get((n2, n1), 0.0))

# ── Draw edges ────────────────────────────────────────────────────────────────
for e in causal_graph.get_graph_edges():
    n1  = name_of[e.get_node1()]
    n2  = name_of[e.get_node2()]
    ep1 = e.get_endpoint1()
    ep2 = e.get_endpoint2()

    cx1, cy1 = POS[n1]
    cx2, cy2 = POS[n2]
    px1, py1 = attach(cx1, cy1, cx2, cy2)
    px2, py2 = attach(cx2, cy2, cx1, cy1)

    conn  = f'arc3,rad={get_rad(n1, n2)}'
    bidir = (ep1 == Endpoint.ARROW and ep2 == Endpoint.ARROW)
    circle = (ep1 == Endpoint.CIRCLE)
    highlight = {
    ('aki_24h_onset_stage_y', 'mechvent_post24h'),
    ('mechvent_24h_onset', 'aki_post24h_stage'),
}
    if (n1, n2) in highlight or (n2, n1) in highlight:
        base = dict(color=RED, lw=2.0, zorder=2, alpha=0.9)
    else:
        base = dict(color=DARK, lw=1.2, zorder=2, alpha=0.65)


    if bidir:
        ax.add_patch(FancyArrowPatch(
            (px1, py1), (px2, py2),
            arrowstyle='<|-|>', mutation_scale=30,
            connectionstyle=conn, **base,
        ))
    else:
        ax.add_patch(FancyArrowPatch(
            (px1, py1), (px2, py2),
            arrowstyle='-|>', mutation_scale=30,
            connectionstyle=conn, **base,
        ))
        if circle:
            ax.add_patch(plt.Circle(
                (px1, py1), CIRCLE_R,
                color=DARK, fill=False, lw=1.2, alpha=0.65, zorder=5,
            ))

# ── Draw nodes ────────────────────────────────────────────────────────────────
for name, (cx, cy) in POS.items():
    fc, tc = node_style(name)
    ax.add_patch(FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle='round,pad=0.10', zorder=3,
        facecolor=fc, edgecolor=DARK, linewidth=1.2,
    ))
    ax.text(cx, cy, LABELS[name],
            ha='center', va='center', fontsize=15.5, fontweight='bold',
            color=tc, zorder=4, linespacing=1.3)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=DARK, lw=1.4, label='Directed (→)',
           marker='>', markersize=5, markerfacecolor=DARK),
    Line2D([0], [0], color=DARK, lw=1.4, label='Bidirected (↔) hidden common cause',
           marker='>', markersize=5, markerfacecolor=DARK),
    Line2D([0], [0], color=DARK, lw=1.4, label='Uncertain Tail  (○→)',
           marker='o', markersize=5, markerfacecolor='none', markeredgecolor=DARK),
    mpatches.Patch(facecolor=GRAY, edgecolor=DARK, label='Clinical Variables and Treatments'),
    mpatches.Patch(facecolor=RED,  edgecolor=DARK, label='Clinical Outcomes'),
]

leg = ax.legend(
    handles=legend_handles,
    loc='lower left', fontsize=14,
    framealpha=0.90, edgecolor=DARK,
    title='Edge Types & Node Categories', title_fontsize=16,
)
leg.get_title().set_color(DARK)


plt.tight_layout()
out = 'poster_figures/poster_graph.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=WHITE)

