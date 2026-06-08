"""
Generate CNN architecture diagram as a PNG using matplotlib.
Saves to results/plots/architecture.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(15, 3.0))
ax.set_xlim(0, 14)
ax.set_ylim(0, 3.2)
ax.axis("off")

# ── colour palette ────────────────────────────────────────────────────────
C_INPUT = "#EBEBEB"
C_CONV  = "#C5D8F0"
C_FLAT  = "#FCE4B0"
C_REPR  = "#B6E8B6"
C_OUT   = "#F5BBBB"
BORDER  = "#444444"
BLUE    = "#0569B9"

# ── layer definitions [x_centre, label_lines, fill, width] ───────────────
layers = [
    (1.0,  ["Input", "4×84×84"],              C_INPUT, 1.5),
    (3.0,  ["Conv1", "32, 8×8, s4", "ReLU"],  C_CONV,  1.7),
    (5.0,  ["Conv2", "64, 4×4, s2", "ReLU"],  C_CONV,  1.7),
    (7.0,  ["Conv3", "64, 3×3, s1", "ReLU"],  C_CONV,  1.7),
    (9.0,  ["Flatten", "3,136"],               C_FLAT,  1.5),
    (11.0, ["fc_repr", "512-dim", "ReLU"],     C_REPR,  1.7),
    (13.0, ["fc_out", "n actions", "Q-values"],C_OUT,   1.7),
]

BOX_H = 1.35
BOX_Y = 0.82       # leave room below for bracket, above for hook

node_positions = {}

for x, lines, fill, w in layers:
    x0 = x - w / 2
    lw = 2.8 if fill == C_REPR else 1.3
    box = FancyBboxPatch((x0, BOX_Y), w, BOX_H,
                         boxstyle="round,pad=0.06",
                         facecolor=fill, edgecolor=BORDER,
                         linewidth=lw, zorder=3)
    ax.add_patch(box)

    n = len(lines)
    spacing = 0.34
    for i, line in enumerate(lines):
        yoff = BOX_Y + BOX_H/2 + ((n - 1)/2 - i) * spacing
        bold = (i == 0)
        ax.text(x, yoff, line,
                ha="center", va="center",
                fontsize=11 if bold else 10,
                fontweight="bold" if bold else "normal",
                color="#111111", zorder=4)

    node_positions[x] = (x0, x0 + w)

# ── arrows ────────────────────────────────────────────────────────────────
mid_y = BOX_Y + BOX_H / 2
xs = [l[0] for l in layers]
for i in range(len(xs) - 1):
    x_start = node_positions[xs[i]][1]
    x_end   = node_positions[xs[i+1]][0]
    ax.annotate("",
        xy=(x_end, mid_y), xytext=(x_start, mid_y),
        arrowprops=dict(arrowstyle="-|>", color=BORDER,
                        lw=1.5, mutation_scale=15),
        zorder=5)

# ── output size labels below conv nodes ──────────────────────────────────
size_labels = {3.0: "32×20×20", 5.0: "64×9×9", 7.0: "64×7×7"}
for x, lbl in size_labels.items():
    ax.text(x, BOX_Y - 0.08, lbl, ha="center", va="top",
            fontsize=9, color="#666666")

# ── forward hook annotation above fc_repr (compact) ──────────────────────
hook_x   = 11.0
hook_bot = BOX_Y + BOX_H
hook_top = hook_bot + 0.28
ax.annotate("",
    xy=(hook_x, hook_bot),
    xytext=(hook_x, hook_top + 0.08),
    arrowprops=dict(arrowstyle="-|>", color=BLUE,
                    lw=1.4, mutation_scale=13))
ax.text(hook_x, hook_top + 0.12,
        "forward hook  (all analysis here)",
        ha="center", va="bottom",
        fontsize=9.5, color=BLUE, fontweight="bold")

# ── backbone bracket ──────────────────────────────────────────────────────
bx0 = node_positions[3.0][0] - 0.10
bx1 = node_positions[7.0][1] + 0.10
bk_y = BOX_Y - 0.30
tick = 0.10
ax.plot([bx0, bx0, bx1, bx1],
        [BOX_Y, bk_y, bk_y, BOX_Y],
        color=BLUE, lw=1.4, alpha=0.65, solid_capstyle="round")
ax.text((bx0 + bx1) / 2, bk_y - 0.07,
        "Convolutional backbone",
        ha="center", va="top",
        fontsize=9.5, color=BLUE, fontstyle="italic")

plt.tight_layout(pad=0.2)
out = "results/plots/architecture.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out}")
