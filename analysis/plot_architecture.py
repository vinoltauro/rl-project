"""
Generate CNN architecture diagram as a PNG using matplotlib.
Saves to results/plots/architecture.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 3.8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis("off")

# ── colour palette ─────────────────────────────────────────────────────────
C_INPUT  = "#E8E8E8"
C_CONV   = "#C5D8F0"
C_FLAT   = "#FCE4B0"
C_REPR   = "#B6E8B6"
C_OUT    = "#F5BBBB"
BORDER   = "#555555"
BLUE     = "#0569B9"

# ── layer definitions  [x_centre, label_lines, fill, width] ────────────────
layers = [
    (1.0,  ["Input", "4×84×84"],                          C_INPUT, 1.5),
    (3.0,  ["Conv1", "32, 8×8, s4", "ReLU"],              C_CONV,  1.7),
    (5.0,  ["Conv2", "64, 4×4, s2", "ReLU"],              C_CONV,  1.7),
    (7.0,  ["Conv3", "64, 3×3, s1", "ReLU"],              C_CONV,  1.7),
    (9.0,  ["Flatten", "3,136"],                           C_FLAT,  1.5),
    (11.0, ["fc_repr", "512-dim", "ReLU"],                 C_REPR,  1.7),
    (13.0, ["fc_out", "n actions", "Q-values"],            C_OUT,   1.7),
]

BOX_H   = 1.55
BOX_Y   = (4 - BOX_H) / 2       # vertically centred

node_positions = {}

for x, lines, fill, w in layers:
    x0 = x - w / 2
    lw = 2.5 if fill == C_REPR else 1.2
    box = FancyBboxPatch((x0, BOX_Y), w, BOX_H,
                         boxstyle="round,pad=0.05",
                         facecolor=fill, edgecolor=BORDER, linewidth=lw,
                         zorder=3)
    ax.add_patch(box)

    # text inside box
    n = len(lines)
    for i, line in enumerate(lines):
        yoff = BOX_Y + BOX_H/2 + (n/2 - i - 0.5) * 0.36
        bold = (i == 0)
        ax.text(x, yoff, line,
                ha="center", va="center",
                fontsize=9.5 if bold else 8.5,
                fontweight="bold" if bold else "normal",
                color="#111111", zorder=4)

    node_positions[x] = (x0, x0 + w)

# ── arrows between boxes ──────────────────────────────────────────────────
mid_y = BOX_Y + BOX_H / 2
xs = [l[0] for l in layers]
for i in range(len(xs) - 1):
    x_start = node_positions[xs[i]][1]     # right edge of current
    x_end   = node_positions[xs[i+1]][0]   # left edge of next
    ax.annotate("",
        xy=(x_end, mid_y), xytext=(x_start, mid_y),
        arrowprops=dict(arrowstyle="-|>", color=BORDER,
                        lw=1.4, mutation_scale=14),
        zorder=5)

# ── output size labels below conv nodes ──────────────────────────────────
size_labels = {3.0: "32×20×20", 5.0: "64×9×9", 7.0: "64×7×7"}
for x, lbl in size_labels.items():
    ax.text(x, BOX_Y - 0.22, lbl, ha="center", va="top",
            fontsize=7.8, color="#666666")

# ── forward hook annotation above fc_repr ────────────────────────────────
hook_x = 11.0
hook_top = BOX_Y + BOX_H + 0.12
ax.annotate("",
    xy=(hook_x, BOX_Y + BOX_H),
    xytext=(hook_x, hook_top + 0.45),
    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3,
                    mutation_scale=12))
ax.text(hook_x, hook_top + 0.52, "forward hook\n(all analysis here)",
        ha="center", va="bottom", fontsize=8.5,
        color=BLUE, fontweight="bold")

# ── backbone bracket ──────────────────────────────────────────────────────
bx0 = node_positions[3.0][0] - 0.08
bx1 = node_positions[7.0][1] + 0.08
by0 = BOX_Y - 0.45
bracket_y = by0 - 0.12
ax.plot([bx0, bx0, bx1, bx1],
        [BOX_Y, bracket_y, bracket_y, BOX_Y],
        color=BLUE, lw=1.3, alpha=0.6)
ax.text((bx0 + bx1) / 2, bracket_y - 0.12,
        "Convolutional backbone",
        ha="center", va="top", fontsize=8.5,
        color=BLUE, fontstyle="italic")

plt.tight_layout(pad=0.3)
out = "results/plots/architecture.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out}")
