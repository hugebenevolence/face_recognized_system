"""System architecture figure for the user guide (two panels).

Drawn in the visual language of the schematics in the figures4papers repository:
panel letters, pastel semantic colours with dark edges, trapezoid network blocks, embedding
vectors drawn as coloured bars, hollow block arrows, dashed grouping boxes, bold coloured
headings and small illustrations instead of text-only boxes. Technology logos mark the stack.

a  Real-time recognition pipeline
b  Services and deployment (Docker Compose)
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT, ASSETS = sys.argv[1], sys.argv[2]

C = {
    "blue": "#0F4D92",
    "blue2": "#3775BA",
    "blue_fill": "#CDE3F7",
    "blue_tint": "#EAF1F9",
    "green": "#2E8B4F",
    "green_fill": "#D4EBD0",
    "green_mid": "#8BCF8B",
    "teal": "#42949E",
    "teal_fill": "#CFE7EA",
    "violet": "#9A4D8E",
    "violet_fill": "#E9D3E5",
    "gray_fill": "#E4E3E3",
    "gray": "#767676",
    "text": "#4D4D4D",
    "edge": "#272727",
    "red": "#B64342",
    "docker": "#2496ED",
    "cyan": "#00D5E8",
}

plt.rcParams.update(
    {
        "font.family": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

W, H = 16.0, 8.4
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


# ------------------------------------------------------------------ drawing helpers
def logo(name, cx, cy, h, z=6):
    img = plt.imread(f"{ASSETS}/tech/{name}.png")
    w = h * img.shape[1] / img.shape[0]
    ax.imshow(img, extent=[cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2], zorder=z, interpolation="bilinear")


def rbox(cx, cy, w, h, fill, edge=C["edge"], lw=2.0, r=0.14, ls="-", z=2):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            linestyle=ls,
            zorder=z,
        )
    )


def text(x, y, s, size=13, color=C["text"], weight="normal", style="normal", ha="center", va="center", z=8):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, fontstyle=style, ha=ha, va=va, zorder=z, linespacing=1.15)


def fat_arrow(x0, x1, y, color="#BFC9D3", head=0.2, shaft=0.11, z=3):
    """Hollow block arrow as used between stages in the reference schematics."""
    L = x1 - x0
    hl = min(head, L * 0.6)
    pts = [(x0, y - shaft), (x1 - hl, y - shaft), (x1 - hl, y - shaft * 2.1), (x1, y), (x1 - hl, y + shaft * 2.1), (x1 - hl, y + shaft), (x0, y + shaft)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=C["text"], linewidth=1.6, zorder=z))


def line_arrow(p, q, both=False, color=C["text"], lw=2.0, rad=0.0, ls="-", z=4):
    ax.add_patch(
        FancyArrowPatch(
            p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=14, linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0, zorder=z,
        )
    )


def trapezoid(cx, cy, w, h_left, h_right, fill, edge=C["edge"], z=3):
    pts = [(cx - w / 2, cy - h_left / 2), (cx - w / 2, cy + h_left / 2), (cx + w / 2, cy + h_right / 2), (cx + w / 2, cy - h_right / 2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=edge, linewidth=2.0, zorder=z))


def embedding_bar(cx, cy, w, h, n=8, z=5):
    """A feature vector drawn as stacked cells, as in the reference figures."""
    ch = h / n
    rng = np.random.default_rng(3)
    for i in range(n):
        shade = 0.45 + 0.5 * rng.random()
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + i * ch), w, ch, facecolor=C["green_mid"], alpha=shade, edgecolor=C["edge"], linewidth=1.0, zorder=z))
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor="none", edgecolor=C["edge"], linewidth=1.8, zorder=z + 1))


def cylinder(cx, cy, w, h, fill, edge=C["edge"], z=3):
    e = 0.17  # half height of the ellipses
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + e), w, h - 2 * e, facecolor=fill, edgecolor="none", zorder=z))
    ax.add_patch(Ellipse((cx, cy - h / 2 + e), w, 2 * e, facecolor=fill, edgecolor=edge, linewidth=2.0, zorder=z))
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + e), w, h - 2 * e, facecolor=fill, edgecolor="none", zorder=z + 0.1))
    ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2 + e, cy + h / 2 - e], color=edge, lw=2.0, zorder=z + 0.2)
    ax.plot([cx + w / 2, cx + w / 2], [cy - h / 2 + e, cy + h / 2 - e], color=edge, lw=2.0, zorder=z + 0.2)
    ax.add_patch(Ellipse((cx, cy + h / 2 - e), w, 2 * e, facecolor="#EAF5E6", edgecolor=edge, linewidth=2.0, zorder=z + 0.3))


def face(cx, cy, s, box=True, landmarks=True, z=5):
    """Abstract face (no real person) with a detection box and the 5 landmarks."""
    ax.add_patch(Ellipse((cx, cy), 0.62 * s, 0.78 * s, facecolor="#F1DCC9", edgecolor=C["edge"], linewidth=1.6, zorder=z))
    t = np.linspace(0, np.pi, 30)
    hx = cx + 0.33 * s * np.cos(t)
    hy = cy + 0.12 * s + 0.30 * s * np.sin(t)
    ax.add_patch(Polygon(np.column_stack([hx, hy]), closed=True, facecolor="#3A3A3A", edgecolor="none", zorder=z - 0.1))
    ex, ey = 0.13 * s, 0.07 * s
    for dx in (-ex, ex):
        ax.add_patch(Ellipse((cx + dx, cy + ey), 0.09 * s, 0.05 * s, facecolor=C["edge"], edgecolor="none", zorder=z + 0.2))
    ax.plot([cx, cx - 0.02 * s], [cy + 0.03 * s, cy - 0.09 * s], color=C["gray"], lw=1.4, zorder=z + 0.2)
    t = np.linspace(np.pi * 1.15, np.pi * 1.85, 20)
    ax.plot(cx + 0.11 * s * np.cos(t), cy - 0.17 * s + 0.05 * s * np.sin(t), color=C["red"], lw=1.6, zorder=z + 0.2)
    if box:
        bw, bh = 0.74 * s, 0.9 * s
        ax.add_patch(Rectangle((cx - bw / 2, cy - bh / 2), bw, bh, facecolor="none", edgecolor="#2EB84B", linewidth=2.6, zorder=z + 0.5))
    if landmarks:
        for lx, ly in [(-ex, ey), (ex, ey), (0, -0.07 * s), (-0.1 * s, -0.17 * s), (0.1 * s, -0.17 * s)]:
            ax.add_patch(Circle((cx + lx, cy + ly), 0.032 * s, facecolor=C["cyan"], edgecolor=C["edge"], linewidth=0.9, zorder=z + 0.6))


def chip(cx, cy, s, fill, color="white", size=11.5, w=None, h=0.27):
    w = w or (0.105 * len(s) + 0.2)
    rbox(cx, cy, w, h, fill, edge=C["edge"], lw=1.3, r=0.07, z=7)
    text(cx, cy, s, size=size, color=color, weight="bold", z=9)


def camera_icon(cx, cy, s):
    ax.add_patch(FancyBboxPatch((cx - 0.5 * s, cy - 0.3 * s), 1.0 * s, 0.6 * s, boxstyle="round,pad=0,rounding_size=0.08", facecolor="#4D4D4D", edgecolor=C["edge"], linewidth=1.8, zorder=5))
    ax.add_patch(Rectangle((cx - 0.3 * s, cy + 0.3 * s), 0.3 * s, 0.1 * s, facecolor="#4D4D4D", edgecolor=C["edge"], linewidth=1.5, zorder=5))
    ax.add_patch(Circle((cx, cy), 0.21 * s, facecolor="#D8E6F3", edgecolor=C["edge"], linewidth=1.8, zorder=6))
    ax.add_patch(Circle((cx, cy), 0.11 * s, facecolor=C["blue2"], edgecolor=C["edge"], linewidth=1.2, zorder=7))
    ax.add_patch(Circle((cx + 0.36 * s, cy + 0.19 * s), 0.04 * s, facecolor="#FF5A4F", edgecolor="none", zorder=7))


def stage_label(cx, y, title, sub, color):
    text(cx, y, title, size=17, color=color, weight="bold")
    text(cx, y - 0.34, sub, size=13, color=C["text"])


# ================================================================== panel a
text(0.18, 8.15, "a", size=30, color=C["edge"], weight="bold", ha="left")
text(0.75, 8.15, "Real-time recognition pipeline", size=18, color=C["blue"], weight="bold", ha="left")

YS, BW, BH = 6.2, 1.55, 1.55
XS = [1.05 + 2.17 * i for i in range(7)]
Y_LABEL = YS - BH / 2 - 0.3

# dashed grouping box around the Python pipeline service (decode to match)
rbox((XS[1] + XS[5]) / 2, YS - 0.15, XS[5] - XS[1] + BW + 0.45, BH + 2.05, C["blue_tint"], edge=C["blue"], lw=2.2, r=0.22, ls=(0, (5, 3)), z=1)
text(XS[1] - BW / 2 - 0.05, YS + BH / 2 + 0.55, "Pipeline service", size=13, color=C["blue"], weight="bold", ha="left")

# 1 source
rbox(XS[0], YS, BW, BH, C["gray_fill"])
camera_icon(XS[0], YS + 0.05, 1.25)
stage_label(XS[0], Y_LABEL, "Source", "RTSP webcam or file", C["gray"])

# 2 decode
rbox(XS[1], YS, BW, BH, C["blue_fill"])
logo("gstreamer", XS[1], YS, 1.0)
stage_label(XS[1], Y_LABEL, "Decode", "GStreamer", C["blue"])

# 3 detect
rbox(XS[2], YS, BW, BH, C["blue_fill"])
face(XS[2], YS, 1.35)
stage_label(XS[2], Y_LABEL, "Detect", "SCRFD landmarks", C["blue"])
logo("onnx", XS[2] - BW / 2 + 0.24, YS - BH / 2 + 0.24, 0.32, z=7)

# 4 track
rbox(XS[3], YS, BW, BH, C["blue_fill"])
for dx, dy, lab, col in [(-0.33, 0.12, "#1", "#2EB84B"), (0.33, -0.12, "#2", C["blue2"])]:
    ax.add_patch(Rectangle((XS[3] + dx - 0.3, YS + dy - 0.38), 0.6, 0.76, facecolor="#F6F9FC", edgecolor=col, linewidth=2.4, zorder=5))
    chip(XS[3] + dx - 0.04, YS + dy + 0.38, lab, col, w=0.42, h=0.24, size=10.5)
stage_label(XS[3], Y_LABEL, "Track", "IoU tracker", C["blue"])

# 5 embed
rbox(XS[4], YS, BW, BH, C["blue_fill"])
trapezoid(XS[4] - 0.22, YS, 0.85, 1.05, 0.55, "#9CC2E8")
text(XS[4] - 0.24, YS, "F", size=24, color=C["blue"], weight="bold", style="italic")
embedding_bar(XS[4] + 0.52, YS, 0.2, 1.0)
stage_label(XS[4], Y_LABEL, "Embed", "ArcFace 512-d", C["blue"])
logo("onnx", XS[4] - BW / 2 + 0.24, YS - BH / 2 + 0.24, 0.32, z=7)

# 6 match
rbox(XS[5], YS, BW, BH, C["green_fill"], edge=C["edge"])
cylinder(XS[5] - 0.18, YS - 0.02, 0.85, 1.15, C["green_mid"])
logo("postgresql", XS[5] + 0.5, YS + 0.08, 0.5, z=8)
stage_label(XS[5], Y_LABEL, "Match", "pgvector search", C["green"])

# 7 output (processed video and event)
rbox(XS[6], YS, BW, BH, "#2B2F36", edge=C["edge"])
ax.add_patch(Rectangle((XS[6] - 0.62, YS - 0.5), 1.24, 1.0, facecolor="#4A525E", edgecolor="none", zorder=4))
face(XS[6] - 0.05, YS + 0.07, 0.95, box=False, landmarks=False, z=5)
ax.add_patch(Rectangle((XS[6] - 0.42, YS - 0.33), 0.74, 0.8, facecolor="none", edgecolor="#2EB84B", linewidth=2.4, zorder=7))
chip(XS[6] - 0.02, YS - 0.43, "Alice 0.92", "#2EB84B", size=9.5, w=0.95, h=0.22)
stage_label(XS[6], Y_LABEL, "Output", "video and events", C["edge"])

# block arrows with the data that flows through them
labels = ["frames", "faces", "tracks", "vectors", "matches", "result"]
for i, lab in enumerate(labels):
    if i in (0, 5):  # the two outer labels would sit on the dashed group border
        lab = ""
    x0, x1 = XS[i] + BW / 2 + 0.05, XS[i + 1] - BW / 2 - 0.05
    fat_arrow(x0, x1, YS, color="#CAD6E2" if i not in (4, 5) else "#D6E9D2")
    text((x0 + x1) / 2, YS + BH / 2 + 0.2, lab, size=12, color=C["gray"], style="italic")

# ================================================================== panel b
text(0.18, 3.95, "b", size=30, color=C["edge"], weight="bold", ha="left")
text(0.75, 3.95, "Services and deployment", size=18, color=C["blue"], weight="bold", ha="left")

# docker compose frame
rbox(8.0, 1.75, 15.2, 3.0, "#F3F9FE", edge=C["docker"], lw=2.4, r=0.25, ls=(0, (6, 3)), z=1)
logo("docker", 1.25, 3.0, 0.42, z=7)
text(1.6, 3.0, "Docker Compose", size=14, color=C["docker"], weight="bold", ha="left")

SY, SW, SH = 1.55, 2.7, 1.55
SX = [2.35, 6.1, 9.85, 13.6]
specs = [
    ("Pipeline", "SCRFD and ArcFace", C["blue_fill"], C["blue"], ["gstreamer", "onnx"]),
    ("PostgreSQL", "pgvector", C["green_fill"], C["green"], ["postgresql"]),
    ("FastAPI", "REST and WebSocket", C["teal_fill"], C["teal"], ["fastapi"]),
    ("Next.js", "Live  Faces  History", C["violet_fill"], C["violet"], ["nextdotjs"]),
]
for (title, sub, fill, col, logos), cx in zip(specs, SX):
    rbox(cx, SY, SW, SH, fill)
    n = len(logos)
    for k, name in enumerate(logos):
        logo(name, cx + (k - (n - 1) / 2) * 0.62, SY + 0.32, 0.55)
    text(cx, SY - 0.28, title, size=16, color=col, weight="bold")
    text(cx, SY - 0.58, sub, size=12.5, color=C["text"])

# data flows between services
line_arrow((SX[0] + SW / 2 + 0.04, SY), (SX[1] - SW / 2 - 0.04, SY), both=True)
text((SX[0] + SX[1]) / 2, SY + 0.22, "embeddings", size=11.5, color=C["gray"], style="italic")
line_arrow((SX[1] + SW / 2 + 0.04, SY), (SX[2] - SW / 2 - 0.04, SY), both=True)
text((SX[1] + SX[2]) / 2, SY + 0.22, "SQL", size=11.5, color=C["gray"], style="italic")
line_arrow((SX[2] + SW / 2 + 0.04, SY), (SX[3] - SW / 2 - 0.04, SY), both=True)
text((SX[2] + SX[3]) / 2, SY + 0.22, "REST", size=11.5, color=C["gray"], style="italic")
# events and video go from the pipeline to the backend over the top of the database box
line_arrow((SX[0] + 0.5, SY + SH / 2 + 0.04), (SX[2] - 0.5, SY + SH / 2 + 0.04), color=C["red"], rad=-0.22, ls=(0, (4, 3)))
text((SX[0] + SX[2]) / 2 + 0.05, SY + SH / 2 + 0.36, "events and MJPEG video", size=12, color=C["red"], style="italic")

fig.savefig(f"{OUT}/architecture.pdf", facecolor="white")
fig.savefig(f"{OUT}/architecture.png", dpi=300, facecolor="white")
print("figure written")
