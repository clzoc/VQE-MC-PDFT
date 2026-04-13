from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from plot_common import configure_plot_style, resolve_output_path, show_figure

DATA_JSON = Path(__file__).resolve().parents[1] / "data" / "tianji_s2_calibration.json"
OUTPUT_PDF = resolve_output_path(__file__, "figS1_tianji_s2_hardware_fidelity.pdf")

SQUARE_POSITIONS = {
    0: (1, 0),
    1: (1, 1),
    2: (0, 1),
    3: (2, 1),
    4: (1, 2),
    5: (2, 3),
    6: (2, 2),
    7: (3, 2),
    8: (3, 1),
    9: (4, 1),
    10: (3, 0),
    11: (2, -1),
    12: (2, 0),
}


def load_calibration_data() -> tuple[dict[int, tuple[float, str, str]], dict[tuple[int, int], float]]:
    calibration = json.loads(DATA_JSON.read_text())
    gate_error = calibration["single_qubit"]["gate_error"]
    cz_error = calibration["two_qubit"]["CZ_error"]

    square_data: dict[int, tuple[float, str, str]] = {}
    for qubit in range(calibration["n_qubits"]):
        fidelity = 1.0 - float(gate_error[f"Q{qubit}"])
        square_data[qubit] = (fidelity, str(qubit), rf"${fidelity * 100:.2f}\%$")

    connection_map: dict[tuple[int, int], float] = {}
    for edge, error in cz_error.items():
        left, right = edge.replace("Q", "").split("_")
        q0 = int(left)
        q1 = int(right)
        connection_map[(q0, q1)] = 1.0 - float(error)

    return square_data, connection_map


def main() -> None:
    configure_plot_style()
    square_data, connection_map = load_calibration_data()

    fig = plt.figure(figsize=(16, 6.5), dpi=90)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 0.10, 1.0, 0.10, 0.05], wspace=0.02)
    ax_left = fig.add_subplot(gs[0, 0])
    cax_left = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])
    cax_right = fig.add_subplot(gs[0, 3])

    cmap_left = plt.cm.Blues
    norm_left = plt.Normalize(vmin=0.9980, vmax=1.0)

    for qubit, (value, label, pct_str) in square_data.items():
        x, y = SQUARE_POSITIONS[qubit]
        square = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=cmap_left(norm_left(value)), edgecolor="white", linewidth=2)
        ax_left.add_patch(square)
        ax_left.text(x, y + 0.15, label, ha="center", va="center", color="black", fontsize=18)
        ax_left.text(x, y - 0.15, pct_str, ha="center", va="center", color="black", fontsize=14)

    ax_left.text(-0.12, 3.50, "(a)", fontsize=28, fontweight="regular", va="top", ha="right", family="sans-serif")
    xlim_left = (-0.5, 4.93)
    ylim_left = (-1.7, 3.7)
    ax_left.set_xlim(*xlim_left)
    ax_left.set_ylim(*ylim_left)
    ax_left.set_aspect("equal", adjustable="box")
    ax_left.axis("off")
    cax_left.set_axis_off()

    bar_ax = cax_left.inset_axes([0.62, 0.06, 0.33, 0.88])
    sm_left = plt.cm.ScalarMappable(cmap=cmap_left, norm=norm_left)
    sm_left.set_array([])
    cbar_left = fig.colorbar(sm_left, cax=bar_ax, ticks=[0.9980, 0.9985, 0.9990, 0.9995, 1.0000])
    cbar_left.ax.yaxis.set_ticks_position("left")
    cbar_left.ax.tick_params(labelsize=18, pad=2)
    cbar_left.set_label("SQ Fidelity", fontsize=28, labelpad=10)
    cbar_left.ax.yaxis.set_label_position("right")

    layout_scale = 1.1
    cmap_right = plt.cm.Greens
    norm_right = plt.Normalize(vmin=0.985, vmax=1.0)

    for (q0, q1), value in connection_map.items():
        x0, y0 = SQUARE_POSITIONS[q0]
        x1, y1 = SQUARE_POSITIONS[q1]
        x = 0.5 * (x0 + x1) * layout_scale
        y = 0.5 * (y0 + y1) * layout_scale
        orientation = np.deg2rad(30)
        if y0 != y1:
            orientation = np.deg2rad(120)
        hexagon = patches.RegularPolygon(
            (x, y),
            numVertices=6,
            radius=1.0 / np.sqrt(15) * 1.20,
            orientation=orientation,
            facecolor=cmap_right(norm_right(value)),
            edgecolor="white",
            linewidth=1.5,
        )
        ax_right.add_patch(hexagon)
        ax_right.text(x, y, f"{value:.3f}", ha="center", va="center", color="black", fontsize=14)

    for qubit, (value, label, _) in square_data.items():
        x, y = SQUARE_POSITIONS[qubit]
        x *= layout_scale
        y *= layout_scale
        circle = patches.Circle((x, y), radius=0.25, facecolor=cmap_left(norm_left(value)), edgecolor="none")
        ax_right.add_patch(circle)
        ax_right.text(x, y, label, ha="center", va="center", color="black", fontsize=18)

    target_ratio = (xlim_left[1] - xlim_left[0]) / (ylim_left[1] - ylim_left[0])
    xlim_right = (-0.8, 4.9)
    ax_right.set_xlim(*xlim_right)
    y_vals = [position[1] * layout_scale for position in SQUARE_POSITIONS.values()]
    y_center = (max(y_vals) + min(y_vals)) / 2
    y_range = (xlim_right[1] - xlim_right[0]) / target_ratio
    ylim_right = (y_center - y_range / 2, y_center + y_range / 2)
    ax_right.set_ylim(*ylim_right)
    ax_right.set_aspect("equal", adjustable="box")
    ax_right.axis("off")
    ax_right.text(-0.12, 3.85, "(b)", fontsize=28, fontweight="regular", va="top", ha="right", family="sans-serif")

    cax_right.set_axis_off()
    bar_ax_right = cax_right.inset_axes([0.05, 0.06, 0.33, 0.88])
    sm_right = plt.cm.ScalarMappable(cmap=cmap_right, norm=norm_right)
    sm_right.set_array([])
    cbar_right = fig.colorbar(sm_right, cax=bar_ax_right, ticks=[0.985, 0.990, 0.995, 1.000])
    cbar_right.ax.yaxis.set_ticks_position("left")
    cbar_right.ax.tick_params(labelsize=18, pad=2)
    cbar_right.set_label("CZ Fidelity", fontsize=28, labelpad=10)
    cbar_right.ax.yaxis.set_label_position("right")

    plt.tight_layout()
    fig.savefig(OUTPUT_PDF, dpi=fig.dpi, bbox_inches="tight")
    show_figure()


if __name__ == "__main__":
    main()
