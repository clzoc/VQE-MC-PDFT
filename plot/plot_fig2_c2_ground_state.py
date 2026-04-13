"""Plot the C2 ground-state comparison figure for the reviewer package."""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.interpolate import CubicSpline

from plot_common import configure_plot_style, resolve_output_path, show_figure

HARTREE_TO_EV = 27.211342176
DATA_DIR = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"
OUTPUT_PDF = resolve_output_path(__file__, "fig2_c2_ground_state.pdf")

METHOD_DATA_CSV = DATA_DIR / "c2_ground_state_pec_exact.csv"
SHCI_REFERENCE_CSV = DATA_DIR / "c2_ground_state_shci_reference.csv"

HF_ATOMIC = -37.6867080514
DFT_ATOMIC = -37.8411471723
CASSCF_ATOMIC = -37.7058453152
MCPDFT_ATOMIC = -37.7930150040
SHCI_ATOMIC = -37.844746

COLOR_PALETTE = [
    (57, 81, 162),
    (114, 170, 207),
    (202, 232, 242),
    (254, 251, 186),
    (253, 185, 107),
    (236, 93, 59),
    (168, 3, 38),
]

REFERENCE_LABEL = "SHCI"


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def dense_curve_samples(x_grid: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_dense = np.linspace(0.77, 3.03, 500)
    y_dense = CubicSpline(x_grid, y_values)(x_dense) * HARTREE_TO_EV
    return x_dense, y_dense


def sample_dense_points(x_dense: np.ndarray, y_dense: np.ndarray, x_points: np.ndarray) -> list[float]:
    samples: list[float] = []
    index = 0
    current_best = np.inf
    for i, x_val in enumerate(x_dense):
        if index >= x_points.size:
            break
        distance = abs(x_val - x_points[index])
        if distance < current_best:
            current_best = distance
        else:
            current_best = np.inf
            index += 1
            samples.append(float(y_dense[i]))
    return samples


def align_ylabels(fig: plt.Figure) -> None:
    x_coord = -0.12
    fig.axes[0].yaxis.set_label_coords(x_coord, 0.5)
    fig.axes[-1].yaxis.set_label_coords(x_coord, 0.5)


def main() -> None:
    configure_plot_style()

    method_df = pd.read_csv(METHOD_DATA_CSV)
    shci_df = pd.read_csv(SHCI_REFERENCE_CSV)

    x = method_df["bond_A"].to_numpy()
    hf_relative = method_df["HF"].to_numpy() - 2 * HF_ATOMIC
    dft_relative = method_df["DFT_PBE"].to_numpy() - 2 * DFT_ATOMIC
    casscf_relative = method_df["CASSCF"].to_numpy() - 2 * CASSCF_ATOMIC
    sa_casscf_relative = method_df["VQE_SA_CASSCF"].to_numpy() - 2 * CASSCF_ATOMIC
    mcpdft_relative = method_df["VQE_MC_PDFT"].to_numpy() - 2 * MCPDFT_ATOMIC

    shci_bond = shci_df["bond_A"].to_numpy()
    shci_relative = shci_df["energy_hartree"].to_numpy() - 2 * SHCI_ATOMIC

    fig = plt.figure(figsize=(9, 14), dpi=90)
    ax_top = fig.add_subplot(2, 1, 1)

    ax_top.set_xlim(0.70, 2.10)
    ax_top.set_ylim(-8, 28)
    ax_top.axes.xaxis.set_ticklabels([])
    ax_top.axes.xaxis.set_tick_params(direction="in", which="both", length=5, width=2, right=True)
    ax_top.axes.yaxis.set_tick_params(direction="in", which="both", length=6, width=2, right=True, pad=15)
    ax_top.set_ylabel("Binding Energy (eV)", fontsize=28, fontweight="regular", labelpad=20)
    plt.yticks(fontsize=28, fontweight="regular")
    ax_top.plot([0.70, 3.10], [0, 0], linewidth=2, c="black", linestyle="--", alpha=0.3)
    for spine in ax_top.spines.values():
        spine.set_linewidth(2)

    top_series = [
        ("HF", hf_relative, "#2E8B57", ">", 20 * 12),
        ("DFT/PBE", dft_relative, "#EAB4F8", "^", 20 * 12),
        ("CASSCF", casscf_relative, rgb_to_hex(COLOR_PALETTE[4]), "P", 20 * 12),
        ("VQE-SA-CASSCF", sa_casscf_relative, rgb_to_hex(COLOR_PALETTE[1]), "o", 20 * 11),
        ("VQE-MC-PDFT", mcpdft_relative, rgb_to_hex(COLOR_PALETTE[0]), "s", 20 * 9),
    ]

    dense_x = np.linspace(0.77, 3.03, 500)
    dense_curves: list[np.ndarray] = []
    sampled_curves: list[list[float]] = []

    for label, relative_energy, color, marker, size in top_series:
        dense_x, dense_y = dense_curve_samples(x, relative_energy)
        dense_curves.append(dense_y)
        ax_top.plot(dense_x, dense_y, linewidth=3, c=color, label=label)
        ax_top.scatter(x, relative_energy * HARTREE_TO_EV, c=color, marker=marker, s=size, linewidth=0)
        sampled_curves.append(sample_dense_points(dense_x, dense_y, x))

    dense_x, shci_dense = dense_curve_samples(shci_bond, shci_relative)
    dense_curves.append(shci_dense)
    ax_top.plot(dense_x, shci_dense, linewidth=3, c=rgb_to_hex(COLOR_PALETTE[5]), label=f"Ref. {REFERENCE_LABEL}")
    ax_top.scatter(
        shci_bond,
        shci_relative * HARTREE_TO_EV,
        c=rgb_to_hex(COLOR_PALETTE[5]),
        marker="D",
        s=200,
        linewidth=0,
    )
    sampled_curves.append(sample_dense_points(dense_x, shci_dense, x))

    handles, labels = ax_top.get_legend_handles_labels()
    legend_markers = [">", "^", "P", "o", "s", "D"]
    legend_sizes = [12, 12, 12, 11, 11, 9]
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=handle.get_color(),
            marker=legend_markers[i],
            linestyle=handle.get_linestyle(),
            markersize=legend_sizes[i],
            label=labels[i],
        )
        for i, handle in enumerate(handles)
    ]
    ax_top.legend(handles=legend_handles, loc="upper right", ncol=2, prop={"size": 18})

    ax_inset = inset_axes(
        ax_top,
        width="65%",
        height="37%",
        loc="lower left",
        bbox_to_anchor=(0.262, 0.375, 1, 1),
        bbox_transform=ax_top.transAxes,
    )
    for spine in ax_inset.spines.values():
        spine.set_linewidth(1.5)
    ax_inset.axes.xaxis.set_tick_params(direction="in", which="both", length=3.5, width=1.5, right=True)
    ax_inset.axes.yaxis.set_tick_params(direction="in", which="both", length=4, width=1.5, right=True, pad=6)

    inset_series = [
        (dense_curves[1], dft_relative, "#EAB4F8", "^", 20 * 12),
        (dense_curves[2], casscf_relative, rgb_to_hex(COLOR_PALETTE[4]), "P", 20 * 12),
        (dense_curves[3], sa_casscf_relative, rgb_to_hex(COLOR_PALETTE[1]), "o", 200),
        (dense_curves[4], mcpdft_relative, rgb_to_hex(COLOR_PALETTE[0]), "s", 20 * 11),
        (dense_curves[5], shci_relative, rgb_to_hex(COLOR_PALETTE[5]), "D", 20 * 9),
    ]
    for dense_y, relative_energy, color, marker, size in inset_series:
        ax_inset.plot(dense_x, dense_y, linewidth=3, c=color)
        if marker == "D":
            ax_inset.scatter(shci_bond, relative_energy * HARTREE_TO_EV, c=color, marker=marker, s=size, linewidth=0)
        else:
            ax_inset.scatter(x, relative_energy * HARTREE_TO_EV, c=color, marker=marker, s=size, linewidth=0)

    plt.xticks([1.2, 1.4, 1.6, 1.8], fontsize=18, fontweight="regular")
    plt.yticks([-7, -6, -5, -4, -3], fontsize=18, fontweight="regular")
    ax_inset.set_xlim(1.10, 1.85)
    ax_inset.set_ylim(-6.5, -2.8)
    mark_inset(ax_top, ax_inset, loc1=3, loc2=1, fc="none", ec="k", lw=1)

    ax_top.text(
        -0.12,
        1.00,
        "(a)",
        transform=ax_top.transAxes,
        fontsize=28,
        fontweight="regular",
        va="top",
        ha="right",
        family="sans-serif",
    )

    ax_bottom = fig.add_subplot(2, 1, 2)
    ax_bottom.set_xlim(0.70, 2.10)
    plt.xticks(size=28, weight="regular")
    ax_bottom.set_ylabel("Deviation $D_e$ (eV)", fontsize=28, fontweight="regular", labelpad=20)
    ax_bottom.plot([0.70, 3.10], [0, 0], linewidth=3, c="black", linestyle="--", alpha=0.3)
    ax_bottom.set_xlabel("Bond distance (Å)", fontsize=28, fontweight="regular")
    ax_bottom.axes.xaxis.set_tick_params(direction="in", which="both", length=5, width=2, top=True, pad=15)
    ax_bottom.axes.yaxis.set_tick_params(direction="in", which="both", width=2, right=True, pad=15)
    ax_bottom.axes.yaxis.set_tick_params(which="major", length=6)
    ax_bottom.axes.yaxis.set_tick_params(which="minor", length=3)
    plt.yticks(size=28, weight="regular")
    for spine in ax_bottom.spines.values():
        spine.set_linewidth(2)

    comparison_series = [
        ("HF", hf_relative, "#2E8B57", ">", 12),
        ("DFT/PBE", dft_relative, "#EAB4F8", "^", 12),
        ("CASSCF", casscf_relative, rgb_to_hex(COLOR_PALETTE[4]), "P", 12),
        ("VQE-SA-CASSCF", sa_casscf_relative, rgb_to_hex(COLOR_PALETTE[1]), "o", 11),
        ("VQE-MC-PDFT", mcpdft_relative, rgb_to_hex(COLOR_PALETTE[0]), "s", 11),
    ]
    reference_points = np.array(sampled_curves[-1])
    for label, relative_energy, color, marker, marker_size in comparison_series:
        deviation = relative_energy * HARTREE_TO_EV - reference_points
        ax_bottom.plot(
            x,
            deviation,
            c=color,
            marker=marker,
            markersize=marker_size,
            linewidth=3,
            linestyle="--",
            label=label,
        )

    bottom_handles, bottom_labels = ax_bottom.get_legend_handles_labels()
    bottom_legend_sizes = [12, 12, 12, 11, 11, 9]
    bottom_legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=handle.get_color(),
            marker=legend_markers[i],
            linestyle=handle.get_linestyle(),
            markersize=bottom_legend_sizes[i],
            label=bottom_labels[i],
        )
        for i, handle in enumerate(bottom_handles)
    ]
    ax_bottom.legend(handles=bottom_legend_handles, loc="upper right", ncol=2, prop={"size": 18})
    ax_bottom.text(
        -0.12,
        1.00,
        "(b)",
        transform=ax_bottom.transAxes,
        fontsize=28,
        fontweight="regular",
        va="top",
        ha="right",
        family="sans-serif",
    )

    align_ylabels(fig)

    def on_resize(event) -> None:
        fig.tight_layout()
        fig.canvas.draw()

    plt.tight_layout()
    fig.canvas.mpl_connect("resize_event", on_resize)
    fig.savefig(OUTPUT_PDF, dpi=fig.dpi, bbox_inches="tight")
    show_figure()


if __name__ == "__main__":
    main()
