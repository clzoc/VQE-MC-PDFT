#!/usr/bin/env python3

"""Plot the Cr2 basis-set comparison figure for the manuscript figure set."""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.interpolate import CubicSpline

from plot_common import configure_plot_style, resolve_output_path, show_figure

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_PDF = resolve_output_path(__file__, "fig5_cr2_basis_set.pdf")
BASIS_SET_CSV = DATA_DIR / "cr2_basis_set_figure.csv"
REFERENCE_CSV = DATA_DIR / "cr2_larsson_reference_curve.csv"

COLOR_PALETTE = [
    (57, 81, 162),
    (114, 170, 207),
    (202, 232, 242),
    (254, 251, 186),
    (253, 185, 107),
    (236, 93, 59),
    (168, 3, 38),
]


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def main() -> None:
    configure_plot_style()

    figure_df = pd.read_csv(BASIS_SET_CSV)
    reference_df = pd.read_csv(REFERENCE_CSV)

    x = figure_df["bond_A"]
    x_ref = reference_df["bond_A"]
    y_ref = reference_df["binding_energy_eV"]

    fig = plt.figure(figsize=(9, 9), dpi=90)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(1.42, 3.43)
    ax.set_ylim(-1.6, -0.2)
    plt.xticks(size=28, weight="regular")
    ax.set_xlabel("Bond distance (Å)", fontsize=28, fontweight="regular")
    ax.axes.xaxis.set_tick_params(direction="in", which="both", length=5, width=2, right=True, pad=15)
    ax.axes.yaxis.set_tick_params(direction="in", which="both", length=6, width=2, right=True, pad=15)
    ax.set_ylabel("Binding Energy (eV)", fontsize=28, fontweight="regular", labelpad=15)
    plt.yticks(fontsize=28, fontweight="regular")
    for spine in ["bottom", "top", "left", "right"]:
        ax.spines[spine].set_linewidth(2)

    xs = np.linspace(1.42, 3.43, 500)

    series_specs = [
        ("vqe_mcpdft_tz_eV", "VQE-MC-PDFT(12e, 22o) TZ", rgb_to_hex(COLOR_PALETTE[0]), "s", None),
        ("vqe_mcpdft_qz_eV", "VQE-MC-PDFT(12e, 22o) QZ", "#EAB4F8", ">", 1.5),
        ("vqe_mcpdft_5z_eV", "VQE-MC-PDFT(12e, 22o) 5Z", "#2E8B57", "d", 1.5),
    ]

    splines = {column: CubicSpline(x, figure_df[column]) for column, *_ in series_specs}
    for column, label, color, marker, linewidths in series_specs:
        ax.plot(xs, splines[column](xs), linewidth=3, linestyle="-", c=color, label=label)
        scatter_kwargs = {"marker": marker, "s": 180, "c": color}
        if linewidths is None:
            scatter_kwargs["linewidth"] = 0
            ax.scatter(x, figure_df[column], **scatter_kwargs)
        else:
            scatter_kwargs["linewidths"] = linewidths
            ax.scatter(x, splines[column](x), **scatter_kwargs)

    ax.plot(
        xs,
        CubicSpline(x_ref, y_ref)(xs),
        linewidth=3,
        c=rgb_to_hex(COLOR_PALETTE[5]),
        label="Ref. Larsson et al.",
    )

    axins = inset_axes(
        ax,
        width="40%",
        height="45%",
        loc="lower left",
        bbox_to_anchor=(0.15, 0.50, 1, 1),
        bbox_transform=ax.transAxes,
    )
    for spine in axins.spines.values():
        spine.set_linewidth(1.5)
    axins.axes.xaxis.set_tick_params(direction="in", which="both", length=3.5, width=1.5, right=True)
    axins.axes.yaxis.set_tick_params(direction="in", which="both", length=4, width=1.5, right=True, pad=6)
    plt.yticks(fontsize=18, fontweight="regular")

    for column, label, color, marker, linewidths in series_specs:
        axins.plot(xs, splines[column](xs), linewidth=3, linestyle="-", c=color, label=label)
        scatter_kwargs = {"marker": marker, "s": 180, "c": color}
        if linewidths is None:
            scatter_kwargs["linewidth"] = 0
            axins.scatter(x, figure_df[column], **scatter_kwargs)
        else:
            scatter_kwargs["linewidths"] = linewidths
            axins.scatter(x, splines[column](x), **scatter_kwargs)

    plt.xticks([1.6, 1.7, 1.8], fontsize=18, fontweight="regular")
    plt.yticks([-1.4, -1.3, -1.2], fontsize=18, fontweight="regular")
    axins.set_xlim(1.58, 1.82)
    axins.set_ylim(-1.43, -1.20)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="k", lw=1)

    handles, labels = ax.get_legend_handles_labels()
    marker_map = {
        "VQE-MC-PDFT(12e, 22o) TZ": "s",
        "VQE-MC-PDFT(12e, 22o) QZ": ">",
        "VQE-MC-PDFT(12e, 22o) 5Z": "d",
        "Ref. Larsson et al.": "",
    }
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=handle.get_color(),
            marker=marker_map.get(label),
            linestyle=handle.get_linestyle(),
            markersize=10,
            label=label,
        )
        for handle, label in zip(handles, labels)
    ]

    fig.axes[0].legend(
        handles=legend_handles,
        bbox_to_anchor=(0.40, 1.01, 0.60, 0.102),
        loc="lower left",
        ncol=1,
        mode="expand",
        borderaxespad=0.0,
        prop={"size": 18},
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 1.00])
    fig.savefig(OUTPUT_PDF, dpi=fig.dpi, bbox_inches="tight")
    show_figure()


if __name__ == "__main__":
    main()
