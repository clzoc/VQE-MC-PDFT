#!/usr/bin/env python3

"""Plot the Cr2 active-space comparison figure for the reviewer package."""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from plot_common import configure_plot_style, resolve_output_path, show_figure

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_PDF = resolve_output_path(__file__, "fig4_cr2_active_space.pdf")
ACTIVE_SPACE_CSV = DATA_DIR / "cr2_active_space_figure.csv"
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

    figure_df = pd.read_csv(ACTIVE_SPACE_CSV)
    reference_df = pd.read_csv(REFERENCE_CSV)

    x = figure_df["bond_A"]
    x_ref = reference_df["bond_A"]
    y_ref = reference_df["binding_energy_eV"]

    fig = plt.figure(figsize=(9, 9), dpi=90)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(1.42, 3.43)
    ax.set_ylim(-1.7, 0.8)
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
        ("vqe_casscf_12e_12o_eV", "VQE-CASSCF(12e, 12o)", rgb_to_hex(COLOR_PALETTE[4]), "P", None),
        ("vqe_casscf_12e_22o_eV", "VQE-CASSCF(12e, 22o)", "#6A0DAD", "h", None),
        ("vqe_casscf_12e_28o_eV", "VQE-CASSCF(12e, 28o)", rgb_to_hex(COLOR_PALETTE[1]), "o", None),
        ("vqe_mcpdft_12e_12o_eV", "VQE-MC-PDFT(12e, 12o)", rgb_to_hex(COLOR_PALETTE[0]), "s", None),
        ("vqe_mcpdft_12e_22o_eV", "VQE-MC-PDFT(12e, 22o)", "#EAB4F8", ">", 1.5),
        ("vqe_mcpdft_12e_28o_eV", "VQE-MC-PDFT(12e, 28o)", "#2E8B57", "d", 1.5),
    ]

    for column, label, color, marker, linewidths in series_specs:
        spline = CubicSpline(x, figure_df[column])
        ax.plot(xs, spline(xs), linewidth=3, linestyle="-", c=color, label=label)
        scatter_kwargs = {"marker": marker, "s": 180, "c": color}
        if linewidths is None:
            scatter_kwargs["linewidth"] = 0
            ax.scatter(x, figure_df[column], **scatter_kwargs)
        else:
            scatter_kwargs["linewidths"] = linewidths
            ax.scatter(x, spline(x), **scatter_kwargs)

    ax.plot(
        xs,
        CubicSpline(x_ref, y_ref)(xs),
        linewidth=3,
        c=rgb_to_hex(COLOR_PALETTE[5]),
        label="Ref. Larsson et al.",
    )

    handles, labels = ax.get_legend_handles_labels()
    marker_map = {
        "VQE-CASSCF(12e, 12o)": "P",
        "VQE-CASSCF(12e, 22o)": "h",
        "VQE-CASSCF(12e, 28o)": "o",
        "VQE-MC-PDFT(12e, 12o)": "s",
        "VQE-MC-PDFT(12e, 22o)": ">",
        "VQE-MC-PDFT(12e, 28o)": "d",
        "Ref. Larsson et al.": "",
    }
    empty_handle = mlines.Line2D([], [], color="none", label="")
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
    legend_handles = legend_handles[:3] + [empty_handle] + legend_handles[3:]

    fig.axes[0].legend(
        handles=legend_handles,
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.102),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
        prop={"size": 17},
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 1.00])
    fig.savefig(OUTPUT_PDF, dpi=fig.dpi, bbox_inches="tight")
    show_figure()


if __name__ == "__main__":
    main()
