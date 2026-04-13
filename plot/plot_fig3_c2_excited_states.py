"""Plot the C2 excited-state comparison figure for the reviewer package."""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pathlib import Path
from scipy.interpolate import CubicSpline

from plot_common import configure_plot_style, resolve_output_path, show_figure

HARTREE_TO_EV = 27.211342176
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_PDF = resolve_output_path(__file__, "fig3_c2_excited_states.pdf")

PES_DATA_CSV = DATA_DIR / "c2_excited_state_pec_exact.csv"
SHCI_PES_DATA_CSV = DATA_DIR / "c2_excited_state_shci_pec.csv"
BENCHMARK_CSV = DATA_DIR / "c2_excitation_energies.csv"

MCPDFT_ATOMIC = -37.8456944620

STATE_ORDER = [
    "X1Sg+",
    "A1Pu",
    "B1Dg",
    "Bp1Sg+",
    "C1Pg",
    "a3Pu",
    "b3Sg-",
    "c3Su+",
    "d3Pg",
]
STATE_LABELS = {
    "X1Sg+": r"$X_1\Sigma_g^+$",
    "A1Pu": r"$A_1\Pi_u$",
    "B1Dg": r"$B_1\Delta_g$",
    "Bp1Sg+": r"$B'_1\Sigma_g^+$",
    "C1Pg": r"$C_1\Pi_g$",
    "a3Pu": r"$a_3\Pi_u$",
    "b3Sg-": r"$b_3\Sigma_g^-$",
    "c3Su+": r"$c_3\Sigma_u^+$",
    "d3Pg": r"$d_3\Pi_g$",
}
COLOR_PALETTE = [
    (128, 116, 200),
    (120, 149, 193),
    (168, 203, 223),
    (214, 239, 244),
    (98, 156, 53),
    (153, 34, 36),
    (227, 98, 93),
    (239, 139, 103),
    (245, 235, 174),
]
MARKERS = ["o", "s", "p", "P", "d", "^", "h", "X", ">"]

LEVEL_STATE_ORDER = [
    "X1Sg+",
    "A1Pu",
    "B1Dg",
    "c3Su+",
    "a3Pu",
    "b3Sg-",
    "d3Pg",
    "C1Pg",
    "Bp1Sg+",
]
LEVEL_LABELS = {
    "X1Sg+": r"$X^1\Sigma_g^+$",
    "A1Pu": r"$A^1\Pi_u$",
    "B1Dg": r"$B^1\Delta_g$",
    "c3Su+": r"$c^3\Sigma_u^+$",
    "a3Pu": r"$a^3\Pi_u$",
    "b3Sg-": r"$b^3\Sigma_g^-$",
    "d3Pg": r"$d^3\Pi_g$",
    "C1Pg": r"$C^1\Pi_g$",
    "Bp1Sg+": r"$B'^1\Sigma_g^+$",
}
LEVEL_POSITIONS = {
    "X1Sg+": 0.5,
    "A1Pu": 0.5,
    "B1Dg": 0.5,
    "a3Pu": 2.0,
    "b3Sg-": 2.0,
    "c3Su+": 3.0,
    "d3Pg": 3.0,
    "C1Pg": 4.5,
    "Bp1Sg+": 4.5,
}
STATE_NAME_MAP = {
    "X1Sg+": "X_1Sigma_g+",
    "A1Pu": "A_1Pi_u",
    "B1Dg": "B_1Delta_g",
    "Bp1Sg+": "B'_1Sigma_g+",
    "C1Pg": "C_1Pi_g",
    "a3Pu": "a_3Pi_u",
    "b3Sg-": "b_3Sigma_g-",
    "c3Su+": "c_3Sigma_u+",
    "d3Pg": "d_3Pi_g",
}


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def align_ylabels(fig: plt.Figure) -> None:
    x_coord = -0.12
    fig.axes[0].yaxis.set_label_coords(x_coord, 0.5)
    fig.axes[-1].yaxis.set_label_coords(x_coord, 0.5)


def load_energy_table(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    return {column: df[column].to_numpy() for column in df.columns}


def benchmark_lookup(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    lookup: dict[str, dict[str, float | None]] = {}
    for _, row in df.iterrows():
        lookup[row["state"]] = {
            "shci": float(row["SHCI_eV"]),
            "exp": float(row["exp_eV"]),
            "tbe": None if pd.isna(row["TBE_eV"]) else float(row["TBE_eV"]),
        }
    return lookup


def main() -> None:
    configure_plot_style()

    pes_data = load_energy_table(PES_DATA_CSV)
    shci_pes_data = load_energy_table(SHCI_PES_DATA_CSV)
    benchmark_data = benchmark_lookup(pd.read_csv(BENCHMARK_CSV))

    x = pes_data["bond_A"]
    x_shci = shci_pes_data["bond_A"]

    fig = plt.figure(figsize=(9, 14), dpi=90)
    ax_top = fig.add_subplot(2, 1, 1)

    ax_top.set_xlim(0.70, 2.5)
    ax_top.set_ylim(-8, 28)
    plt.xticks(fontsize=28, fontweight="regular")
    ax_top.set_xlabel("Bond distance (Å)", fontsize=28, fontweight="regular")
    ax_top.axes.xaxis.set_tick_params(direction="in", which="both", length=5, width=2, right=True, pad=15)
    ax_top.axes.yaxis.set_tick_params(direction="in", which="both", length=6, width=2, right=True, pad=15)
    ax_top.set_ylabel("Binding Energy (eV)", fontsize=28, fontweight="regular", labelpad=15)
    plt.yticks(fontsize=28, fontweight="regular")
    ax_top.plot([0.70, 3.10], [0, 0], linewidth=2, c="black", linestyle="--", alpha=0.3)
    for spine in ax_top.spines.values():
        spine.set_linewidth(2)

    for i, state in enumerate(STATE_ORDER):
        y = (pes_data[state] - 2 * MCPDFT_ATOMIC) * HARTREE_TO_EV
        ax_top.scatter(x, y, c=rgb_to_hex(COLOR_PALETTE[i]), marker=MARKERS[i], s=200, linewidth=0)

    for i, state in enumerate(STATE_ORDER):
        if state == "C1Pg":
            y = (shci_pes_data[state][:4] - 75 - 0.108 - 2 * MCPDFT_ATOMIC) * HARTREE_TO_EV
            temp_x = x_shci[:4]
            dense_x = np.linspace(0.77, 1.32, 500)
            dense_y = CubicSpline(temp_x, y)(dense_x)
        else:
            y = (shci_pes_data[state] - 75 - 0.108 - 2 * MCPDFT_ATOMIC) * HARTREE_TO_EV
            dense_x = np.linspace(0.77, 3.03, 500)
            dense_y = CubicSpline(x_shci, y)(dense_x)
        ax_top.plot(dense_x, dense_y, lw=3, linestyle="--", c=rgb_to_hex(COLOR_PALETTE[i]), label=STATE_LABELS[state])

    handles, labels = ax_top.get_legend_handles_labels()
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=handle.get_color(),
            marker=MARKERS[i],
            linestyle=handle.get_linestyle(),
            markersize=10,
            label=labels[i],
        )
        for i, handle in enumerate(handles)
    ]
    ax_top.legend(handles=legend_handles, loc="upper right", ncol=3, prop={"size": 18})

    ax_inset = inset_axes(
        ax_top,
        width="65%",
        height="45%",
        loc="lower left",
        bbox_to_anchor=(0.275, 0.28, 1, 1),
        bbox_transform=ax_top.transAxes,
    )
    for spine in ax_inset.spines.values():
        spine.set_linewidth(1.5)
    plt.xticks(fontsize=18, fontweight="regular")
    ax_inset.axes.xaxis.set_tick_params(direction="in", which="both", length=3.5, width=1.5, right=True)
    ax_inset.axes.yaxis.set_tick_params(direction="in", which="both", length=4, width=1.5, right=True, pad=6)
    plt.yticks(fontsize=18, fontweight="regular")

    for i, state in enumerate(STATE_ORDER):
        y = (pes_data[state] - 2 * MCPDFT_ATOMIC) * HARTREE_TO_EV
        ax_inset.scatter(x, y, c=rgb_to_hex(COLOR_PALETTE[i]), marker=MARKERS[i], s=200, linewidth=0)

    for i, state in enumerate(STATE_ORDER):
        if state == "C1Pg":
            y = (shci_pes_data[state][:4] - 75 - 0.108 - 2 * MCPDFT_ATOMIC) * HARTREE_TO_EV
            temp_x = x_shci[:4]
            dense_x = np.linspace(0.77, 1.32, 500)
            dense_y = CubicSpline(temp_x, y)(dense_x)
        else:
            y = (shci_pes_data[state] - 75 - 0.108 - 2 * MCPDFT_ATOMIC) * HARTREE_TO_EV
            dense_x = np.linspace(0.77, 3.03, 500)
            dense_y = CubicSpline(x_shci, y)(dense_x)
        ax_inset.plot(dense_x, dense_y, lw=3, linestyle="--", c=rgb_to_hex(COLOR_PALETTE[i]))

    ax_inset.set_xlim(0.95, 1.85)
    ax_inset.set_ylim(-6.6, 0.6)
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
    for spine in ax_bottom.spines.values():
        spine.set_linewidth(2)
    ax_bottom.axes.xaxis.set_tick_params(direction="in", which="both", length=5, width=2, right=True)
    ax_bottom.axes.yaxis.set_tick_params(direction="in", which="both", length=6, width=2, right=True, pad=15)
    ax_bottom.set_ylabel(r"Energy - $E_0$ (eV)", fontsize=28, fontweight="regular", labelpad=15)
    plt.yticks(fontsize=28, fontweight="regular")
    ax_bottom.set_xlim(0, 5)
    ax_bottom.set_ylim(-0.5, 4.8)
    ax_bottom.set_xticks([])

    x_fine = np.linspace(x.min(), x.max(), 1000)
    splines = {state: CubicSpline(x, pes_data[state]) for state in LEVEL_STATE_ORDER}
    ground_state = "X1Sg+"
    ground_dense = splines[ground_state](x_fine)
    min_ground = np.min(ground_dense)
    re_ground = x_fine[np.argmin(ground_dense)]

    excitation_energies: dict[str, dict[str, float]] = {}
    for state in LEVEL_STATE_ORDER:
        if state == ground_state:
            continue
        min_excited = np.min(splines[state](x_fine))
        excitation_energies[state] = {
            "adiabatic": (min_excited - min_ground) * HARTREE_TO_EV,
            "vertical": (splines[state](re_ground) - min_ground) * HARTREE_TO_EV,
            "shci": benchmark_data[state]["shci"],
            "exp": benchmark_data[state]["exp"],
        }
        tbe_value = benchmark_data[state]["tbe"]
        if tbe_value is not None:
            excitation_energies[state]["tbe"] = tbe_value

    ground_position = LEVEL_POSITIONS[ground_state]
    ax_bottom.plot([ground_position - 0.4, ground_position + 0.4], [0, 0], color="black", lw=3, zorder=2, alpha=1.0)
    ax_bottom.text(ground_position + 0.45, 0, LEVEL_LABELS[ground_state], ha="left", va="center", fontsize=22)

    for state, energies in excitation_energies.items():
        pos = LEVEL_POSITIONS[state]
        ax_bottom.scatter(
            pos - 0.2,
            energies["vertical"],
            marker="d",
            color="#1E90FF",
            s=180,
            zorder=3,
            edgecolors="black",
            linewidth=0.5,
            alpha=1.0,
        )
        ax_bottom.scatter(
            pos - 0.06,
            energies["adiabatic"],
            marker="o",
            color="#6A0DAD",
            s=180,
            zorder=3,
            edgecolors="black",
            linewidth=0.5,
        )
        ax_bottom.scatter(
            pos + 0.1,
            energies["shci"],
            marker="s",
            color="#2E8B57",
            s=180,
            zorder=3,
            edgecolors="black",
            linewidth=0.5,
        )
        ax_bottom.plot([pos - 0.4, pos + 0.4], [energies["exp"], energies["exp"]], color="black", linestyle="-", lw=3, alpha=0.7)

        if "tbe" in energies:
            tbe_offset = 0.25 if state == "B1Dg" else 0.2
            ax_bottom.scatter(
                pos + tbe_offset,
                energies["tbe"],
                marker="v",
                color="#EC0F0C",
                s=180,
                zorder=3,
                edgecolors="black",
                linewidth=0.5,
            )

        label_offset = -0.95 if state in {"C1Pg", "Bp1Sg+"} else 0.45
        ax_bottom.text(
            pos + label_offset,
            energies["exp"],
            LEVEL_LABELS[state],
            ha="left",
            va="center",
            fontsize=22,
        )

    ax_bottom.legend(
        handles=[
            mlines.Line2D([], [], color="#1E90FF", marker="d", markersize=10, lw=0, label="Vertical"),
            mlines.Line2D([], [], color="#6A0DAD", marker="o", markersize=10, lw=0, label="Adiabatic"),
            mlines.Line2D([], [], color="#2E8B57", marker="s", markersize=10, lw=0, label="SHCI"),
            mlines.Line2D([], [], color="#EC0F0C", marker="v", markersize=10, lw=0, label="TBE CBS QUEST"),
            mlines.Line2D([], [], color="black", linestyle="-", lw=3, label="Experimental"),
        ],
        fontsize=18,
        loc="upper left",
    )
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
