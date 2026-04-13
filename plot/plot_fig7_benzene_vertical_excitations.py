"""Plot benzene vertical excitation energies for the reviewer package."""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from plot_common import configure_plot_style, resolve_output_path, show_figure

DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "benzene_excitations.csv"
OUTPUT_PDF = resolve_output_path(__file__, "fig7_benzene_vertical_excitations.pdf")

STATE_ORDER = ["3B1u", "3E1u", "1B2u", "3B2u", "1B1u"]
TRANSITION_TO_STATE = {
    "11A1g_13B1u": "3B1u",
    "11A1g_13E1u": "3E1u",
    "11A1g_11B2u": "1B2u",
    "11A1g_13B2u": "3B2u",
    "11A1g_11B1u": "1B1u",
}
STATE_LABELS = {
    "3B1u": r"$^3B_{1u}$",
    "3E1u": r"$^3E_{1u}$",
    "1B2u": r"$^1B_{2u}$",
    "3B2u": r"$^3B_{2u}$",
    "1B1u": r"$^1B_{1u}$",
}
STATE_POSITIONS = {name: i + 1 for i, name in enumerate(STATE_ORDER)}
PLOT_DATA_STYLE = {
    "TDDFT aug-cc-pVTZ": {"column": "TDDFT_eV", "marker": "p", "color": "#845EC2"},
    "NES-VMC/Psiformer": {"column": "NES_VMC_eV", "marker": "s", "color": "#2E8B57"},
    "VQE-MCPDFT aug-cc-pVTZ": {"column": "VQE_MCPDFT_eV", "marker": "^", "color": "#0F07F1"},
}


def main() -> None:
    configure_plot_style()

    benzene_df = pd.read_csv(DATA_CSV)
    benzene_df["state"] = benzene_df["transition"].map(TRANSITION_TO_STATE)
    benzene_df = benzene_df.set_index("state").loc[STATE_ORDER]

    fig, ax = plt.subplots(figsize=(9, 7))
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.xaxis.set_tick_params(direction="in", which="both", length=5, width=2, right=True)
    ax.axes.yaxis.set_tick_params(direction="in", which="both", length=6, width=2, right=True, pad=15)
    ax.set_ylabel(r"Energy - $E_0$ (eV)", fontsize=28, fontweight="regular", labelpad=15)
    plt.yticks([4.0, 5.0, 6.0, 7.0], fontsize=28, fontweight="regular")
    ax.yaxis.set_ticklabels([r"$4.0$", r"$5.0$", r"$6.0$", r"$7.0$"])

    for state in STATE_ORDER:
        x_pos = STATE_POSITIONS[state]
        energy = benzene_df.loc[state, "TBE_CBS_eV"]
        ax.plot([x_pos - 0.4, x_pos + 0.4], [energy, energy], color="black", linestyle="-", lw=2.4, zorder=1)
        offset = -0.90 if state == "1B1u" else 0.45
        ax.text(x_pos + offset, energy, STATE_LABELS[state], ha="left", va="center", fontsize=26)

    for i, (method_name, details) in enumerate(PLOT_DATA_STYLE.items()):
        x_coords = []
        y_coords = []
        for state in STATE_ORDER:
            x_coords.append(STATE_POSITIONS[state] - 0.2 + i * 0.2)
            y_coords.append(benzene_df.loc[state, details["column"]])
        ax.scatter(
            x_coords,
            y_coords,
            marker=details["marker"],
            color=details["color"],
            s=300,
            label=method_name,
            zorder=2,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_ylim(3.9, 7.1)
    ax.set_xlim(0.5, len(STATE_ORDER) + 0.5)
    ax.set_xticks([])
    ax.set_xticklabels([])

    legend_handles = [mlines.Line2D([], [], color="black", linestyle="-", lw=2.4, label="TBE CBS QUEST")]
    for method_name, details in PLOT_DATA_STYLE.items():
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=details["color"],
                marker=details["marker"],
                linestyle="None",
                markersize=12,
                label=method_name,
            )
        )
    ax.legend(handles=legend_handles, loc="upper left", prop={"size": 18})

    def on_resize(event) -> None:
        fig.tight_layout()
        fig.canvas.draw()

    plt.tight_layout()
    fig.canvas.mpl_connect("resize_event", on_resize)
    fig.savefig(OUTPUT_PDF, dpi=fig.dpi, bbox_inches="tight")
    show_figure()


if __name__ == "__main__":
    main()
