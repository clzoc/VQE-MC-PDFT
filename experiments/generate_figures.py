"""Generate publication-quality figures for the PNAS manuscript.

Reads pre-computed data from ``data/`` CSV files and produces Figs. 2-7
matching the manuscript. Figures are saved to ``figures/``.

Usage:
    python experiments/generate_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Isolated atom energies (Ha) for binding energy conversion
C_ATOM_ENERGY = -37.8456944620  # VQE-MC-PDFT cc-pVTZ
CR_ATOM_ENERGY_SV = -1043.2084407135  # Ahlrichs-SV

HA_TO_EV = 27.2114


def fig2_c2_ground_state():
    """Fig. 2: C2 ground-state PEC and error analysis."""
    df = pd.read_csv(DATA_DIR / "c2_ground_state_pec.csv")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    r = df["bond_A"]
    methods = {
        "HF": ("HF", "gray", "--"),
        "DFT_PBE": ("DFT/PBE", "orange", "--"),
        "CASSCF": ("CASSCF", "blue", "--"),
        "VQE_SA_CASSCF": ("VQE-SA-CASSCF", "green", "-"),
        "VQE_MC_PDFT": ("VQE-MC-PDFT", "purple", "-"),
    }

    for col, (label, color, ls) in methods.items():
        if col in df.columns:
            be = (df[col] - 2 * C_ATOM_ENERGY) * HA_TO_EV
            ax1.plot(r, be, label=label, color=color, linestyle=ls)

    if "SHCI" in df.columns:
        be_shci = (df["SHCI"] - 2 * C_ATOM_ENERGY) * HA_TO_EV
        ax1.plot(r, be_shci, "rd", label="Ref. SHCI", markersize=4)
        # Deviation plot
        for col, (label, color, ls) in methods.items():
            if col in df.columns:
                be = (df[col] - 2 * C_ATOM_ENERGY) * HA_TO_EV
                ax2.plot(r, be - be_shci, label=label, color=color, linestyle=ls)
        ax2.axhline(0, color="red", linewidth=0.5)

    ax1.set_ylabel("Binding Energy (eV)")
    ax1.legend(fontsize=7)
    ax1.set_title("(a)")
    ax2.set_xlabel("Bond distance (A)")
    ax2.set_ylabel("Deviation (eV)")
    ax2.set_title("(b)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_c2_ground_state.pdf", dpi=300)
    plt.close()


def fig3_c2_excited_states():
    """Fig. 3: C2 excited-state PECs."""
    df = pd.read_csv(DATA_DIR / "c2_excited_state_pec.csv")
    fig, ax = plt.subplots(figsize=(6, 5))

    r = df["bond_A"]
    states = [c for c in df.columns if c != "bond_A"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(states)))

    for state, color in zip(states, colors):
        be = (df[state] - 2 * C_ATOM_ENERGY) * HA_TO_EV
        ax.plot(r, be, label=state, color=color)

    ax.set_xlabel("Bond distance (A)")
    ax.set_ylabel("Energy (eV)")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("C2 excited-state PECs (VQE-MC-PDFT)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_c2_excited_states.pdf", dpi=300)
    plt.close()


def fig4_cr2_active_space():
    """Fig. 4: Cr2 active-space scaling."""
    df = pd.read_csv(DATA_DIR / "cr2_active_space.csv")
    fig, ax = plt.subplots(figsize=(6, 5))

    r = df["bond_A"]
    styles = {
        "12e12o": ("s", "C0"), "12e22o": ("o", "C1"), "12e28o": ("^", "C2"),
    }
    for suffix, (marker, color) in styles.items():
        cas_col = f"CASSCF_{suffix}"
        pdft_col = f"MCPDFT_{suffix}"
        if cas_col in df.columns:
            # Compute binding energy relative to 2 * atom energy at largest R
            atom_e = df[cas_col].iloc[-1] / 2.0
            be_cas = (df[cas_col] - 2 * atom_e) * HA_TO_EV
            ax.plot(r, be_cas, marker=marker, color=color, linestyle="--",
                    fillstyle="none", label=f"VQE-CASSCF({suffix})", markersize=5)
        if pdft_col in df.columns:
            atom_e = df[pdft_col].iloc[-1] / 2.0
            be_pdft = (df[pdft_col] - 2 * atom_e) * HA_TO_EV
            ax.plot(r, be_pdft, marker=marker, color=color, linestyle="-",
                    label=f"VQE-MC-PDFT({suffix})", markersize=5)

    ax.set_xlabel("Bond distance (A)")
    ax.set_ylabel("Binding Energy (eV)")
    ax.legend(fontsize=6)
    ax.set_title("Cr2 active-space scaling")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_cr2_active_space.pdf", dpi=300)
    plt.close()


def fig5_cr2_basis_set():
    """Fig. 5: Cr2 basis-set convergence."""
    df = pd.read_csv(DATA_DIR / "cr2_basis_set.csv")
    fig, ax = plt.subplots(figsize=(6, 5))

    r = df["bond_A"]
    for label, color in [("TZ", "C0"), ("QZ", "C1"), ("5Z", "C2")]:
        col = f"MCPDFT_{label}"
        if col in df.columns:
            atom_e = df[col].iloc[-1] / 2.0
            be = (df[col] - 2 * atom_e) * HA_TO_EV
            ax.plot(r, be, "o-", label=f"VQE-MC-PDFT {label}", color=color, markersize=4)

    ax.set_xlabel("Bond distance (A)")
    ax.set_ylabel("Binding Energy (eV)")
    ax.legend()
    ax.set_title("Cr2 basis-set convergence (12e,22o)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_cr2_basis_set.pdf", dpi=300)
    plt.close()


def fig6_cr2_boxplot():
    """Fig. 6: Cr2 binding energy distributions with qubit scaling."""
    df = pd.read_csv(DATA_DIR / "cr2_boxplot_raw.csv")
    fig, ax = plt.subplots(figsize=(6, 5))

    qubit_cols = [c for c in df.columns if c.startswith("qubits_")]
    data = []
    labels = []
    for col in sorted(qubit_cols):
        n_orb = int(col.split("_")[1])
        n_qubits = 2 * n_orb
        be = (df[col] - 2 * CR_ATOM_ENERGY_SV) * HA_TO_EV
        data.append(be.values)
        labels.append(f"{n_qubits}")

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black", markersize=5))
    colors = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Reference lines
    ref_mps = (-2086.43490 - 2 * CR_ATOM_ENERGY_SV) * HA_TO_EV
    ref_shci = (-2086.44456 - 2 * CR_ATOM_ENERGY_SV) * HA_TO_EV
    ax.axhline(ref_mps, color="blue", linestyle="--", label="MPS-LCC")
    ax.axhline(ref_shci, color="orange", linestyle="--", label="SHCI")

    ax.set_xlabel("Qubits Utilization")
    ax.set_ylabel("Binding Energy (eV)")
    ax.legend(fontsize=8)
    ax.set_title("Cr2 at 1.50 A (48e, 42o)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_cr2_boxplot.pdf", dpi=300)
    plt.close()


def fig7_benzene():
    """Fig. 7: Benzene vertical excitation energies."""
    df = pd.read_csv(DATA_DIR / "benzene_excitations.csv")
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(df))
    width = 0.2

    for i, (col, label, color) in enumerate([
        ("TBE_CBS_eV", "TBE CBS QUEST", "black"),
        ("TDDFT_eV", "TDDFT", "purple"),
        ("NES_VMC_eV", "NES-VMC/Psiformer", "blue"),
        ("VQE_MCPDFT_eV", "VQE-MC-PDFT", "green"),
    ]):
        if col in df.columns:
            ax.bar(x + i * width, df[col], width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(df["transition"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Excitation Energy (eV)")
    ax.legend(fontsize=7)
    ax.set_title("Benzene vertical pi->pi* excitations")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_benzene.pdf", dpi=300)
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generating Fig. 2: C2 ground-state PEC...")
    fig2_c2_ground_state()
    print("Generating Fig. 3: C2 excited-state PECs...")
    fig3_c2_excited_states()
    print("Generating Fig. 4: Cr2 active-space scaling...")
    fig4_cr2_active_space()
    print("Generating Fig. 5: Cr2 basis-set convergence...")
    fig5_cr2_basis_set()
    print("Generating Fig. 6: Cr2 boxplot...")
    fig6_cr2_boxplot()
    print("Generating Fig. 7: Benzene excitations...")
    fig7_benzene()
    print(f"All figures saved to {FIG_DIR}/")
