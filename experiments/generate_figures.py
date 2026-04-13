"""Generate figure PDFs by running the dedicated scripts under ``plot/``.

Usage:
    python experiments/generate_figures.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT_DIR / "figures"
PLOT_DIR = ROOT_DIR / "plot"

PLOT_SCRIPTS = [
    "plot_fig2_c2_ground_state.py",
    "plot_fig3_c2_excited_states.py",
    "plot_fig4_cr2_active_space.py",
    "plot_fig5_cr2_basis_set.py",
    "plot_fig6_cr2_qubit_utilization.py",
    "plot_fig7_benzene_vertical_excitations.py",
    "plot_supp_tianji_s2_hardware_fidelity.py",
]


def run_plot_script(script_name: str) -> None:
    script_path = PLOT_DIR / script_name
    env = os.environ.copy()
    env["MPLBACKEND"] = env.get("MPLBACKEND", "Agg")
    env["PLOT_OUTPUT_DIR"] = str(FIG_DIR)
    subprocess.run([sys.executable, str(script_path)], check=True, env=env, cwd=ROOT_DIR)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for script_name in PLOT_SCRIPTS:
        print(f"Running {script_name}...")
        run_plot_script(script_name)
    print(f"All figures saved to {FIG_DIR}/")
