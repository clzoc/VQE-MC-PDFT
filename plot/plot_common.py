"""Shared plotting helpers for reviewer-facing figure scripts."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

SANS_SERIF_FONTS = ["Arial", "Helvetica", "DejaVu Sans"]


def configure_plot_style() -> None:
    """Apply the common sans-serif plotting style used across the repository."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = SANS_SERIF_FONTS
    plt.rc("text", usetex=False)
    plt.rcParams["mathtext.fontset"] = "stixsans"
    plt.rcParams["axes.unicode_minus"] = False


def resolve_output_path(script_file: str, filename: str) -> Path:
    """Resolve an output artifact path relative to the current script file."""
    output_dir = os.environ.get("PLOT_OUTPUT_DIR")
    if output_dir:
        return Path(output_dir).resolve() / filename
    return Path(script_file).resolve().with_name(filename)


def show_figure() -> None:
    """Show a figure only when running with an interactive Matplotlib backend."""
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
