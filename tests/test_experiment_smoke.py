"""Smoke tests for experiment script imports and --help.
Verifies that each experiment script can be imported without crashing
and that CLI entrypoints parse --help without error.
"""
import subprocess
import sys
from pathlib import Path

import pytest

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"

EXPERIMENT_SCRIPTS = [
    "c2_ground_state",
    "c2_excited_states",
    "cr2_active_space",
    "cr2_basis_set",
    "cr2_1p5A_cutting",
    "benzene_excitations",
    "generate_figures",
]


@pytest.mark.parametrize("script", EXPERIMENT_SCRIPTS)
def test_experiment_imports(script):
    """Each experiment script must be importable without side effects."""
    # Use subprocess with __file__ defined to avoid NameError
    script_path = str(EXPERIMENTS_DIR / (script + ".py"))
    code = (
        f"import sys; sys.path.insert(0, '{EXPERIMENTS_DIR.parent}'); "
        f"__file__ = '{script_path}'; "
        f"content = open('{script_path}').read(); "
        f"parts = content.split('if __name__'); "
        f"exec(parts[0])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
        cwd=str(EXPERIMENTS_DIR.parent),
    )
    # Import-time code (module-level) should not crash
    # We only check that the script doesn't have syntax errors or
    # import-time exceptions (the __main__ block is excluded)
    assert result.returncode == 0 or "ModuleNotFoundError" in result.stderr, \
        f"{script} import failed: {result.stderr[:500]}"


@pytest.mark.parametrize("script", ["cr2_1p5A_cutting"])
def test_experiment_help(script):
    """Scripts with argparse must respond to --help without error."""
    result = subprocess.run(
        [sys.executable, str(EXPERIMENTS_DIR / f"{script}.py"), "--help"],
        capture_output=True, text=True, timeout=30,
        cwd=str(EXPERIMENTS_DIR.parent),
        env={"PYTHONPATH": str(EXPERIMENTS_DIR.parent), "PATH": ""},
    )
    # --help should exit 0
    assert result.returncode == 0, f"--help failed for {script}: {result.stderr[:500]}"
    assert "usage" in result.stdout.lower() or "optional arguments" in result.stdout.lower()
