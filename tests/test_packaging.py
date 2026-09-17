"""Public import and package metadata checks; building is a separate CI job."""

import importlib.metadata
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

import vol5dkit


ROOT = Path(__file__).resolve().parents[1]


def test_public_version_matches_project_and_installed_metadata():
    try:
        import tomllib
    except ModuleNotFoundError:
        # Python 3.10 has no stdlib TOML reader. CI installs this checkout, so
        # distribution metadata provides the version generated from pyproject.
        try:
            expected = importlib.metadata.version("vol5dkit")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("install this checkout to verify metadata on Python 3.10")
    else:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        expected = project["version"]
    assert vol5dkit.__version__ == expected
    try:
        installed = importlib.metadata.version("vol5dkit")
    except importlib.metadata.PackageNotFoundError:
        return
    assert installed == expected


def test_core_and_display_data_import_do_not_load_optional_backends():
    code = """
from pathlib import Path
import sys
import torch
import vol5dkit
import vol5dkit.viewer._data
options = vol5dkit.Display(torch.zeros(1, 1, 1, 1, 1), name="test", window=(0, 1))

assert Path(vol5dkit.__file__).resolve() == Path(sys.argv[1]).resolve()
for name in ('voltensor', 'SimpleITK', 'nibabel', 'monai', 'nrrd', 'torchio',
             'PySide6', 'vispy', 'OpenGL', 'matplotlib', 'seaborn'):
    assert name not in sys.modules, name
assert not torch.cuda.is_initialized()
assert callable(vol5dkit.view) and options.window == (0, 1)
assert not hasattr(vol5dkit, 'run')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "src/vol5dkit/__init__.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_installed_core_dependencies_exclude_optional_backends():
    try:
        metadata = importlib.metadata.metadata("vol5dkit")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("install this checkout to inspect its distribution metadata")
    requirements = metadata.get_all("Requires-Dist", [])
    core = {
        re.match(r"[\w.-]+", requirement).group(0).lower()
        for requirement in requirements
        if "extra ==" not in requirement
    }
    assert core == {"torch", "numpy"}
    assert "gui" in metadata.get_all("Provides-Extra", [])
