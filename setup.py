from setuptools import setup, find_packages
import os
import re
import subprocess


def _normalize_prerelease(ver):
    """Convert common pre-release tag formats to PEP 440."""
    ver = re.sub(r"[-.]?alpha[.-]?", "a", ver)
    ver = re.sub(r"[-.]?beta[.-]?", "b", ver)
    ver = re.sub(r"[-.]?rc[.-]?", "rc", ver)
    return ver


def _get_version_from_git():
    """Get version from git describe.

    At install time importlib.metadata can return stale data from a
    previous installation, so setup.py must resolve the version
    directly from git.
    """
    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            ["git", "describe", "--tags", "--long"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        desc = result.stdout.strip().lstrip("v")
        parts = desc.rsplit("-", 2)
        if len(parts) != 3 or not parts[2].startswith("g"):
            raise RuntimeError(f"Cannot parse git describe output: {desc!r}")

        ver, distance, sha = parts[0], parts[1], parts[2][1:]
        ver = _normalize_prerelease(ver)

        if int(distance) == 0:
            return ver
        return f"{ver}.dev{distance}+{sha}"

    except Exception:
        return "unknown"


setup(
    name="gt_pyg",
    version=_get_version_from_git(),
    description="Implementation of the Graph Transformer architecture in Pytorch-geometric",
    packages=find_packages(exclude=["tests*", "*.tests", "*.tests.*"]),
    install_requires=[
        "torch>=1.13.0",
        "torch_geometric",
        "numpy",
        "rdkit",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
        "examples": [
            "notebook",
            "ipykernel",
            "ipywidgets",
            "pandas",
            "matplotlib",
            "seaborn",
            "scikit-learn",
            "scipy",
        ],
        "all": [
            "gt_pyg[dev]",
            "gt_pyg[examples]",
        ],
    },
    include_package_data=True,
)
