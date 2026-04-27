from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from setuptools import find_packages, setup


def _load_version_utils():
    version_utils_path = (
        Path(__file__).resolve().parent / "gt_pyg" / "_version_utils.py"
    )
    spec = spec_from_file_location("gt_pyg_setup_version_utils", version_utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load version helpers from {version_utils_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    PACKAGE_VERSION = _load_version_utils()._get_version_from_git()
except Exception:
    PACKAGE_VERSION = "0+unknown"


setup(
    name="gt_pyg",
    version=PACKAGE_VERSION,
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
        "chembl": ["chembl_structure_pipeline"],
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
            "gt_pyg[chembl]",
            "gt_pyg[examples]",
        ],
    },
    include_package_data=True,
)
