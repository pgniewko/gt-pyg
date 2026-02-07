from setuptools import setup, find_packages
import os

_version_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_pyg", "_version.py")
_ns = {"__file__": _version_path}
exec(open(_version_path).read(), _ns)

setup(
    name="gt_pyg",
    version=_ns["__version__"],
    description="Implementation of the Graph Transformer architecture in Pytorch-geometric",
    packages=find_packages(exclude=["tests*", "*.tests", "*.tests.*"]),
    install_requires=[
        "torch>=1.13.0",
        "torch_geometric",
        "numpy",
        "pandas",
        "rdkit",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    include_package_data=True,
)
