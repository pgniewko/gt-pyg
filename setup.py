from setuptools import setup, find_packages

exec(open("gt_pyg/_version.py").read())

setup(
    name="gt_pyg",
    version=__version__,
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
    },
    include_package_data=True,
)
