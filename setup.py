from setuptools import setup, find_packages

setup(
    name="virp",
    version="2.2.0",
    packages=find_packages(),
    install_requires=[
        "pymatgen",
        "chgnet",
        "matgl==1.0.0",
        "dgl",
        "scikit-learn",
        "shap",
        "ortools",
        "mace-torch"
    ],
)
