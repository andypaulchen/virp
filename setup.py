from setuptools import setup, find_packages

setup(
    name="virp",
    version="1.2.1",
    packages=find_packages(),
    install_requires=[
        "pymatgen",
        "chgnet",
        "matgl==1.0.0",
        "dgl",
        "poshcar>=2.0.2"
    ],
)
