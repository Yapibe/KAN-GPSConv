from setuptools import setup, find_packages

setup(
    name="KAN-GPSConv",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch-geometric",
        "numpy",
        "scikit-learn",
        "matplotlib",
    ],
)
